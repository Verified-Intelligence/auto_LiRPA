""" Activation operators or other unary nonlinear operators"""
import torch
from torch import Tensor
from collections import OrderedDict
from .base import *
from .clampmult import multiply_by_A_signs

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class BoundActivation(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.requires_input_bounds = [0]
        self.relaxed = False
        self.use_default_ibp = True

    def _init_masks(self, x):
        self.mask_pos = x.lower >= 0
        self.mask_neg = x.upper <= 0
        self.mask_both = torch.logical_not(torch.logical_or(self.mask_pos, self.mask_neg))

    def init_linear_relaxation(self, x, dim_opt=None):
        self._init_masks(x)
        self.lw = torch.zeros_like(x.lower)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()

    def add_linear_relaxation(self, mask, type, k, x0, y0):
        if type == 'lower':
            w_out, b_out = self.lw, self.lb
        else:
            w_out, b_out = self.uw, self.ub

        if mask is None:
            if isinstance(k, Tensor) and k.ndim > 0:
                w_out[:] = k
            else:
                w_out.fill_(k)
        else:
            if isinstance(k, Tensor):
                w_out[..., mask] = k[..., mask].to(w_out)
            else:
                w_out[..., mask] = k

        if (not isinstance(x0, Tensor) and x0 == 0
                and not isinstance(y0, Tensor) and y0 == 0):
            pass
        else:
            b = -x0 * k + y0
            if mask is None:
                if b.ndim > 0:
                    b_out[:] = b
                else:
                    b_out.fill_(b)
            else:
                b_out[..., mask] = b[..., mask]

    def bound_relax(self, x):
        return not_implemented_op(self, 'bound_relax')

    def bound_backward(self, last_lA, last_uA, x):
        if not self.relaxed:
            self.init_linear_relaxation(x)
            self.bound_relax(x)

        def _bound_oneside(last_A, sign=-1):
            if last_A is None:
                return None, 0
            if sign == -1:
                w_pos, b_pos, w_neg, b_neg = (
                    self.lw.unsqueeze(0), self.lb.unsqueeze(0),
                    self.uw.unsqueeze(0), self.ub.unsqueeze(0))
            else:
                w_pos, b_pos, w_neg, b_neg = (
                    self.uw.unsqueeze(0), self.ub.unsqueeze(0),
                    self.lw.unsqueeze(0), self.lb.unsqueeze(0))
            w_pos = maybe_unfold_patches(w_pos, last_A)
            w_neg = maybe_unfold_patches(w_neg, last_A)
            b_pos = maybe_unfold_patches(b_pos, last_A)
            b_neg = maybe_unfold_patches(b_neg, last_A)
            if self.batch_dim == 0:
                _A, _bias = multiply_by_A_signs(last_A, w_pos, w_neg, b_pos, b_neg)
            elif self.batch_dim == -1:
                # FIXME: why this is different from above?
                mask = torch.gt(last_A, 0.).to(torch.float)
                _A = last_A * (mask * w_pos.unsqueeze(1) +
                               (1 - mask) * w_neg.unsqueeze(1))
                _bias = last_A * (mask * b_pos.unsqueeze(1) +
                                  (1 - mask) * b_neg.unsqueeze(1))
                if _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=list(range(2, _bias.ndim)))
            else:
                raise NotImplementedError

            return _A, _bias

        lA, lbias = _bound_oneside(last_lA, sign=-1)
        uA, ubias = _bound_oneside(last_uA, sign=+1)

        return [(lA, uA)], lbias, ubias

    @staticmethod
    @torch.jit.script
    def bound_forward_w(
            relax_lw: Tensor, relax_uw: Tensor, x_lw: Tensor, x_uw: Tensor, dim: int):
        lw = (relax_lw.unsqueeze(dim).clamp(min=0) * x_lw +
              relax_lw.unsqueeze(dim).clamp(max=0) * x_uw)
        uw = (relax_uw.unsqueeze(dim).clamp(max=0) * x_lw +
              relax_uw.unsqueeze(dim).clamp(min=0) * x_uw)
        return lw, uw

    @staticmethod
    @torch.jit.script
    def bound_forward_b(
            relax_lw: Tensor, relax_uw: Tensor, relax_lb: Tensor,
            relax_ub: Tensor, x_lb: Tensor, x_ub: Tensor):
        lb = relax_lw.clamp(min=0) * x_lb + relax_lw.clamp(max=0) * x_ub + relax_lb
        ub = relax_uw.clamp(max=0) * x_lb + relax_uw.clamp(min=0) * x_ub + relax_ub
        return lb, ub

    def bound_forward(self, dim_in, x):
        if not self.relaxed:
            self.init_linear_relaxation(x)
            self.bound_relax(x)

        assert (x.lw is None) == (x.uw is None)

        dim = 1 if self.lw.ndim > 0 else 0

        if x.lw is not None:
            lw, uw = BoundActivation.bound_forward_w(self.lw, self.uw, x.lw, x.uw, dim)
        else:
            lw = uw = None
        lb, ub = BoundActivation.bound_forward_b(self.lw, self.uw, self.lb, self.ub, x.lb, x.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        return self.forward(h_L), self.forward(h_U)


class BoundOptimizableActivation(BoundActivation):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        # Stages:
        #   * `init`: initializing parameters
        #   * `opt`: optimizing parameters
        #   * `reuse`: not optimizing parameters but reuse saved values
        # If `None`, it means activation optimization is currently not used.
        self.opt_stage = None
        self.alpha = OrderedDict()
        # Save patch sizes during bound_backward() for each output_node.
        self.patch_size = {}
        # Location of batch dimension in self.alpha. Must be set by children.
        self.alpha_batch_dim = None
        # A torch.bool mask of shape Tensor([batch_size]) that conditions the sample of alpha and beta to update
        # If set to None, update all samples
        # If not None, select those corresponding to 1 to update
        self.alpha_beta_update_mask = None

    def opt_init(self):
        """Enter the stage for initializing bound optimization. Optimized bounds
        are not used in this stage."""
        self.opt_stage = 'init'

    def opt_start(self):
        """Start optimizing bounds."""
        self.opt_stage = 'opt'

    def opt_reuse(self):
        """ Reuse optimizing bounds """
        self.opt_stage = 'reuse'

    def opt_no_reuse(self):
        """ Finish reusing optimized bounds """
        if self.opt_stage == 'reuse':
            self.opt_stage = None

    def opt_end(self):
        """ End optimizing bounds """
        self.opt_stage = None

    def init_opt_parameters(self, start_nodes):
        """ start_nodes: a list of starting nodes [(node, size)] during
        CROWN backward bound propagation"""
        raise NotImplementedError

    def clip_alpha_(self):
        pass

    def init_linear_relaxation(self, x, dim_opt=None):
        self._init_masks(x)
        # The first dimension of size 2 is used for lA and uA respectively,
        # when computing intermediate bounds.
        if self.opt_stage in ['opt', 'reuse'] and dim_opt is not None:
            # For optimized bounds, we have independent lw for each output dimension for bound optimization.
            # If the output layer is a fully connected layer, len(dim_opt) = 1.
            # If the output layer is a conv layer, len(dim_opt) = 3 but we only use the out_c dimension to create slopes/bias.
            # Variables are shared among out_h, out_w dimensions so far.
            dim = dim_opt if isinstance(dim_opt, int) else dim_opt[0]
            self.lw = torch.zeros(2, dim, *x.lower.shape).to(x.lower)
        else:
            # Without optimized bounds, the lw, lb (slope, biase) etc only depend on intermediate layer bounds,
            # and are shared among different output dimensions.
            self.lw = torch.zeros_like(x.lower)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()

    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None):
        self._start = start_node.name

        if self.opt_stage not in ['opt', 'reuse']:
            last_A = last_lA if last_lA is not None else last_uA
            # Returned [(lA, uA)], lbias, ubias
            As, lbias, ubias = super().bound_backward(last_lA, last_uA, x)
            if isinstance(last_A, Patches):
                A_prod = As[0][1].patches if As[0][0] is None else As[0][1].patches
                # FIXME: Unify this function with BoundReLU
                # Save the patch size, which will be used in init_slope() to determine the number of optimizable parameters.
                if start_node is not None:
                    if last_A.unstable_idx is not None:
                        # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                        self.patch_size[start_node.name] = [last_A.output_shape[1], A_prod.size(1), last_A.output_shape[2], last_A.output_shape[3], A_prod.size(-3), A_prod.size(-2), A_prod.size(-1)]
                    else:
                        # Regular patches.
                        self.patch_size[start_node.name] = A_prod.size()
            return As, lbias, ubias
        assert self.batch_dim == 0

        if not self.relaxed:
            self.init_linear_relaxation(x, dim_opt=start_shape)
            self.bound_relax(x)

        def _bound_oneside(last_A, sign=-1):
            if last_A is None:
                return None, 0
            if sign == -1:
                w_pos, b_pos, w_neg, b_neg = self.lw[0], self.lb[0], self.uw[0], self.ub[0]
            else:
                w_pos, b_pos, w_neg, b_neg = self.uw[1], self.ub[1], self.lw[1], self.lb[1]
            w_pos = maybe_unfold_patches(w_pos, last_A)
            w_neg = maybe_unfold_patches(w_neg, last_A)
            b_pos = maybe_unfold_patches(b_pos, last_A)
            b_neg = maybe_unfold_patches(b_neg, last_A)
            A_prod, _bias = multiply_by_A_signs(last_A, w_pos, w_neg, b_pos, b_neg)
            return A_prod, _bias

        lA, lbias = _bound_oneside(last_lA, sign=-1)
        uA, ubias = _bound_oneside(last_uA, sign=+1)

        return [(lA, uA)], lbias, ubias

    def _no_bound_parameters(self):
        raise AttributeError('Bound parameters have not been initialized.'
                             'Please call `compute_bounds` with `method=CROWN-optimized`'
                             ' at least once.')

    def dump_optimized_params(self):
        raise NotImplementedError

    def restore_optimized_params(self):
        raise NotImplementedError

    def set_alpha_beta_update_mask(self, mask):
        self.alpha_beta_update_mask = mask

    def clean_alpha_beta_update_mask(self):
        self.alpha_beta_update_mask = None
