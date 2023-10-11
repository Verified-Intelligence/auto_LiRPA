""" Bivariate operators"""
import torch
from torch import Tensor
from typing import Dict, Optional
from .base import *
from .activation_base import BoundOptimizableActivation
from .nonlinear import BoundSqrt
from .clampmult import multiply_by_A_signs
from ..utils import *
from .solver_utils import grb


class MulHelper:
    """Handle linear relaxation for multiplication.

    This helper can be used by BoundMul, BoundMatMul,
    BoundLinear (with weight perturbation).
    """

    def __init__(self):
        pass

    @staticmethod
    def interpolated_relaxation(x_l: Tensor, x_u: Tensor,
                                y_l: Tensor, y_u: Tensor,
                                r_l: Optional[Tensor] = None,
                                r_u: Optional[Tensor] = None
                               ) -> Tuple[Tensor, Tensor, Tensor,
                                          Tensor, Tensor, Tensor]:
        """Interpolate two optimal linear relaxations for optimizable bounds."""
        if r_l is None and r_u is None:
            alpha_l, beta_l, gamma_l = y_l, x_l, -y_l * x_l
            alpha_u, beta_u, gamma_u = y_u, x_l, -y_u * x_l
            return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u
        else:
            assert isinstance(r_l, Tensor) and isinstance(r_u, Tensor)
            # TODO (for zhouxing/qirui): this function may benefit from JIT,
            # because it has many element-wise operation which can be fused.
            # Need to benchmark and see performance.
            alpha_l = (y_l - y_u) * r_l + y_u
            beta_l = (x_l - x_u) * r_l + x_u
            gamma_l = (y_u * x_u - y_l * x_l) * r_l - y_u * x_u
            alpha_u = (y_u - y_l) * r_u + y_l
            beta_u = (x_l - x_u) * r_u + x_u
            gamma_u = (y_l * x_u - y_u * x_l) * r_u - y_l * x_u
            return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    @staticmethod
    def get_relaxation(x_l: Tensor, x_u: Tensor, y_l: Tensor, y_u: Tensor,
                       opt_stage: Optional[str],
                       alphas: Optional[Dict[str, Tensor]],
                       start_name: Optional[str],
                      ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if opt_stage in ['opt', 'reuse']:
            assert x_l.ndim == y_l.ndim
            ns = start_name
            alphas[ns].data[:] = alphas[ns].data[:].clamp(min=0, max=1)
            return MulHelper.interpolated_relaxation(
                x_l, x_u, y_l, y_u, alphas[ns][:2], alphas[ns][2:4])
        else:
            return MulHelper.interpolated_relaxation(x_l, x_u, y_l, y_u)

    @staticmethod
    def get_forward_relaxation(x_l, x_u, y_l, y_u, opt_stage, alpha, start_name):
        # Broadcast
        # FIXME perhaps use a more efficient way
        x_l = x_l + torch.zeros_like(y_l)
        x_u = x_u + torch.zeros_like(y_u)
        y_l = y_l + torch.zeros_like(x_l)
        y_u = y_u + torch.zeros_like(x_u)
        return MulHelper.get_relaxation(x_l, x_u, y_l, y_u, opt_stage, alpha, start_name)

    @staticmethod
    def _get_gap(x, y, alpha, beta):
        return x * y - alpha * x - beta * y


class BoundMul(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True
        self.mul_helper = MulHelper()

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    def get_relaxation_opt(self, x_l, x_u, y_l, y_u):
        return self.mul_helper.get_relaxation(
            x_l, x_u, y_l, y_u, self.opt_stage, getattr(self, 'alpha', None),
            self._start)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        x_l = self.inputs[0].lower
        y_l = self.inputs[1].lower
        assert x_l.ndim == y_l.ndim
        shape = [max(x_l.shape[i], y_l.shape[i]) for i in range(x_l.ndim)]
        alpha = torch.ones(4, size_spec, *shape, device=x_l.device)
        return alpha

    def bound_relax(self, x, y, init=False, dim_opt=None):
        if init:
            pass
        (alpha_l, beta_l, gamma_l,
         alpha_u, beta_u, gamma_u) = self.get_relaxation_opt(
            x.lower, x.upper, y.lower, y.upper)
        self.lw = [alpha_l, beta_l]
        self.lb = gamma_l
        self.uw = [alpha_u, beta_u]
        self.ub = gamma_u

    @staticmethod
    def _multiply_by_const(x, const):
        if isinstance(x, torch.Tensor):
            return x * const
        elif isinstance(x, Patches):
            # Multiplies patches by a const. Assuming const is a tensor, and it must be in nchw format.
            assert isinstance(const, torch.Tensor) and const.ndim == 4
            if const.size(0) == x.patches.size(1) and const.size(1) == x.patches.size(-3) and const.size(2) == const.size(3) == 1:
                # The case that we can do channel-wise broadcasting multiplication
                # Shape of const: (batch, in_c, 1, 1)
                # Shape of patches when unstable_idx is None: (spec, batch, in_c, patch_h, patch_w)
                # Shape of patches when unstable_idx is not None: (out_c, batch, out_h, out_w, in_c, patch_h, patch_w)
                const_reshaped = const
            else:
                assert x.unstable_idx is None and (x.padding == 0 or x.padding == [0,0,0,0]) and x.stride == 1 and x.patches.size(-1) == x.patches.size(-2) == 1
                # The assumed dimension is (out_c, N, out_h, out_w, in_c, 1, 1) with padding =1 and stride = 0.
                # In this special case we can directly multiply.
                # After reshape it is (1, N, H, W, C, 1, 1)
                const_reshaped = const.permute(0, 2, 3, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return x.create_similar(x.patches * const_reshaped)
        else:
            raise ValueError(f'Unsupported x type {type(x)}')

    def bound_backward_constant(self, last_lA, last_uA, x, y, op=None,
                                reduce_bias=True, **kwargs):
        assert reduce_bias
        op = BoundMul._multiply_by_const if op is None else op
        # Handle the case of multiplication by a constant.
        factor = None
        if x.perturbed:
            factor = y.forward_value
        if y.perturbed:
            factor = x.forward_value
        # No need to compute A matrix if it is Constant.
        lAx = (None if not x.perturbed or last_lA is None
               else self.broadcast_backward(op(last_lA, factor), x))
        uAx = (None if not x.perturbed or last_uA is None
               else self.broadcast_backward(op(last_uA, factor), x))
        lAy = (None if not y.perturbed or last_lA is None
               else self.broadcast_backward(op(last_lA, factor), y))
        uAy = (None if not y.perturbed or last_uA is None
               else self.broadcast_backward(op(last_uA, factor), y))
        return [(lAx, uAx), (lAy, uAy)], 0., 0.

    def bound_backward(self, last_lA, last_uA, x, y, start_node=None, **kwargs):
        if start_node is not None:
            self._start = start_node.name
        if self.is_linear_op:
            ret = self.bound_backward_constant(last_lA, last_uA, x, y, **kwargs)
        else:
            ret = self.bound_backward_both_perturbed(
                last_lA, last_uA, x, y, **kwargs)
        return ret

    def bound_backward_both_perturbed(self, last_lA, last_uA, x, y,
                                      reduce_bias=True, **kwargs):
        self.bound_relax(x, y)

        def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos,
                           alpha_neg, beta_neg, gamma_neg, opt=False):
            if last_A is None:
                return None, None, 0

            if type(last_A) == Patches:
                assert reduce_bias
                assert last_A.identity == 0
                # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W].
                # Here out_c is the spec dimension.
                # for patches mode, we need to unfold the alpha_pos/neg and beta_pos/neg
                alpha_pos = maybe_unfold_patches(alpha_pos, last_A)
                alpha_neg = maybe_unfold_patches(alpha_neg, last_A)
                beta_pos = maybe_unfold_patches(beta_pos, last_A)
                beta_neg = maybe_unfold_patches(beta_neg, last_A)
                gamma_pos = maybe_unfold_patches(gamma_pos, last_A)
                gamma_neg = maybe_unfold_patches(gamma_neg, last_A)
                A_x, bias = multiply_by_A_signs(
                    last_A, alpha_pos, alpha_neg, gamma_pos, gamma_neg)
                A_y, _ = multiply_by_A_signs(
                    last_A, beta_pos, beta_neg, None, None)
            elif type(last_A) == Tensor:
                last_A_pos, last_A_neg = last_A.clamp(min=0), last_A.clamp(max=0)
                A_x, _ = multiply_by_A_signs(last_A, alpha_pos, alpha_neg, None, None)
                A_y, _ = multiply_by_A_signs(last_A, beta_pos, beta_neg, None, None)
                A_x = self.broadcast_backward(A_x, x)
                A_y = self.broadcast_backward(A_y, y)
                if reduce_bias:
                    if opt:
                        bias = (torch.einsum('sb...,sb...->sb',
                                             last_A_pos, gamma_pos)
                                + torch.einsum('sb...,sb...->sb',
                                               last_A_neg, gamma_neg))
                    else:
                        bias = (self.get_bias(last_A_pos, gamma_pos.squeeze(0)) +
                            self.get_bias(last_A_neg, gamma_neg.squeeze(0)))
                else:
                    assert not opt
                    bias = last_A_pos * gamma_pos + last_A_neg * gamma_neg
                    assert len(x.output_shape) == bias.ndim - 1
                    assert len(y.output_shape) == bias.ndim - 1
                    bias_x = bias_y = bias
                    for i in range(2, bias.ndim):
                        if bias_x.shape[i] != x.output_shape[i - 1]:
                            assert x.output_shape[i - 1] == 1
                            bias_x = bias_x.sum(i, keepdim=True)
                    for i in range(2, bias.ndim):
                        if bias_y.shape[i] != y.output_shape[i - 1]:
                            assert y.output_shape[i - 1] == 1
                            bias_y = bias_y.sum(i, keepdim=True)
                    bias = (bias_x, bias_y)
            else:
                raise NotImplementedError(last_A)
            return A_x, A_y, bias

        alpha_l, beta_l, gamma_l = self.lw[0], self.lw[1], self.lb
        alpha_u, beta_u, gamma_u = self.uw[0], self.uw[1], self.ub

        if self.opt_stage in ['opt', 'reuse']:
            lA_x, lA_y, lbias = _bound_oneside(
                last_lA, alpha_l[0], beta_l[0], gamma_l[0],
                alpha_u[0], beta_u[0], gamma_u[0], opt=True)
            uA_x, uA_y, ubias = _bound_oneside(
                last_uA, alpha_u[1], beta_u[1], gamma_u[1],
                alpha_l[1], beta_l[1], gamma_l[1], opt=True)
        else:
            alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
            beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)
            gamma_l, gamma_u = gamma_l.unsqueeze(0), gamma_u.unsqueeze(0)
            lA_x, lA_y, lbias = _bound_oneside(
                last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
            uA_x, uA_y, ubias = _bound_oneside(
                last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    def bound_forward(self, dim_in, x, y):
        if self.is_linear_op:
            raise NotImplementedError
        return self.bound_forward_both_perturbed(dim_in, x, y)

    def bound_forward_both_perturbed(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = x.lw, x.lb, x.uw, x.ub
        y_lw, y_lb, y_uw, y_ub = y.lw, y.lb, y.uw, y.ub

        (alpha_l, beta_l, gamma_l,
         alpha_u, beta_u, gamma_u) = MulHelper.get_forward_relaxation(
             x.lower, x.upper, y.lower, y.upper, self.opt_stage, getattr(self, 'alpha', None), self._start)

        if x_lw is None: x_lw = 0
        if y_lw is None: y_lw = 0
        if x_uw is None: x_uw = 0
        if y_uw is None: y_uw = 0

        lw = alpha_l.unsqueeze(1).clamp(min=0) * x_lw + alpha_l.unsqueeze(1).clamp(max=0) * x_uw
        lw = lw + beta_l.unsqueeze(1).clamp(min=0) * y_lw + beta_l.unsqueeze(1).clamp(max=0) * y_uw
        lb = (alpha_l.clamp(min=0) * x_lb + alpha_l.clamp(max=0) * x_ub +
             beta_l.clamp(min=0) * y_lb + beta_l.clamp(max=0) * y_ub + gamma_l)
        uw = alpha_u.unsqueeze(1).clamp(max=0) * x_lw + alpha_u.unsqueeze(1).clamp(min=0) * x_uw
        uw = uw + beta_u.unsqueeze(1).clamp(max=0) * y_lw + beta_u.unsqueeze(1).clamp(min=0) * y_uw
        ub = (alpha_u.clamp(max=0) * x_lb + alpha_u.clamp(min=0) * x_ub +
             beta_u.clamp(max=0) * y_lb + beta_u.clamp(min=0) * y_ub + gamma_u)

        return LinearBound(lw, lb, uw, ub)

    @staticmethod
    def interval_propagate_constant(x, y, op=lambda x, const: x * const):
        # x is constant
        const = x[0]
        inp_lb = y[0]
        inp_ub = y[1]
        pos_mask = (const > 0).to(dtype=inp_lb.dtype)
        neg_mask = 1. - pos_mask
        lb = op(inp_lb, const * pos_mask) + op(inp_ub, const * neg_mask)
        ub = op(inp_ub, const * pos_mask) + op(inp_lb, const * neg_mask)
        return lb, ub

    def interval_propagate(self, x, y):
        if self.is_linear_op:
            if not self.inputs[0].perturbed:
                return self.interval_propagate_constant(x, y)
            elif not self.inputs[1].perturbed:
                return self.interval_propagate_constant(y, x)
            else:
                assert False
        else:
            return self.interval_propagate_both_perturbed(x, y)

    @staticmethod
    def interval_propagate_both_perturbed(*v):
        x, y = v[0], v[1]
        if x is y:
            # A shortcut for x * x.
            h_L, h_U = v[0]
            r0 = h_L * h_L
            r1 = h_U * h_U
            # When h_L < 0, h_U > 0, lower bound is 0.
            # When h_L < 0, h_U < 0, lower bound is h_U * h_U.
            # When h_L > 0, h_U > 0, lower bound is h_L * h_L.
            l = F.relu(h_L) - F.relu(-h_U)
            return l * l, torch.max(r0, r1)

        r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
        lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
        upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
        return lower, upper

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor):
            self.solver_vars = self.forward(*v)
            return
        gvar_array = np.array(v[0])
        gvar_array = gvar_array * v[1].cpu().numpy()
        self.solver_vars = gvar_array.tolist()

    def update_requires_input_bounds(self):
        self.is_linear_op = False
        for inp in self.inputs:
            if not inp.perturbed:
                # If any of the two inputs are constant, we do not need input bounds.
                self.is_linear_op = True
        if self.is_linear_op:
            # One input is constant; no bounds required.
            self.requires_input_bounds = []
            self.splittable = False
        else:
            # Both inputs are perturbed. Need relaxation.
            self.requires_input_bounds = [0, 1]
            self.splittable = True


class BoundDiv(Bound):

    def forward(self, x, y):
        # FIXME (05/11/2022): ad-hoc implementation for layer normalization
        if isinstance(self.inputs[1], BoundSqrt):
            input = self.inputs[0].inputs[0]
            x = input.forward_value
            n = input.forward_value.shape[-1]

            dev = x * (1. - 1. / n) - (x.sum(dim=-1, keepdim=True) - x) / n
            dev_sqr = dev ** 2
            s = (dev_sqr.sum(dim=-1, keepdim=True) - dev_sqr) / dev_sqr.clamp(min=epsilon)
            sqrt = torch.sqrt(1. / n * (s + 1))
            return torch.sign(dev) * (1. / sqrt)

        self.x, self.y = x, y
        return x / y
