#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import torch
from .base import *
from .clampmult import multiply_by_A_signs
from .activation_base import BoundOptimizableActivation


class BoundMinMax(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.options = options
        self.requires_input_bounds = [0, 1]
        self.op = None

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l = self.inputs[0].lower
        # Alpha dimension is (2, output_shape, batch, *shape).
        shape = [2, size_spec] + list(l.shape)
        return torch.ones(shape, device=l.device)

    def clip_alpha(self):
        # See https://www.overleaf.com/read/jzgrcmqtqpcx#9dbf97 for the math behind this code.
        lb_x = self._cached_lb_x
        ub_x = self._cached_ub_x
        lb_y = self._cached_lb_y
        ub_y = self._cached_ub_y

        for v in self.alpha.values():
            eps = torch.tensor(1e-6).to(lb_x.dtype)
            if self.op == 'max':
                # Case 1: l_x >= u_y
                case1 = (lb_x >= ub_y).requires_grad_(False).to(lb_x.dtype)
                alpha_u_lb = torch.zeros_like(case1)
                alpha_u_ub = torch.zeros_like(case1)
                alpha_l_lb = torch.zeros_like(case1)
                alpha_l_ub = torch.zeros_like(case1)

                # Case 2: l_x < u_y && u_x > u_y
                case2 = ((lb_x < ub_y) * (ub_x > ub_y)).requires_grad_(False).to(lb_x.dtype)
                alpha_u_lb += case2 * (ub_x - ub_y) / (ub_x - torch.maximum(lb_x, lb_y))
                alpha_u_ub += case2
                alpha_l_ub += case2

                # Case 3: l_x < u_y && u_x == u_y
                case3 = ((lb_x < ub_y) * (ub_x == ub_y)).requires_grad_(False).to(lb_x.dtype)
                alpha_u_ub += case3
                alpha_l_ub += case3

                alpha_u_lb = torch.clamp(alpha_u_lb, min=eps)
                alpha_u_ub = torch.clamp(alpha_u_ub, min=eps)
            elif self.op == 'min':
                # Case 1: l_y >= u_x
                case1 = (lb_y >= ub_x).requires_grad_(False).to(lb_x.dtype)
                alpha_u_lb = torch.zeros_like(case1)
                alpha_u_ub = torch.zeros_like(case1)
                alpha_l_lb = torch.zeros_like(case1)
                alpha_l_ub = torch.zeros_like(case1)

                # Case 2: l_y < u_x && l_y > l_x
                case2 = ((lb_y < ub_x) * (lb_y > lb_x)).requires_grad_(False).to(lb_x.dtype)
                alpha_u_ub += case2
                alpha_l_lb += case2 * (lb_y - lb_x) / (torch.minimum(ub_x, ub_y) - lb_x)
                alpha_l_ub += case2

                # Case 3: l_y < u_x && l_y == l_x
                case3 = ((lb_y < ub_x) * (lb_y == lb_x)).requires_grad_(False).to(lb_x.dtype)
                alpha_u_ub += case3
                alpha_l_ub += case3

                alpha_l_lb = torch.clamp(alpha_l_lb, min=eps)
                alpha_l_ub = torch.clamp(alpha_l_ub, min=eps)

            v.data[0] = torch.clamp(v.data[0], alpha_u_lb, alpha_u_ub)
            v.data[1] = torch.clamp(v.data[1], alpha_l_lb, alpha_l_ub)

    def forward(self, x, y):
        if self.op == 'max':
            return torch.max(x, y)
        elif self.op == 'min':
            return torch.min(x, y)
        else:
            raise NotImplementedError

    def _backward_relaxation(self, x, y, start_node=None):
        # See https://www.overleaf.com/read/jzgrcmqtqpcx#9dbf97 for the math behind this code.

        lb_x = x.lower
        ub_x = x.upper
        lb_y = y.lower
        ub_y = y.upper

        if self.opt_stage in ['opt', 'reuse']:
            selected_alpha = self.alpha[start_node.name]
            alpha_u = selected_alpha[0]
            alpha_l = selected_alpha[1]
        else:
            alpha_u = alpha_l = 1

        ub_x = ub_x.unsqueeze(0)
        ub_y = ub_y.unsqueeze(0)
        lb_x = lb_x.unsqueeze(0)
        lb_y = lb_y.unsqueeze(0)

        if self.op == 'max':
            swapped_inputs = ub_x < ub_y
        elif self.op == 'min':
            swapped_inputs = lb_y < lb_x
        else:
            raise NotImplementedError
        lb_x, lb_y = torch.where(swapped_inputs, lb_y, lb_x), torch.where(swapped_inputs, lb_x, lb_y)
        ub_x, ub_y = torch.where(swapped_inputs, ub_y, ub_x), torch.where(swapped_inputs, ub_x, ub_y)

        self._cached_lb_x = lb_x.detach()
        self._cached_ub_x = ub_x.detach()
        self._cached_lb_y = lb_y.detach()
        self._cached_ub_y = ub_y.detach()

        epsilon = 1e-6
        ub_x = torch.max(ub_x, lb_x + epsilon)
        ub_y = torch.max(ub_y, lb_y + epsilon)
        # Ideally, if x or y are constant, this layer should be replaced by a ReLU
        # max{x, c} = max{x − c, 0} + c
        # min{x, c} = −max{−x, −c} = −(max{−x + c, 0} − c) = −max{−x + c, 0} + c
        if torch.any(lb_x + 1e-4 >= ub_x) or torch.any(lb_y + 1e-4 >= ub_y):
            print("Warning: MinMax layer (often used for clamping) received at "
                  "least one input with lower bound almost equal to the upper "
                  "bound. This can happen e.g. if x or y are constants. Consider "
                  "replacing this layer with a ReLU for higher efficieny.")
        assert torch.all(ub_x != lb_x) and torch.all(ub_y != lb_y), (
            'Lower/upper bounds are too close and epsilon was rounded away. '
            'To fix this, increase epsilon.'
        )

        if isinstance(alpha_u, torch.Tensor):
            assert alpha_u.shape[1:] == ub_x.shape[1:]
            shape = alpha_u.shape
        else:
            shape = ub_x.shape
        upper_dx = torch.zeros(shape, device=ub_x.device)
        upper_dy = torch.zeros(shape, device=ub_x.device)
        lower_dx = torch.zeros(shape, device=ub_x.device)
        lower_dy = torch.zeros(shape, device=ub_x.device)
        upper_b = torch.zeros(shape, device=ub_x.device)
        lower_b = torch.zeros(shape, device=ub_x.device)
        if self.op == 'max':
            # Case 1: l_x >= u_y
            case1 = (lb_x >= ub_y).requires_grad_(False).to(lb_x.dtype)
            upper_dx += case1
            lower_dx += case1

            # Case 2: l_x < u_y && u_x > u_y
            case2 = ((lb_x < ub_y) * (ub_x > ub_y)).requires_grad_(False).to(lb_x.dtype)
            upper_dx += case2 * (ub_y - ub_x) / (alpha_u * (lb_x - ub_x))
            upper_dy += case2 * (alpha_u - 1) * (ub_y - ub_x) / (alpha_u * (ub_y - lb_y))
            upper_b += case2 * (ub_x - (ub_x * (ub_y - ub_x)) / (alpha_u * (lb_x - ub_x))
                                - ((alpha_u - 1) * (ub_y - ub_x) * lb_y) / (alpha_u * (ub_y - lb_y)))
            lower_dx += case2 * (1 - alpha_l)
            lower_dy += case2 * alpha_l

            # Case 3: l_x < u_y && u_x == u_y
            case3 = ((lb_x < ub_y) * (ub_x == ub_y)).requires_grad_(False).to(lb_x.dtype)
            upper_dx += case3 * alpha_u * (ub_x - torch.maximum(lb_x, lb_y)) / (ub_x - lb_x)
            upper_dy += case3 * alpha_u * (ub_x - torch.maximum(lb_x, lb_y)) / (ub_y - lb_y)
            upper_b += case3 * (ub_x -
                        (alpha_u * (ub_x - torch.maximum(lb_x, lb_y)) * lb_x) / (ub_x - lb_x) -
                        (alpha_u * (ub_x - torch.maximum(lb_x, lb_y)) * ub_y) / (ub_y - lb_y))
            lower_dx += case3 * (1 - alpha_l)
            lower_dy += case3 * alpha_l
        elif self.op == 'min':
            # Case 1: l_y >= u_x
            case1 = (lb_y >= ub_x).requires_grad_(False).to(lb_x.dtype)
            upper_dx = case1.clone()
            lower_dx = case1.clone()
            upper_dy = torch.zeros_like(case1)
            lower_dy = torch.zeros_like(case1)
            upper_b = torch.zeros_like(case1)
            lower_b = torch.zeros_like(case1)

            # Case 2: l_y < u_x && l_y > l_x
            case2 = ((lb_y < ub_x) * (lb_y > lb_x)).requires_grad_(False).to(lb_x.dtype)
            upper_dx += case2 * (1 - alpha_u)
            upper_dy += case2 * alpha_u
            lower_dx += case2 * (lb_x - lb_y) / (alpha_l * (lb_x - ub_x))
            lower_dy += case2 * (alpha_l - 1) * (lb_x - lb_y) / (alpha_l * (ub_y - lb_y))
            lower_b += case2 * (lb_y - (ub_x * (lb_x - lb_y)) / (alpha_l * (lb_x - ub_x))
                                - ((alpha_l - 1) * (lb_x - lb_y) * lb_y) / (alpha_l * (ub_y - lb_y)))

            # Case 3: l_y < u_x && l_y == l_x
            case3 = ((lb_y < ub_x) * (lb_y == lb_x)).requires_grad_(False).to(lb_x.dtype)
            upper_dx += case3 * (1 - alpha_u)
            upper_dy += case3 * alpha_u
            lower_dx += case3 * alpha_l * (torch.minimum(ub_x, ub_y) - lb_x) / (ub_x - lb_x)
            lower_dy += case3 * alpha_l * (torch.minimum(ub_x, ub_y) - lb_x) / (ub_y - lb_y)
            lower_b += case3 * (lb_x -
                        (alpha_l * (torch.minimum(ub_x, ub_y) - lb_x) * lb_x) / (ub_x - lb_x) -
                        (alpha_l * (torch.minimum(ub_x, ub_y) - lb_x) * ub_y) / (ub_y - lb_y))
        else:
            raise NotImplementedError

        lower_dx, lower_dy = torch.where(swapped_inputs, lower_dy, lower_dx), torch.where(swapped_inputs, lower_dx, lower_dy)
        upper_dx, upper_dy = torch.where(swapped_inputs, upper_dy, upper_dx), torch.where(swapped_inputs, upper_dx, upper_dy)

        return upper_dx, upper_dy, upper_b, lower_dx, lower_dy, lower_b

    def bound_backward(self, last_lA, last_uA, x=None, y=None, start_shape=None,
                       start_node=None, **kwargs):
        # Get element-wise CROWN linear relaxations.
        upper_dx, upper_dy, upper_b, lower_dx, lower_dy, lower_b = \
            self._backward_relaxation(x, y, start_node)

        # Choose upper or lower bounds based on the sign of last_A
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0
            # Obtain the new linear relaxation coefficients based on the signs in last_A.
            _A, _bias = multiply_by_A_signs(last_A, d_pos, d_neg, b_pos, b_neg)
            if isinstance(last_A, Patches):
                # Save the patch size, which will be used in init_slope() to determine the number of optimizable parameters.
                A_prod = _A.patches
                if start_node is not None:
                    # Regular patches.
                    self.patch_size[start_node.name] = A_prod.size()
            return _A, _bias

        # In patches mode we might need an unfold.
        # lower_dx, lower_dy, upper_dx, upper_dy, lower_b, upper_b: 1, batch, current_c, current_w, current_h or None
        # In _backward_relaxation, the lb_x etc. potentially got swapped. This may cause the memory to become
        # non-contiguous. This is not a problem if the spec_size is 1, e.g. if alphas are shared.
        upper_dx = upper_dx.contiguous()
        upper_dy = upper_dy.contiguous()
        lower_dx = lower_dx.contiguous()
        lower_dy = lower_dy.contiguous()
        upper_b = upper_b.contiguous()
        lower_b = lower_b.contiguous()


        upper_dx = maybe_unfold_patches(upper_dx, last_lA if last_lA is not None else last_uA)
        upper_dy = maybe_unfold_patches(upper_dy, last_lA if last_lA is not None else last_uA)
        lower_dx = maybe_unfold_patches(lower_dx, last_lA if last_lA is not None else last_uA)
        lower_dy = maybe_unfold_patches(lower_dy, last_lA if last_lA is not None else last_uA)
        upper_b = maybe_unfold_patches(upper_b, last_lA if last_lA is not None else last_uA)
        lower_b = maybe_unfold_patches(lower_b, last_lA if last_lA is not None else last_uA)

        uAx, ubias = _bound_oneside(last_uA, upper_dx, lower_dx, upper_b, lower_b)
        uAy, ubias2 = _bound_oneside(last_uA, upper_dy, lower_dy, upper_b, lower_b)
        if isinstance(ubias, torch.Tensor):
            assert isinstance(ubias2, torch.Tensor)
            assert torch.all(ubias == ubias2)
        else:
            assert ubias == ubias2 == 0
        lAx, lbias = _bound_oneside(last_lA, lower_dx, upper_dx, lower_b, upper_b)
        lAy, lbias2 = _bound_oneside(last_lA, lower_dy, upper_dy, lower_b, upper_b)
        if isinstance(lbias, torch.Tensor):
            assert isinstance(lbias2, torch.Tensor)
            assert torch.all(lbias == lbias2)
        else:
            assert lbias == lbias2 == 0

        return [(lAx, uAx), (lAy, uAy)], lbias, ubias

    def interval_propagate(self, *v):
        h_Lx, h_Ux = v[0][0], v[0][1]
        h_Ly, h_Uy = v[1][0], v[1][1]
        return self.forward(h_Lx, h_Ly), self.forward(h_Ux, h_Uy)


class BoundMax(BoundMinMax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = 'max'


class BoundMin(BoundMinMax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = 'min'
