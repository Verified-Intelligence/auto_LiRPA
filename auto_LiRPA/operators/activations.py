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
""" Activation operators or other unary nonlinear operators, not including
those placed in separate files."""
import torch
from .base import *
from .activation_base import BoundActivation, BoundOptimizableActivation
from .clampmult import multiply_by_A_signs

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class BoundSoftplus(BoundActivation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x)


class BoundAbs(BoundActivation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ibp_intermediate = True

    def forward(self, x):
        return x.abs()

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        x_L = x.lower.clamp(max=0)
        x_U = torch.max(x.upper.clamp(min=0), x_L + 1e-8)
        mask_neg = x_U <= 0
        mask_pos = x_L >= 0
        y_L = x_L.abs()
        y_U = x_U.abs()
        upper_k = (y_U - y_L) / (x_U - x_L)
        upper_b = y_L - upper_k * x_L
        lower_k = (mask_neg * (-1.0) + mask_pos * 1.0)
        lower_b = (mask_neg + mask_pos) * (y_L - lower_k * x_L)
        if last_uA is not None:
            # Special case if we only want the upper bound with non-negative coefficients
            if last_uA.min() >= 0:
                uA = last_uA * upper_k
                ubias = self.get_bias(last_uA, upper_b)
            else:
                last_uA_pos = last_uA.clamp(min=0)
                last_uA_neg = last_uA.clamp(max=0)
                uA = last_uA_pos * upper_k + last_uA_neg * lower_k
                ubias = (self.get_bias(last_uA_pos, upper_b)
                         + self.get_bias(last_uA_neg, lower_b))
        else:
            uA, ubias = None, 0
        if last_lA is not None:
            if last_lA.max() <= 0:
                lA = last_lA * upper_k
                lbias = self.get_bias(last_lA, upper_b)
            else:
                last_lA_pos = last_lA.clamp(min=0)
                last_lA_neg = last_lA.clamp(max=0)
                lA = last_lA_pos * lower_k + last_lA_neg * upper_k
                lbias = (self.get_bias(last_lA_pos, lower_b)
                         + self.get_bias(last_lA_neg, upper_b))
        else:
            lA, lbias = None, 0
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        lower = ((h_U < 0) * h_U.abs() + (h_L > 0) * h_L.abs())
        upper = torch.max(h_L.abs(), h_U.abs())
        return lower, upper


class BoundATenHeaviside(BoundOptimizableActivation):
    def forward(self, *x):
        self.input_shape = x[0].shape
        # x[0]: input; x[1]: value when x == 0
        return torch.heaviside(x[0], x[1])

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        return self.forward(v[0][0], v[1][0]), self.forward(v[0][1], v[1][0])

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l = self.inputs[0].lower
        return torch.zeros_like(l).unsqueeze(0).repeat(2, *[1] * l.ndim)

    def clip_alpha(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0., 1.)

    def bound_backward(self, last_lA, last_uA, *x, start_node=None,
                       start_shape=None, **kwargs):
        x = x[0]
        if x is not None:
            lb_r = x.lower
            ub_r = x.upper
        else:
            lb_r = self.lower
            ub_r = self.upper

        if self.opt_stage not in ['opt', 'reuse']:
            # zero slope:
            upper_d = torch.zeros_like(lb_r, device=lb_r.device, dtype=lb_r.dtype)
            lower_d = torch.zeros_like(ub_r, device=ub_r.device, dtype=ub_r.dtype)
        else:
            upper_d = self.alpha[start_node.name][0].clamp(0, 1) * (1. / (-lb_r).clamp(min=1e-3))
            lower_d = self.alpha[start_node.name][1].clamp(0, 1) * (1. / (ub_r.clamp(min=1e-3)))

        upper_b = torch.ones_like(lb_r, device=lb_r.device, dtype=lb_r.dtype)
        lower_b = torch.zeros_like(lb_r, device=lb_r.device, dtype=lb_r.dtype)
        # For stable neurons, set fixed slope and bias.
        ub_mask = (ub_r <= 0).to(dtype=ub_r.dtype)
        lb_mask = (lb_r >= 0).to(dtype=lb_r.dtype)
        upper_b = upper_b - upper_b * ub_mask
        lower_b = lower_b * (1. - lb_mask) + lb_mask
        upper_d = upper_d - upper_d * ub_mask - upper_d * lb_mask
        lower_d = lower_d - lower_d * lb_mask - lower_d * ub_mask
        upper_d = upper_d.unsqueeze(0)
        lower_d = lower_d.unsqueeze(0)
        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None
        ubias = lbias = 0
        if last_uA is not None:
            neg_uA = last_uA.clamp(max=0)
            pos_uA = last_uA.clamp(min=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
            ubias = (pos_uA * upper_b + neg_uA * lower_b).flatten(2).sum(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            lbias = (pos_lA * lower_b + neg_lA * upper_b).flatten(2).sum(-1)

        return [(lA, uA), (None, None)], lbias, ubias


class BoundSqr(BoundOptimizableActivation):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True

    def forward(self, x):
        return x ** 2

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        upper_k = x.lower + x.upper
        # Upper bound: connect the two points (x_l, x_l^2) and (x_u, x_u^2).
        # The upper bound should always be better than IBP.
        self.add_linear_relaxation(
            mask=None, type='upper', k=upper_k, x0=x.lower)

        if self.opt_stage in ['opt', 'reuse']:
            mid = self.alpha[self._start]
        else:
            # Lower bound is a z=0 line if x_l and x_u have different signs.
            # Otherwise, the lower bound is a tangent line at x_l.
            # The lower bound should always be better than IBP.
            # If both x_l and x_u < 0, select x_u. If both > 0, select x_l.
            # If x_l < 0 and x_u > 0, we use the z=0 line as the lower bound.
            mid = F.relu(x.lower) - F.relu(-x.upper)

        self.add_linear_relaxation(mask=None, type='lower', k=2 * mid, x0=mid)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(2, size_spec, *l.shape, device=l.device)
        alpha.data[:2] = F.relu(l) - F.relu(-u)
        return alpha

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        lower = ((h_U < 0) * (h_U ** 2) + (h_L > 0) * (h_L ** 2))
        upper = torch.max(h_L ** 2, h_U ** 2)
        return lower, upper


class BoundHardTanh(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True
        self.activation_name = "HardTanh"

    def forward(self, x, min_val, max_val):
        return F.hardtanh(x, min_val=min_val, max_val=max_val)

    def clip_alpha(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0., 1.)

    def bound_backward(self, last_lA, last_uA, x, min_val, max_val, start_node=None,
                        reduce_bias=True, **kwargs):
        preact_lb = x.lower
        preact_ub = x.upper

        a = -1
        b = 1

        preact_ub = torch.max(preact_ub, preact_lb + 1e-8)
        direct_d = (F.hardtanh(preact_ub) - F.hardtanh(preact_lb)) / (
                    preact_ub - preact_lb)

        # Upper bound
        mask_direct_upper = preact_ub <= b
        mask_triangle_upper = (preact_ub - b) < (b - preact_lb)
        upper_triangle = (b - F.hardtanh(preact_lb)) / (b - preact_lb)

        if self.opt_stage in ['opt', 'reuse'] and hasattr(self, 'alpha'):
            upper_triangle[preact_lb > a] = 1
            selected_alpha_upper = self.alpha[start_node.name][0]

            if last_lA is not None:
                lb_upper_d = torch.max(torch.min(selected_alpha_upper, upper_triangle),
                                     torch.zeros_like(selected_alpha_upper))
                lb_upper_d = mask_direct_upper * direct_d + \
                             torch.logical_not(mask_direct_upper) * lb_upper_d
                lb_upper_b = mask_direct_upper * (F.hardtanh(preact_lb) - lb_upper_d * preact_lb) + \
                             torch.logical_not(mask_direct_upper) * (b - lb_upper_d * b)

            if last_uA is not None:
                ub_upper_d = torch.max(torch.min(selected_alpha_upper, upper_triangle),
                                     torch.zeros_like(selected_alpha_upper))
                ub_upper_d = mask_direct_upper * direct_d + \
                             torch.logical_not(mask_direct_upper) * ub_upper_d
                ub_upper_b = mask_direct_upper * (F.hardtanh(preact_lb) - ub_upper_d * preact_lb) + \
                             torch.logical_not(mask_direct_upper) * (b - ub_upper_d * b)
        else:
            upper_d = mask_direct_upper * direct_d + \
                      torch.logical_not(mask_direct_upper) * mask_triangle_upper * upper_triangle
            self.init_upper_d = upper_d
            lb_upper_d = ub_upper_d = upper_d.unsqueeze(0)
            lb_upper_b = ub_upper_b = mask_direct_upper * (
                        F.hardtanh(preact_lb) - lb_upper_d * preact_lb) + \
                                      torch.logical_not(mask_direct_upper) * (b - lb_upper_d * b)

        # Lower bound
        mask_direct_lower = preact_lb >= a
        mask_triangle_lower = (preact_ub - a) > (a - preact_lb)
        lower_triangle = (F.hardtanh(preact_ub) - a) / (preact_ub - a)

        if self.opt_stage in ['opt', 'reuse'] and hasattr(self, 'alpha'):
            lower_triangle[preact_lb < b] = 1
            selected_alpha_lower = self.alpha[start_node.name][1]

            if last_lA is not None:
                lb_lower_d = torch.max(torch.min(selected_alpha_lower, lower_triangle),
                                     torch.zeros_like(selected_alpha_lower))
                lb_lower_d = mask_direct_lower * direct_d + \
                             torch.logical_not(mask_direct_lower) * lb_lower_d
                lb_lower_b = mask_direct_lower * (F.hardtanh(preact_ub) - lb_lower_d * preact_ub) + \
                             torch.logical_not(mask_direct_lower) * (a - lb_lower_d * a)

            if last_uA is not None:
                ub_lower_d = torch.max(torch.min(selected_alpha_lower, lower_triangle),
                                     torch.zeros_like(selected_alpha_lower))
                ub_lower_d = mask_direct_lower * direct_d + \
                             torch.logical_not(mask_direct_lower) * ub_lower_d
                ub_lower_b = mask_direct_lower * (F.hardtanh(preact_ub) - ub_lower_d * preact_ub) + \
                             torch.logical_not(mask_direct_lower) * (a - ub_lower_d * a)
        else:
            lower_d = mask_direct_lower * direct_d + \
                      torch.logical_not(mask_direct_lower) * mask_triangle_lower * lower_triangle
            self.init_lower_d = lower_d
            lb_lower_d = ub_lower_d = lower_d.unsqueeze(0)
            lb_lower_b = ub_lower_b = mask_direct_lower * (
                        F.hardtanh(preact_ub) - lb_lower_d * preact_ub) + \
                                      torch.logical_not(mask_direct_lower) * (a - lb_lower_d * a)

        # final bounds
        uA = lA = None
        ubias = lbias = 0

        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            uA = ub_upper_d * pos_uA + ub_lower_d * neg_uA
            ubias = torch.einsum('bij,bij->bi', pos_uA, ub_upper_b) + \
                    torch.einsum('bij,bij->bi', neg_uA, ub_lower_b)
        
        if last_lA is not None:
            pos_lA = last_lA.clamp(min=0)
            neg_lA = last_lA.clamp(max=0)
            lA = lb_upper_d * neg_lA + lb_lower_d * pos_lA
            lbias = torch.einsum('bij,bij->bi', pos_lA, lb_lower_b) + \
                    torch.einsum('bij,bij->bi', neg_lA, lb_upper_b)

        return [(lA, uA), (None, None), (None, None)], lbias, ubias

    def _init_opt_parameters_impl(self, size_spec, name_start=None):
        if hasattr(self, 'init_upper_d'):
            alpha = torch.empty(2, size_spec, *self.init_upper_d.shape, device=self.init_upper_d.device)
            for i in range(size_spec):
                alpha.data[0, i] = self.init_upper_d
                alpha.data[1, i] = self.init_lower_d
            return alpha
        else:
            l = self.inputs[0].lower
            return torch.zeros(2, size_spec, *l.shape, device=l.device, dtype=l.dtype)

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        return F.hardtanh(h_L, min_val=-1.0, max_val=1.0), F.hardtanh(h_U, min_val=-1.0, max_val=1.0)


class BoundFloor(BoundActivation):
    def forward(self, x):
        return torch.floor(x)

    def bound_relax(self, x, init=False):
        if init:
            self.init_linear_relaxation(x)
        self.lb += torch.floor(x.lower)
        self.ub += torch.floor(x.upper)


class BoundMultiPiecewiseNonlinear(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True

    def forward(self, x, weight, offset):
        return (F.relu(x.unsqueeze(-1) - offset) * weight).sum(dim=-1)

    def clip_alpha(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0., 1.)

    def bound_backward(self, last_lA, last_uA, x, weight, offset,
                       reduce_bias=True, start_node=None, **kwargs):
        assert not self.is_input_perturbed(1)
        assert not self.is_input_perturbed(2)

        weight = self.inputs[1].forward_value
        offset = self.inputs[2].forward_value

        relu_x_lower = (x.lower.unsqueeze(-1) - offset).clamp(max=0)
        relu_x_upper = (x.upper.unsqueeze(-1) - offset).clamp(min=0)
        relu_x_upper = torch.max(relu_x_upper, relu_x_lower + 1e-8)
        relu_upper_k = relu_x_upper / (relu_x_upper - relu_x_lower)
        relu_upper_b = -relu_x_lower * relu_upper_k
        if self.opt_stage not in ['opt', 'reuse']:
            self.init_lower_k = relu_lower_k = (relu_upper_k > 0.5).to(relu_upper_k)
            relu_lower_k_for_lA = relu_lower_k_for_uA = relu_lower_k.unsqueeze(0)
        else:
            relu_lower_k = self.alpha[start_node.name]
            relu_lower_k_for_lA = relu_lower_k[0]
            relu_lower_k_for_uA = relu_lower_k[1]
        relu_lower_b = torch.zeros_like(relu_upper_b)
        relu_lower_b = relu_lower_b.unsqueeze(0)
        relu_upper_k = relu_upper_k.unsqueeze(0)
        relu_upper_b = relu_upper_b.unsqueeze(0)

        def _bound_oneside(last_A, pos_k, pos_b, neg_k, neg_b, weight, offset, reduce_bias):
            if last_A is None:
                return None, 0
            last_A = last_A.unsqueeze(-1) * weight
            A_pos = last_A.clamp(min=0)
            A_neg = last_A.clamp(max=0)
            A = A_pos * pos_k + A_neg * neg_k
            b = -A * offset + A_pos * pos_b + A_neg * neg_b
            A = A.sum(dim=-1)
            if reduce_bias:
                b = b.sum(dim=[-1, -2])
            else:
                b = b.sum(dim=-1)
            return A, b

        lA, lb = _bound_oneside(last_lA, relu_lower_k_for_lA, relu_lower_b,
                                relu_upper_k, relu_upper_b,
                                weight, offset, reduce_bias)
        uA, ub = _bound_oneside(last_uA, relu_upper_k, relu_upper_b,
                                relu_lower_k_for_uA, relu_lower_b,
                                weight, offset, reduce_bias)

        return [(lA, uA), (None, None), (None, None)], lb, ub

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        alpha = torch.empty(2, size_spec, *self.init_lower_k.shape,
                            device=self.init_lower_k.device)
        alpha.data[:2] = self.init_lower_k
        return alpha

    def get_split_mask(self, lower, upper, input_index):
        offset = self.inputs[2].forward_value
        return ((lower.unsqueeze(-1) < offset) & (upper.unsqueeze(-1) > offset)).any(dim=-1)
