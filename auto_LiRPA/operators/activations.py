#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
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

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class BoundSoftplus(BoundActivation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x)


class BoundAbs(BoundActivation):
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
        lower_b = (mask_neg + mask_pos) * ( y_L - lower_k * x_L )
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
        return x**2

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

        self.add_linear_relaxation(mask=None, type='lower', k=2*mid, x0=mid)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(2, size_spec, *l.shape, device=l.device)
        alpha.data[:2] = F.relu(l) - F.relu(-u)
        return alpha

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        lower = ((h_U < 0) * (h_U**2) + (h_L > 0) * (h_L**2))
        upper = torch.max(h_L**2, h_U**2)
        return lower, upper
