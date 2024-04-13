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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tanh import BoundTanh
from .base import logger


# FIXME resolve duplicate code with BoundTanh
class BoundGelu(BoundTanh):
    sqrt_2 = math.sqrt(2)

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options, precompute=False)
        self.ibp_intermediate = False
        self.act_func = F.gelu
        def d_act_func(x):
            return (0.5 * (1 + torch.erf(x / self.sqrt_2))
                    + x * torch.exp(-0.5 * x ** 2) / math.sqrt(2 * torch.pi))
        self.d_act_func = d_act_func
        def d2_act_func(x):
            return (2 * torch.exp(-0.5 * x ** 2) / math.sqrt(2 * torch.pi)
                    - x ** 2 * torch.exp(-0.5 * x ** 2) / math.sqrt(2 * torch.pi))
        self.d2_act_func = d2_act_func
        self.precompute_relaxation(self.act_func, self.d_act_func)

    def _init_masks(self, x):
        lower = x.lower
        upper = x.upper
        self.mask_left_pos = torch.logical_and(lower >= -self.sqrt_2, upper <= 0)
        self.mask_left_neg = upper <= -self.sqrt_2
        self.mask_left = torch.logical_xor(upper <= 0,
                torch.logical_or(self.mask_left_pos, self.mask_left_neg))

        self.mask_right_pos = lower >= self.sqrt_2
        self.mask_right_neg = torch.logical_and(upper <= self.sqrt_2, lower >= 0)
        self.mask_right = torch.logical_xor(lower >= 0,
                torch.logical_or(self.mask_right_pos, self.mask_right_neg))

        self.mask_2 = torch.logical_and(torch.logical_and(upper > 0, upper <= self.sqrt_2),
                    torch.logical_and(lower < 0, lower >= -self.sqrt_2))
        self.mask_left_3 = torch.logical_and(lower < -self.sqrt_2, torch.logical_and(
            upper > 0, upper <= self.sqrt_2))
        self.mask_right_3 = torch.logical_and(upper > self.sqrt_2, torch.logical_and(
            lower < 0, lower >= -self.sqrt_2))
        self.mask_4 = torch.logical_and(lower < -self.sqrt_2, upper > self.sqrt_2)
        self.mask_both = torch.logical_or(self.mask_2, torch.logical_or(self.mask_4,
                    torch.logical_or(self.mask_left_3, self.mask_right_3)))

    @torch.no_grad()
    def precompute_relaxation(self, func, dfunc, x_limit=1000):
        """
        This function precomputes the tangent lines that will be used as
        lower/upper bounds for S-shapes functions.
        """
        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

        logger.debug('Precomputing relaxation for GeLU (pre-activation limit: %f)',
                     x_limit)

        def check_lower(upper, d):
            """Given two points upper, d (d <= upper), check if the slope at d
            will be less than f(upper) at upper."""
            k = dfunc(d)
            # Return True if the slope is a lower bound.
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            """Given two points lower, d (d >= lower), check if the slope at d
            will be greater than f(lower) at lower."""
            k = dfunc(d)
            # Return True if the slope is a upper bound.
            return k * (lower - d) + func(d) >= func(lower)

        # Given an upper bound point (>=0), find a line that is guaranteed to
        # be a lower bound of this function.
        upper = self.step_pre * torch.arange(
            0, self.num_points_pre + 5, device=self.device) + self.sqrt_2
        r = torch.ones_like(upper)
        # Initial guess, the tangent line is at -1.
        l = -torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an lower bound at f(upper).
            checked = check_lower(upper, l).int()
            # If the initial guess is not smaller enough, then double it (-2, -4, etc).
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to
        # be an lower bound at f(upper).
        # We want to further tighten this bound by moving it closer to 0.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_lower_right = l.clone()

        # Do the same again:
        # Given an lower bound point (<=0), find a line that is guaranteed to
        # be an upper bound of this function.
        lower = (
            -self.step_pre * torch.arange(
                0, self.num_points_pre + 5, device=self.device
            ) + self.sqrt_2).clamp(min=0.01)
        l = torch.zeros_like(upper) + self.sqrt_2
        r = torch.zeros_like(upper) + x_limit
        while True:
            checked = check_upper(lower, r).int()
            r = checked * r + (1 - checked) * (r * 2)
            if checked.sum() == l.numel():
                break
        for _ in range(max_iter):
            m = (l + r) / 2
            checked = check_upper(lower, m).int()
            l = (1 - checked) * m + checked * l
            r = (1 - checked) * r + checked * m
        self.d_upper_right = r.clone()

        upper = -self.step_pre * torch.arange(
            0, self.num_points_pre + 5, device=self.device) - self.sqrt_2
        r = torch.zeros_like(upper) - 0.7517916
        # Initial guess, the tangent line is at -1.
        l = torch.zeros_like(upper) - self.sqrt_2
        while True:
            checked = check_lower(upper, r).int()
            r = checked * r + (1 - checked) * (r * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be
        # an lower bound at f(upper).
        # We want to further tighten this bound by moving it closer to 0.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = (1 - checked) * m + checked * l
            r = (1 - checked) * r + checked * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_lower_left = r.clone()

        # Do the same again:
        # Given an lower bound point (<=0), find a line that is guaranteed to
        # be an upper bound of this function.
        lower = (
            self.step_pre * torch.arange(
                0, self.num_points_pre + 5, device=self.device
            ) - self.sqrt_2).clamp(max=0)
        l = torch.zeros_like(upper) - x_limit
        r = torch.zeros_like(upper) - self.sqrt_2
        while True:
            checked = check_upper(lower, l).int()
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        for _ in range(max_iter):
            m = (l + r) / 2
            checked = check_upper(lower, m).int()
            l = (1 - checked) * m + checked * l
            r = (1 - checked) * r + checked * m
        self.d_upper_left = r.clone()

        logger.debug('Done')

    def opt_init(self):
        super().opt_init()
        self.tp_right_lower_init = {}
        self.tp_right_upper_init = {}
        self.tp_left_lower_init = {}
        self.tp_left_upper_init = {}
        self.tp_both_lower_init = {}

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = [size_spec] + list(l.shape)
        alpha = torch.empty(14, *shape, device=l.device)
        alpha.data[:4] = ((l + u) / 2).unsqueeze(0).expand(4, *shape)
        alpha.data[4:6] = self.tp_right_lower_init[name_start].expand(2, *shape)
        alpha.data[6:8] = self.tp_right_upper_init[name_start].expand(2, *shape)
        alpha.data[8:10] = self.tp_left_lower_init[name_start].expand(2, *shape)
        alpha.data[10:12] = self.tp_left_upper_init[name_start].expand(2, *shape)
        alpha.data[12:14] = self.tp_both_lower_init[name_start].expand(2, *shape)
        return alpha

    def forward(self, x):
        return F.gelu(x)

    def bound_relax_impl(self, x, func, dfunc):
        lower, upper = x.lower, x.upper
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connect
        # (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # Fixed bounds that cannot be optimized. self.mask_neg are the masks
        # for neurons with upper bound <= 0.
        # Upper bound for the case of input lower bound <= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=torch.logical_or(
                torch.logical_or(self.mask_left_pos, self.mask_right_neg),
                self.mask_both
            ), type='upper', k=k_direct, x0=lower, y0=y_l)
        # Lower bound for the case of input upper bound >= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=torch.logical_or(self.mask_left_neg,
                    self.mask_right_pos), type='lower', k=k_direct, x0=lower, y0=y_l)

        # Indices of neurons with input upper bound >= sqrt(2),
        # whose optimal slope to lower bound on the right side was pre-computed.
        d_lower_right = self.retrieve_from_precompute(
            self.d_lower_right, upper - self.sqrt_2, lower)

        # Indices of neurons with input lower bound <= -sqrt(2),
        # whose optimal slope to lower bound on the left side was pre-computed.
        d_lower_left = self.retrieve_from_precompute(
            self.d_lower_left, -lower - self.sqrt_2, upper)

        # Indices of neurons with input lower bound <= sqrt(2),
        # whose optimal slope to upper bound on the right side was pre-computed.
        d_upper_right = self.retrieve_from_precompute(
            self.d_upper_right, -lower + self.sqrt_2, upper)

        # Indices of neurons with input lower bound <= sqrt(2),
        # whose optimal slope to upper bound on the right side was pre-computed.
        d_upper_left = self.retrieve_from_precompute(
            self.d_upper_left, -lower - self.sqrt_2, upper)

        if self.opt_stage in ['opt', 'reuse']:
            if not hasattr(self, 'alpha'):
                # Raise an error if alpha is not created.
                self._no_bound_parameters()
            ns = self._start

            # Clipping is done here rather than after `opt.step()` call
            # because it depends on pre-activation bounds
            self.alpha[ns].data[0:2] = torch.max(
                torch.min(self.alpha[ns][0:2], upper), lower)
            self.alpha[ns].data[2:4] = torch.max(
                torch.min(self.alpha[ns][2:4], upper), lower)
            self.alpha[ns].data[4:6] = torch.max(
                torch.min(self.alpha[ns][4:6], d_lower_right), lower)
            self.alpha[ns].data[6:8] = torch.max(
                self.alpha[ns][6:8], d_upper_right)
            self.alpha[ns].data[8:10] = torch.min(
                torch.max(self.alpha[ns][8:10], d_lower_left), upper)
            self.alpha[ns].data[10:12] = torch.min(
                self.alpha[ns][10:12], d_upper_left)
            self.alpha[ns].data[12:14] = torch.min(
                torch.max(self.alpha[ns][12:14], d_lower_left), d_lower_right)

            # shape [2, out_c, n, c, h, w].
            tp_pos = self.alpha[ns][0:2]  # For upper bound relaxation
            tp_neg = self.alpha[ns][2:4]  # For lower bound relaxation
            tp_right_lower = self.alpha[ns][4:6]
            tp_right_upper = self.alpha[ns][6:8]
            tp_left_lower = self.alpha[ns][8:10]
            tp_left_upper = self.alpha[ns][10:12]
            tp_both_lower = self.alpha[ns][12:14]

            # No need to use tangent line, when the tangent point is at the left
            # side of the preactivation lower bound. Simply connect the two sides.
            mask_direct = torch.logical_and(self.mask_right, k_direct < dfunc(lower))
            self.add_linear_relaxation(
                mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_or(self.mask_right_3,
                    torch.logical_xor(self.mask_right, mask_direct)), type='lower',
                k=dfunc(tp_right_lower), x0=tp_right_lower)
            mask_direct = torch.logical_and(self.mask_left, k_direct > dfunc(upper))
            self.add_linear_relaxation(
                mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_or(self.mask_left_3,
                    torch.logical_xor(self.mask_left, mask_direct)), type='lower',
                k=dfunc(tp_left_lower), x0=tp_left_lower)

            mask_direct = torch.logical_and(self.mask_right, k_direct < dfunc(upper))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_right, mask_direct), type='upper',
                k=dfunc(tp_right_upper), x0=tp_right_upper)
            mask_direct = torch.logical_and(self.mask_left, k_direct > dfunc(lower))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_left, mask_direct), type='upper',
                k=dfunc(tp_left_upper), x0=tp_left_upper)

            self.add_linear_relaxation(
                mask=self.mask_4, type='lower', k=dfunc(tp_both_lower), x0=tp_both_lower)
            self.add_linear_relaxation(
                mask=torch.logical_or(torch.logical_or(self.mask_left_pos, self.mask_right_neg),
                    self.mask_2), type='lower', k=dfunc(tp_neg), x0=tp_neg)
            self.add_linear_relaxation(
                mask=torch.logical_or(self.mask_right_pos,
                    self.mask_left_neg), type='upper', k=dfunc(tp_pos), x0=tp_pos)
        else:
            if self.opt_stage == 'init':
                # Initialize optimizable slope.
                tp_right_lower_init = d_lower_right.detach()
                tp_right_upper_init = d_upper_right.detach()
                tp_left_lower_init = d_lower_left.detach()
                tp_left_upper_init = d_upper_left.detach()
                tp_both_lower_init = d_lower_right.detach()

                ns = self._start
                self.tp_right_lower_init[ns] = tp_right_lower_init
                self.tp_right_upper_init[ns] = tp_right_upper_init
                self.tp_left_lower_init[ns] = tp_left_lower_init
                self.tp_left_upper_init[ns] = tp_left_upper_init
                self.tp_both_lower_init[ns] = tp_both_lower_init

            # Not optimized (vanilla CROWN bound).
            # Use the middle point slope as the lower/upper bound. Not optimized.
            m = (lower + upper) / 2
            y_m = func(m)
            k = dfunc(m)
            # Lower bound is the middle point slope for the case input upper bound <= 0.
            # Note that the upper bound in this case is the direct line between
            # (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(
                mask=torch.logical_or(
                    torch.logical_or(self.mask_left_pos, self.mask_right_neg),
                    self.mask_2
                ), type='lower', k=k, x0=m, y0=y_m)
            # Upper bound is the middle point slope for the case input lower bound >= 0.
            # Note that the lower bound in this case is the direct line between
            # (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=torch.logical_or(self.mask_right_pos,
                    self.mask_left_neg), type='upper', k=k, x0=m, y0=y_m)

            # Now handle the case where input lower bound <=0 and upper bound >= 0.
            # A tangent line starting at d_lower is guaranteed to be a lower bound
            # given the input upper bound.
            mask_direct = torch.logical_and(self.mask_right, k_direct < dfunc(lower))
            self.add_linear_relaxation(mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            # Otherwise we do not use the direct line, we use the d_lower slope.
            self.add_linear_relaxation(
                mask=torch.logical_or(torch.logical_or(self.mask_right_3, self.mask_4),
                    torch.logical_xor(self.mask_right, mask_direct)), type='lower',
                k=dfunc(d_lower_right), x0=d_lower_right)
            mask_direct = torch.logical_and(self.mask_left, k_direct > dfunc(upper))
            self.add_linear_relaxation(mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_or(self.mask_left_3,
                    torch.logical_xor(self.mask_left, mask_direct)), type='lower',
                k=dfunc(d_lower_left), x0=d_lower_left)

            mask_direct = torch.logical_and(self.mask_right, k_direct < dfunc(upper))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_right, mask_direct), type='upper',
                k=dfunc(d_upper_right), x0=d_upper_right)
            mask_direct = torch.logical_and(self.mask_left, k_direct > dfunc(lower))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_left, mask_direct), type='upper',
                k=dfunc(d_upper_left), x0=d_upper_left)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        self.bound_relax_impl(x, self.act_func, self.d_act_func)

    def interval_propagate(self, *v):
        pl, pu = self.forward(v[0][0]), self.forward(v[0][1])
        pl, pu = torch.min(pl, pu), torch.max(pl, pu)
        min_global = self.forward(torch.tensor(-0.7517916))
        pl, pu = torch.min(min_global, torch.min(pl, pu)), torch.max(pl, pu)
        return pl, pu


class GELUOp(torch.autograd.Function):
    sqrt_2 = math.sqrt(2)
    sqrt_2pi = math.sqrt(2 * math.pi)

    @staticmethod
    def symbolic(g, x):
        return g.op('custom::Gelu', x)

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.nn.functional.gelu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (0.5 * (1 + torch.erf(x / GELUOp.sqrt_2))
                + x * torch.exp(-0.5 * x ** 2) / GELUOp.sqrt_2pi)
        return grad_input * grad


class GELU(nn.Module):
    def forward(self, x):
        return GELUOp.apply(x)
