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
"""Unary nonlinearities other than activation functions."""
import torch
from .base import *
from .tanh import BoundTanh


# TODO too much code in this class is a duplicate of BoundTanh
class BoundOptimizableNonLinear(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         precompute=False)
        # activation function needs to be nn.module
        self.splittable = True
        self.act_func = None
        self.d_act_func = None
        self.inflections = []
        self.extremes = []

    def branch_input_domain(self, lb, ub):
        lower = lb
        upper = ub
        num_inflection = torch.zeros_like(lower)
        inflection_mat = lower
        for inflection in self.inflections:
            num_inflection += torch.logical_and(
                lower <= inflection, upper >= inflection)
            inflection_mat = torch.where(
                torch.logical_and(lower <= inflection, upper >= inflection),
                torch.tensor(inflection, device=lb.device), inflection_mat)
        inflection_mask = num_inflection <= 1.

        extreme_mask = torch.ones_like(lower)
        for extreme in self.extremes:
            extreme_mask *= torch.logical_or(lower >= extreme, upper <= extreme)

        self.sigmoid_like_mask = torch.logical_and(inflection_mask, extreme_mask)
        self.branch_mask = torch.logical_xor(torch.ones_like(lower), self.sigmoid_like_mask)
        self.inflection_mat = torch.where(self.sigmoid_like_mask, inflection_mat, lower)

        self.mask_neg = torch.logical_and((self.d2_act_func(lower) >= 0),
            torch.logical_and((self.d2_act_func(upper) >= 0),
            self.sigmoid_like_mask))
        self.mask_pos = torch.logical_and((self.d2_act_func(lower) < 0),
            torch.logical_and((self.d2_act_func(upper) < 0),
            self.sigmoid_like_mask))
        self.mask_both = torch.logical_xor(self.sigmoid_like_mask,
            torch.logical_or(self.mask_neg, self.mask_pos))
        self.convex_concave = self.d2_act_func(lower) >= 0

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        return super()._init_opt_parameters_impl(size_spec, name_start, num_params=10)

    def generate_inflections(self, lb, ub):
        raise NotImplementedError

    def bound_relax_branch(self, lb, ub):
        raise NotImplementedError

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        lb = x.lower
        ub = x.upper
        self.generate_inflections(lb, ub)
        self.branch_input_domain(lb, ub)
        super().bound_relax_impl(x, self.act_func, self.d_act_func)
        lower_slope, lower_bias, upper_slope, upper_bias = self.bound_relax_branch(lb, ub)
        self.lw = self.lw * self.sigmoid_like_mask + self.branch_mask * lower_slope
        self.lb = self.lb * self.sigmoid_like_mask + self.branch_mask * lower_bias
        self.uw = self.uw * self.sigmoid_like_mask + self.branch_mask * upper_slope
        self.ub = self.ub * self.sigmoid_like_mask + self.branch_mask * upper_bias


class BoundPow(BoundOptimizableNonLinear):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.ibp_intermediate = False
        self.has_constraint = True
        self.exponent = 2
        def act_func(x):
            return torch.pow(x, self.exponent)
        self.act_func = act_func
        def d_act_func(x):
            return self.exponent * torch.pow(x, self.exponent - 1)
        self.d_act_func = d_act_func
        def d2_act_func(x):
            return self.exponent * (self.exponent - 1) * torch.pow(x, self.exponent - 2)
        self.d2_act_func = d2_act_func

    def generate_inflections(self, lb, ub):
        if self.exponent % 2:
            self.inflections = [0.]
        else:
            self.extremes = [0.]

    def generate_d_lower_upper(self, lower, upper):
        if self.exponent % 2:
            # Indices of neurons with input upper bound >=0,
            # whose optimal slope to lower bound the function was pre-computed.
            # Note that for neurons with also input lower bound >=0, they will be masked later.
            d_upper = self.retrieve_from_precompute(self.d_upper, upper, lower)

            # Indices of neurons with lower bound <=0,
            # whose optimal slope to upper bound the function was pre-computed.
            d_lower = self.retrieve_from_precompute(self.d_lower, -lower, upper)
            return d_lower, d_upper
        else:
            return torch.zeros_like(upper), torch.zeros_like(upper)

    @torch.no_grad()
    def precompute_relaxation(self, func, dfunc, x_limit = 500):
        """
        This function precomputes the tangent lines that will be used as
        lower/upper bounds for S-shapes functions.
        """
        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

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
            0, self.num_points_pre + 5, device=self.device)
        r = torch.zeros_like(upper)
        # Initial guess, the tangent line is at -1.
        l = -torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an lower bound at f(upper).
            checked = check_upper(upper, l).int()
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
            checked = check_upper(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_upper = l.clone()

        # Do the same again:
        # Given an lower bound point (<=0), find a line that is guaranteed to
        # be an upper bound of this function.
        lower = -self.step_pre * torch.arange(
            0, self.num_points_pre + 5, device=self.device)
        l = torch.zeros_like(upper)
        r = torch.ones_like(upper)
        while True:
            checked = check_lower(lower, r).int()
            r = checked * r + (1 - checked) * (r * 2)
            if checked.sum() == l.numel():
                break
        for _ in range(max_iter):
            m = (l + r) / 2
            checked = check_lower(lower, m).int()
            l = (1 - checked) * m + checked * l
            r = (1 - checked) * r + checked * m
        self.d_lower = r.clone()

    def forward(self, x, y):
        return torch.pow(x, y)

    def bound_backward(self, last_lA, last_uA, x, y, start_node=None,
                       start_shape=None, **kwargs):
        assert not self.is_input_perturbed(1)
        self._start = start_node.name if start_node is not None else None
        y = y.value
        if y == int(y):
            x.upper = torch.max(x.upper, x.lower + 1e-8)
            self.exponent = int(y)
            assert self.exponent >= 2
            if self.exponent % 2:
                self.precompute_relaxation(self.act_func, self.d_act_func)

            As, lbias, ubias = super().bound_backward(
                last_lA, last_uA, x, start_node, start_shape, **kwargs)
            return [As[0], (None, None)], lbias, ubias
        else:
            raise NotImplementedError('Exponent is not supported yet')

    def bound_forward(self, dim_in, x, y):
        assert y.lower == y.upper == int(y.lower)
        y = y.lower
        x.upper = torch.max(x.upper, x.lower + 1e-8)
        self.exponent = int(y)
        assert self.exponent >= 2
        if self.exponent % 2:
            self.precompute_relaxation(self.act_func, self.d_act_func)
        return super().bound_forward(dim_in, x)

    def bound_relax_branch(self, lb, ub):
        if self.opt_stage in ['opt', 'reuse']:
            if not hasattr(self, 'alpha'):
                # Raise an error if alpha is not created.
                self._no_bound_parameters()
            ns = self._start

            self.alpha[ns].data[8:10] = torch.max(
                torch.min(self.alpha[ns][8:10], ub), lb)
            lb_point = self.alpha[ns][8:10]
            lower_slope = self.d_act_func(lb_point)
            lower_bias = self.act_func(lb_point) - lower_slope * lb_point
        else:
            lower_slope = 0
            lower_bias = 0

        upper_slope = (self.act_func(ub) - self.act_func(lb)) / (ub - lb).clamp(min=1e-8)
        upper_bias = self.act_func(ub) - ub * upper_slope
        return lower_slope, lower_bias, upper_slope, upper_bias

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        exp = v[1][0]
        assert exp == int(exp)
        exp = int(exp)
        pl, pu = torch.pow(v[0][0], exp), torch.pow(v[0][1], exp)
        if exp % 2 == 1:
            return pl, pu
        else:
            pl, pu = torch.min(pl, pu), torch.max(pl, pu)
            mask = 1 - ((v[0][0] < 0) * (v[0][1] > 0)).to(pl.dtype)
            return pl * mask, pu

    def clamp_interim_bounds(self):
        if self.exponent % 2 == 0:
            self.cstr_lower = self.lower.clamp(min=0)
            self.cstr_upper = self.upper.clamp(min=0)
            self.cstr_interval = (self.cstr_lower, self.cstr_upper)
