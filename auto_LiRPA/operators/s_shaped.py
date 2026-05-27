#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""S-shaped base class, activation functions, and relevant ops."""
import torch
from torch.nn import Module
from torch.autograd import Function
from .base import *
from .activation_base import BoundOptimizableActivation


class BoundSShaped(BoundOptimizableActivation):
    """
    Base class for computing output bounds of globally and partially s-shaped nonlinear functions
    (e.g., sigmoid, tanh, sin, cos) over given input intervals.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None, activation=(None, None, None), precompute=False):
        super().__init__(attr, inputs, output_index, options)
        if options is None:
            options = {}
        self.splittable = True
        self.inverse_s_shape = False
        self.ibp_intermediate = True

        self.activation = activation
        self.activation_name = activation[0]

        self.act_func = activation[1]
        self.d_act_func = activation[2]

        self.step_pre = 0.01
        if precompute:
            self.precompute_relaxation(self.act_func, self.d_act_func)
            self.precompute_dfunc_values(self.act_func, self.d_act_func)
        # TODO make them configurable when implementing a general nonlinear activation.
        # Neurons whose gap between pre-activation bounds is smaller than this
        # threshold will be masked and don't need branching.
        self.split_min_gap = 1e-2  # 1e-4
        # Neurons whose pre-activation bounds don't overlap with this range
        # are considered as stable (with values either 0 or 1) and don't need
        # branching.
        self.split_range = (self.range_l, self.range_u)
        # The initialization will be adjusted if the pre-activation bounds are too loose.
        self.loose_threshold = options.get(self.activation_name, {}).get(
            'loose_threshold', None)
        self.convex_concave = None
        self.activation_bound_option = options.get('activation_bound_option', 'adaptive')

        self.inflections = [0.]
        self.extremes = []
        self.sigmoid_like_mask = None

        # FIXME: Smoothness enhancement for s-shaped functions should be enabled by default.
        # This enhancement makes the linear bounds change smoothly between different cases.
        # We provide this option only to reproduce results from previous papers.
        self.disable_smoothness_enhancement = options.get(
            's_shaped_disable_smoothness_enhancement', False)

    def opt_init(self):
        super().opt_init()
        self.tp_both_lower_init = {}
        self.tp_both_upper_init = {}

    def branch_input_domain(self, lb, ub):
        # For functions that are only partially s-shaped, such as sin and cos, the non-s-shaped intervals are identified
        # and masked here. sigmoid_like_mask marks the strictly s-shaped intervals, and branch_mask marks the non-s-
        # shaped ones. For globally s-shaped functions like tanh and sigmoid, sigmoid_like_mask stores all 1s and
        # branch_mask stores all 0s.
        self.sigmoid_like_mask = torch.ones_like(lb, dtype=torch.bool)
        self.branch_mask = torch.zeros_like(lb, dtype=torch.bool)

    def _init_opt_parameters_impl(self, size_spec, name_start, num_params=10):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = l.shape
        # Alpha dimension is (num_params, output_shape, batch, *shape) for the s-shaped activation function.
        alpha = torch.empty(num_params, size_spec, *shape, device=l.device)
        alpha.data[:4] = (l + u) / 2
        alpha.data[4:6] = self.tp_both_lower_init[name_start]
        alpha.data[6:8] = self.tp_both_upper_init[name_start]
        if num_params > 8:
            alpha.data[8:] = 0
        return alpha

    @torch.no_grad()
    def precompute_relaxation(self, func, dfunc, x_limit=500):
        """
        This function precomputes the tangent lines that will be used as
        lower/upper bounds for S-shaped functions centered at 0 along the x-axis.
        """
        self.x_limit = x_limit
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

        logger.debug('Precomputing relaxation for %s (pre-activation limit: %f)',
                     self.__class__.__name__, x_limit)

        def check_lower(upper, d):
            """Given two points upper, d (d <= upper),
            check if the slope at d will be less than f(upper) at upper."""
            k = dfunc(d)
            # Return True if the slope is a lower bound.
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            """Given two points lower, d (d >= lower),
            check if the slope at d will be greater than f(lower) at lower."""
            k = dfunc(d)
            # Return True if the slope is a upper bound.
            return k * (lower - d) + func(d) >= func(lower)

        # Given an upper bound point (>=0), find a line that is guaranteed to be a lower bound of this function.
        upper = self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        r = torch.zeros_like(upper)
        # Initial guess, the tangent line is at -1.
        l = -torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an lower bound at f(upper).
            checked = check_lower(upper, l).int()
            # If the initial guess is not smaller enough, then double it (-2, -4, etc).
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
        # We want to further tighten this bound by moving it closer to 0.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_lower = l.clone()

        # Do the same again:
        # Given an lower bound point (<=0), find a line that is guaranteed to be an upper bound of this function.
        lower = -self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        l = torch.zeros_like(upper)
        r = torch.ones_like(upper)
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
        self.d_upper = r.clone()

        logger.debug('Done')

    def precompute_dfunc_values(self, func, dfunc, x_limit=500):
        """
        This function precomputes a list of values for dfunc.
        """
        upper = self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
        self.dfunc_values = dfunc(upper)

    def forward(self, x):
        return self.act_func(x)

    def retrieve_from_precompute(self, precomputed_d, input_bound, default_d):
        """
        precomputed_d: The precomputed tangent points.
        input_bound: The input bound of the function.
        default_d: If input bound goes out of precompute range, we will use default_d.
        All of the inputs should share the same shape.
        """

        # divide input bound into number of steps to the inflection point (at x=0)
        index = torch.max(
            torch.zeros(input_bound.numel(), dtype=torch.long, device=input_bound.device),
            (input_bound / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        # If precompute range is smaller than input, tangent points will be taken from default.
        # The default value should be a guaranteed bound
        if index.max() >= precomputed_d.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            return torch.where(
                (index < precomputed_d.numel()).view(input_bound.shape),
                torch.index_select(
                    precomputed_d, 0, index.clamp(max=precomputed_d.numel() - 1)
                ).view(input_bound.shape),
                default_d,
            ).view(input_bound.shape)
        else:
            return torch.index_select(precomputed_d, 0, index).view(input_bound.shape)

    def generate_d_lower_upper(self, lower, upper):
        # Indices of neurons with input upper bound >=0, whose optimal slope to
        # lower bound the function was pre-computed.
        # Note that for neurons with also input lower bound >=0,
        # they will be masked later.
        d_lower = self.retrieve_from_precompute(self.d_lower, upper, lower)

        # Indices of neurons with lower bound <=0, whose optimal slope to upper
        # bound the function was pre-computed.
        d_upper = self.retrieve_from_precompute(self.d_upper, -lower, upper)
        return d_lower, d_upper

    def retrieve_d_from_k(self, k, func):
        d_indices = torch.searchsorted(torch.flip(self.dfunc_values, [0]), k, right=False)
        d_indices = self.num_points_pre - d_indices + 4
        d_left = d_indices * self.step_pre
        d_right = d_left + self.step_pre
        y_left = func(d_left)
        y_right = func(d_right)
        k_left = self.dfunc_values[d_indices]
        k_right = self.dfunc_values[torch.clamp(d_indices+1, max=self.dfunc_values.shape[0]-1)]
        # We choose the intersection of two tangent lines
        d_return = (k_left * d_left - k_right * d_right - y_left + y_right) / (k_left - k_right).clamp(min=1e-8)
        mask_almost_the_same = abs(k_left - k_right) < 1e-5
        d_return[mask_almost_the_same] = d_left[mask_almost_the_same]
        y_d = k_left * (d_return - d_left) + y_left
        return d_return, y_d

    def bound_relax_impl_same_slope(self, x, func, dfunc):
        lower, upper = x.lower, x.upper
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)
        mask_almost_the_same = abs(upper - lower) < 1e-4
        k_direct[mask_almost_the_same] = dfunc(lower)[mask_almost_the_same]

        mask_direct_lower = k_direct <= dfunc(lower)
        mask_direct_upper = k_direct <= dfunc(upper)

        # We now find the tangent line with the same slope of k_direct
        # In the case of "mask_direct_lower(or upper)", there should be only one possible tangent point
        # at which we obtain the same slope within the interval [lower, upper]
        d, y_d = self.retrieve_d_from_k(k_direct, func)
        d[lower + upper < 0] *= -1  # This is the case "direct upper"
        y_d[lower + upper < 0] = 2 * func(torch.tensor(0)) - y_d[lower + upper < 0]
        d_clamped = torch.clamp(d, min=lower, max=upper)
        y_d[d_clamped != d] = func(d_clamped[d_clamped != d])
        self.add_linear_relaxation(
            mask=mask_direct_lower, type='lower', k=k_direct, x0=lower, y0=y_l
        )
        self.add_linear_relaxation(
            mask=mask_direct_lower, type='upper', k=k_direct, x0=d_clamped, y0=y_d
        )
        self.add_linear_relaxation(
            mask=mask_direct_upper, type='upper', k=k_direct, x0=upper, y0=y_u
        )
        self.add_linear_relaxation(
            mask=mask_direct_upper, type='lower', k=k_direct, x0=d_clamped, y0=y_d
        )
        # Now we turn to the case where no direct line can be used
        d_lower, d_upper = self.generate_d_lower_upper(lower, upper)
        mask_both = torch.logical_not(mask_direct_upper + mask_direct_lower)
        # To make sure upper and lower bounds have the same slope,
        # we need the two tangents to be symmetrical
        d_same_slope = torch.max(torch.abs(d_lower), torch.abs(d_upper))
        k = dfunc(d_same_slope)
        y_d_same_slope = func(d_same_slope)
        y_d_same_slope_opposite = 2*func(torch.tensor(0)) - y_d_same_slope
        self.add_linear_relaxation(
            mask=mask_both, type='upper', k=k, x0=d_same_slope, y0=y_d_same_slope
        )
        self.add_linear_relaxation(
            mask=mask_both, type='lower', k=k, x0=-d_same_slope, y0=y_d_same_slope_opposite
        )

    def bound_relax_impl(self, x, func, dfunc):
        lower, upper = x.lower, x.upper
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connecting the two endpoints of the function inside the interval:
        # (lower, func(lower)) and (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # Fixed bounds that cannot be optimized.
        # self.mask_neg are the masks for neurons with upper bound <= 0, i.e., the whole input interval lies below 0.
        # self.mask_pos are the masks for neurons with lower bound >= 0, i.e., the whole input interval lies above 0.
        # For negative intervals, we can derive the linear upper bound by connecting the two endpoints,
        # i.e., starting from (lower, func(lower)) and setting the slope to k_direct.
        self.add_linear_relaxation(
            mask=self.mask_neg, type='upper', k=k_direct, x0=lower, y0=y_l)
        # For positive intervals, we connect the two endpoints to find the linear lower bound instead.
        self.add_linear_relaxation(
            mask=self.mask_pos, type='lower', k=k_direct, x0=lower, y0=y_l)

        # Store the x-coordinates of the points of tangencies.
        # d_lower is the closest value to upper such that the tangent line at (d_lower, func(d_lower)) still lower-
        # bounds the function in interval (lower, upper).
        # d_upper is the closest value to lower such that the tangent line at (d_lower, func(d_lower)) still upper-
        # bounds the function in interval (lower, upper).
        # d_lower and d_upper can be regarded as the default points of tangencies to draw linear bounds through.
        d_lower, d_upper = self.generate_d_lower_upper(lower, upper)

        # self.mask_both is the masks for neurons where lower < 0 < upper, i.e., the input interval contains 0.
        # mask_direct_lower is the masks for neurons whose input interval contains zero and whose linear lower bound can
        # be derived by connecting the two endpoints.
        # mask_direct_upper is the masks for neurons whose input interval contains zero and whose linear upper bound can
        # be derived by connecting the two endpoints.
        if self.convex_concave is None:
            mask_direct_lower = k_direct < dfunc(lower)
            mask_direct_upper = k_direct < dfunc(upper)
        else:
            mask_direct_lower = torch.where(
                self.convex_concave,
                k_direct < dfunc(lower), k_direct > dfunc(upper))
            mask_direct_upper = torch.where(
                self.convex_concave,
                k_direct < dfunc(upper), k_direct > dfunc(lower))
        mask_direct_lower = torch.logical_and(mask_direct_lower, self.mask_both)
        mask_direct_upper = torch.logical_and(mask_direct_upper, self.mask_both)

        if self.opt_stage in ['opt', 'reuse']:
            if not hasattr(self, 'alpha'):
                # Raise an error if alpha is not created.
                self._no_bound_parameters()
            ns = self._start

            # Clamping is done here rather than after `opt.step()` call
            # because it depends on pre-activation bounds
            self.alpha[ns].data[0:2] = torch.max(
                torch.min(self.alpha[ns][0:2], upper), lower)
            self.alpha[ns].data[2:4] = torch.max(
                torch.min(self.alpha[ns][2:4], upper), lower)
            if self.convex_concave is None:
                self.alpha[ns].data[4:6] = torch.min(
                    self.alpha[ns][4:6], d_lower)
                self.alpha[ns].data[6:8] = torch.max(
                    self.alpha[ns][6:8], d_upper)
            else:
                self.alpha[ns].data[4:6, :] = torch.where(
                    self.convex_concave,
                    torch.max(lower, torch.min(self.alpha[ns][4:6, :], d_lower)),
                    torch.min(upper, torch.max(self.alpha[ns][4:6, :], d_lower))
                )
                self.alpha[ns].data[6:8, :] = torch.where(
                    self.convex_concave,
                    torch.min(upper, torch.max(self.alpha[ns][6:8, :], d_upper)),
                    torch.max(lower, torch.min(self.alpha[ns][6:8, :], d_upper))
                )

            # shape [2, out_c, n, c, h, w].
            tp_pos = self.alpha[ns][0:2]  # For upper bound relaxation
            tp_neg = self.alpha[ns][2:4]  # For lower bound relaxation
            tp_both_lower = self.alpha[ns][4:6]
            tp_both_upper = self.alpha[ns][6:8]

            # No need to use tangent line, when the tangent point is at the left
            # side of the preactivation lower bound. Simply connect the two sides.
            self.add_linear_relaxation(
                mask=mask_direct_lower, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_lower), type='lower',
                k=dfunc(tp_both_lower), x0=tp_both_lower, y0=func(tp_both_lower))

            self.add_linear_relaxation(
                mask=mask_direct_upper, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_upper), type='upper',
                k=dfunc(tp_both_upper), x0=tp_both_upper, y0=func(tp_both_upper))

            self.add_linear_relaxation(
                mask=self.mask_neg, type='lower', k=dfunc(tp_neg),
                x0=tp_neg, y0=func(tp_neg))
            self.add_linear_relaxation(
                mask=self.mask_pos, type='upper', k=dfunc(tp_pos),
                x0=tp_pos, y0=func(tp_pos))
        else:
            if self.opt_stage == 'init':
                # Initialize optimizable slope.
                tp_both_lower_init = d_lower.detach()
                tp_both_upper_init = d_upper.detach()

                if self.loose_threshold is not None:
                    # We will modify d_lower and d_upper inplace.
                    # So make a copy for these two.
                    tp_both_lower_init = tp_both_lower_init.clone()
                    tp_both_upper_init = tp_both_upper_init.clone()
                    # A different initialization if the pre-activation bounds
                    # are too loose
                    loose = torch.logical_or(lower < -self.loose_threshold,
                                            upper > self.loose_threshold)
                    d_lower[loose] = lower[loose]
                    d_upper[loose] = upper[loose]

                ns = self._start
                self.tp_both_lower_init[ns] = tp_both_lower_init
                self.tp_both_upper_init[ns] = tp_both_upper_init

            # Not optimized (vanilla CROWN bound).
            # Use the middle point slope as the lower/upper bound. Not optimized.
            m = (lower + upper) / 2
            y_m = func(m)
            k_m = dfunc(m)
            # Lower bound is the middle point slope for the case input upper bound <= 0.
            # Note that the upper bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_neg, type='lower', k=k_m, x0=m, y0=y_m)
            # Upper bound is the middle point slope for the case input lower bound >= 0.
            # Note that the lower bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_pos, type='upper', k=k_m, x0=m, y0=y_m)
            # Now handle the case where input lower bound <=0 and upper bound >= 0.
            # A tangent line starting at d_lower is guaranteed to be a lower bound given the input upper bound.
            k = dfunc(d_lower)
            # Another possibility is to use the direct line as the lower bound, when this direct line does not intersect with f.
            # This is only valid when the slope at the input lower bound has a slope greater than the direct line.
            self.add_linear_relaxation(mask=mask_direct_lower, type='lower', k=k_direct, x0=lower, y0=y_l)
            # Otherwise (i.e., when the input interval cross zero and mask_direct_lower is not true),
            # we do not use the direct line, we use the d_lower slope.
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_lower),
                type='lower', k=k, x0=d_lower, y0=func(d_lower))
            # Do the same for the upper bound side when input lower bound <=0 and upper bound >= 0.
            k = dfunc(d_upper)
            self.add_linear_relaxation(
                mask=mask_direct_upper, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct_upper),
                type='upper', k=k, x0=d_upper, y0=func(d_upper))

            if self.disable_smoothness_enhancement:
                return
            # Partially modify the linear bound computation for intervals that contains 0 so that the linear bound
            # changes smoothly w.r.t to the input bounds. For example, when we fix the input lower bound and drag the
            # input upper bound, we do not expect the linear bound to change abruptly at any point.
            # Therefore, under certain conditions, we do not use the above heuristics. Instead, we draw a tangent line
            # through the middle point (m, func(m)) where m = (lower + upper) / 2 and use it as a linear bound.
            if self.inverse_s_shape:
                # When the function has an inverse s-shape (such as pow3), we switch to drawing a tangent line through
                # the middle point as the lower bound when the default point of tangency is on the left of the middle
                # point. Otherwise, the lower bound will be too loose on the side of the input upper bound. The change
                # will make the bound on the other side a little bit looser as a tradeoff for overall tightness.
                self.add_linear_relaxation(
                    mask=torch.logical_and(self.mask_both, d_lower < m),
                    type='lower', k=k_m, x0=m, y0=y_m)
                # We make a similar change to the linear upper bound when the default point of tangency is on
                # the right of the middle point.
                self.add_linear_relaxation(
                    mask=torch.logical_and(self.mask_both, d_upper >= m),
                    type='upper', k=k_m, x0=m, y0=y_m)
            elif self.sigmoid_like_mask is not None:
                # self.sigmoid_like_mask is originally defined for periodic functions like sin and cos. It marks
                # intervals on the s-shaped or flipped-s-shaped parts of the function. Whether the part is flipped-s-
                # shaped is determined by comparing func(lower) and func(upper). Currently, some overall s-shaped
                # function, such as tanh and sigmoid, also has this mask. In the future, we will make it default for
                # both completely and partially s-shaped functions to reduce branching in the code.
                y_l = func(lower)
                y_u = func(upper)
                # If the input interval is on the s-shaped part of the function, we switch to drawing a tangent line
                # through the middle point as the lower bound when the default point of tangency is on the right of the
                # middle point.
                self.add_linear_relaxation(
                    mask=torch.logical_and(torch.logical_and(self.sigmoid_like_mask, y_l < y_u), d_lower >= m),
                    type='lower', k=k_m, x0=m, y0=y_m)
                # We switch to drawing a tangent line through the middle point as the upper bound when the default point
                # of tangency is on the left of the middle point.
                self.add_linear_relaxation(
                    mask=torch.logical_and(torch.logical_and(self.sigmoid_like_mask, y_l < y_u), d_upper < m),
                    type='upper', k=k_m, x0=m, y0=y_m)
                # If the input interval is on the flipped-s-shaped part of the function, we flip the condition as well
                # as whether we change the lower or upper bound.
                self.add_linear_relaxation(
                    mask=torch.logical_and(torch.logical_and(self.sigmoid_like_mask, y_l >= y_u), d_lower < m),
                    type='lower', k=k_m, x0=m, y0=y_m)
                self.add_linear_relaxation(
                    mask=torch.logical_and(torch.logical_and(self.sigmoid_like_mask, y_l >= y_u), d_upper >= m),
                    type='upper', k=k_m, x0=m, y0=y_m)
            else:
                # Handle simple cases where the function has the most common s shape. Now it serves as a safeguard
                # against any child operator class whose self.sigmoid_like_mask is uninitialized. Here self.mask_both is
                # equivalent to self.sigmoid_like_mask & (y_l < y_u) in the case above.
                self.add_linear_relaxation(
                    mask=torch.logical_and(self.mask_both, d_lower >= m),
                    type='lower', k=k_m, x0=m, y0=y_m)
                self.add_linear_relaxation(
                    mask=torch.logical_and(self.mask_both, d_upper < m),
                    type='upper', k=k_m, x0=m, y0=y_m)

    def bound_relax_branch(self, lb, ub):
        # For functions that are only partially s-shaped, such as sin and cos, the non-s-shaped intervals are re-bounded
        # here. This method returns the linear bound coefficients (lower_slope, lower_bias, upper_slope, upper_bias) of
        # the non-s-shaped intervals. For globally s-shaped functions like tanh and sigmoid, the method returns 0s.
        return 0., 0., 0., 0.

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        lb = x.lower
        ub = x.upper
        self.branch_input_domain(lb, ub)
        if self.activation_bound_option == 'same-slope':
            self.bound_relax_impl_same_slope(x, self.act_func, self.d_act_func)
        else:
            self.bound_relax_impl(x, self.act_func, self.d_act_func)
        lower_slope, lower_bias, upper_slope, upper_bias = self.bound_relax_branch(lb, ub)
        self.lw = self.lw * self.sigmoid_like_mask + self.branch_mask * lower_slope
        self.lb = self.lb * self.sigmoid_like_mask + self.branch_mask * lower_bias
        self.uw = self.uw * self.sigmoid_like_mask + self.branch_mask * upper_slope
        self.ub = self.ub * self.sigmoid_like_mask + self.branch_mask * upper_bias

    def get_split_mask(self, lower, upper, input_index):
        assert input_index == 0
        return torch.logical_and(
            upper - lower >= self.split_min_gap,
            torch.logical_or(upper >= self.split_range[0],
                             lower <= self.split_range[1])
        )

class BoundPow(BoundSShaped):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        self.exponent = 2
        super().__init__(attr, inputs, output_index, options)
        self.ibp_intermediate = False
        self.has_constraint = True

        def act_func(x):
            return torch.pow(x, self.exponent)
        self.act_func = act_func
        def d_act_func(x):
            return self.exponent * torch.pow(x, self.exponent - 1)
        self.d_act_func = d_act_func
        def d2_act_func(x):
            return self.exponent * (self.exponent - 1) * torch.pow(x, self.exponent - 2)
        self.d2_act_func = d2_act_func

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

    @torch.no_grad()
    def precompute_relaxation(self, func, dfunc, x_limit = 500):
        """
        This function precomputes the tangent lines that will be used as
        lower/upper bounds for S-shapes functions.
        """
        self.x_limit = x_limit
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
        self._noninteger_exponent = False  # Reset flag
        y_val = y.value

        # Only positive exponents are supported
        assert y_val > 0, f"Exponent must be positive. Got y={y_val}"

        if y_val == 1:
            # Identity: pow(x, 1) = x, bounds pass through directly.
            # Set exponent to 1 so that subsequent constraint logic (e.g., clamping)
            # sees the correct exponent instead of a stale default.
            self.exponent = 1
            return [(last_lA, last_uA), (None, None)], 0, 0
        elif y_val == int(y_val) and int(y_val) >= 2:
            # Integer exponent >= 2: use existing S-shaped logic
            x.upper = torch.max(x.upper, x.lower + 1e-8)
            self.exponent = int(y_val)
            if self.exponent % 2:
                self.precompute_relaxation(self.act_func, self.d_act_func)

            As, lbias, ubias = super().bound_backward(
                last_lA, last_uA, x, start_node, start_shape, **kwargs)
            return [As[0], (None, None)], lbias, ubias
        else:
            # Non-integer exponent: use simple convex/concave bounding
            # Requires x >= 0 for non-integer powers
            assert x.lower.min() >= 0, (
                f"x must be non-negative for non-integer exponent y={y_val}. "
                f"Got x.lower.min() = {x.lower.min()}"
            )
            x.upper = torch.max(x.upper, x.lower + 1e-8)
            self.exponent = y_val
            self._noninteger_exponent = True

            # Use BoundOptimizableActivation's bound_backward which properly handles
            # optimization mode and passes dim_opt to bound_relax
            As, lbias, ubias = BoundOptimizableActivation.bound_backward(
                self, last_lA, last_uA, x, start_node=start_node,
                start_shape=start_shape, **kwargs)
            return [As[0], (None, None)], lbias, ubias

    def bound_forward(self, dim_in, x, y):
        assert y.lower == y.upper  # exponent must be constant
        y_val = y.lower

        # Only positive exponents are supported
        assert y_val > 0, f"Exponent must be positive. Got y={y_val}"

        if y_val == 1:
            # Identity: pow(x, 1) = x, bounds pass through directly
            self.exponent = 1
            return x

        x.upper = torch.max(x.upper, x.lower + 1e-8)
        self.exponent = y_val

        if y_val == int(y_val) and int(y_val) >= 2:
            # Integer exponent >= 2: use existing logic
            self.exponent = int(y_val)
            if self.exponent % 2:
                self.precompute_relaxation(self.act_func, self.d_act_func)
            return super().bound_forward(dim_in, x)
        else:
            # Non-integer exponent: requires x >= 0
            assert x.lower.min() >= 0, (
                f"x must be non-negative for non-integer exponent y={y_val}. "
                f"Got x.lower.min() = {x.lower.min()}"
            )
            self._noninteger_exponent = True
            return self.bound_forward_noninteger(dim_in, x)

    def _init_opt_parameters_impl(self, size_spec, name_start, **kwargs):
        """Initialize alpha parameters for optimization.

        For non-integer exponents: 2 parameters (tangent point), like BoundSqrt.
        For integer exponents: 10 parameters (S-shaped), defer to parent.
        """
        if hasattr(self, '_noninteger_exponent') and self._noninteger_exponent:
            # Non-integer: 2 alpha parameters for tangent point
            # alpha[0] for lA backprop, alpha[1] for uA backprop
            l, u = self.inputs[0].lower, self.inputs[0].upper
            if 0 < self.exponent < 1:
                l = torch.clamp(l, min=1e-9)
            alpha = torch.empty(2, size_spec, *l.shape, device=l.device)
            alpha.data[:] = (l + u) / 2
            return alpha
        else:
            # Integer: use S-shaped parent implementation
            return super()._init_opt_parameters_impl(size_spec, name_start, **kwargs)

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

    def bound_relax(self, x, init=False, dim_opt=None):
        # Check if we're in non-integer mode
        if hasattr(self, '_noninteger_exponent') and self._noninteger_exponent:
            self.bound_relax_noninteger(x, init, dim_opt)
            return

        # For powers with odd exponents, such as x^3, the overall shape is inverse S-like.
        self.inverse_s_shape = self.exponent % 2 == 1
        if self.exponent % 2:
            self.inflections = [0.]
        else:
            self.extremes = [0.]
        super().bound_relax(x, init, dim_opt)

    def bound_relax_noninteger(self, x, init=False, dim_opt=None):
        """
        Linear relaxation for non-integer exponents with x >= 0.
        Uses simple convex/concave bounding (no S-shaped complexity needed).

        For 0 < y < 1: concave -> lower=secant, upper=tangent
        For y > 1: convex -> lower=tangent, upper=secant
        """
        if init:
            self.init_linear_relaxation(x, dim_opt)

        y = self.exponent
        l, u = x.lower, x.upper

        # For 0 < y < 1, clamp lower bound for derivative/tangent computations
        # to avoid infinite derivative at x=0. Secant uses original l (f(0)=0 is fine).
        if 0 < y < 1:
            l_tang = torch.clamp(l, min=1e-9)
        else:
            l_tang = l

        f_l, f_u = self.act_func(l), self.act_func(u)

        # Secant slope: uses original l (well-defined even at 0)
        k_secant = (f_u - f_l) / (u - l).clamp(min=1e-8)

        # Tangent point: alpha[0] is optimized for lA backprop, alpha[1] for uA.
        # Both entries are used as independent tangent points (like BoundSqrt).
        if self.opt_stage in ['opt', 'reuse'] and hasattr(self, 'alpha') and self._start in self.alpha:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], l_tang), u)
            m = self.alpha[self._start]
        else:
            m = (l_tang + u) / 2

        k_tangent = self.d_act_func(m)
        f_m = self.act_func(m)

        if 0 < y < 1:
            # Concave: lower=secant, upper=tangent
            self.add_linear_relaxation(mask=None, type='lower', k=k_secant, x0=l, y0=f_l)
            self.add_linear_relaxation(mask=None, type='upper', k=k_tangent, x0=m, y0=f_m)
        else:
            # Convex (y > 1): lower=tangent, upper=secant
            self.add_linear_relaxation(mask=None, type='lower', k=k_tangent, x0=m, y0=f_m)
            self.add_linear_relaxation(mask=None, type='upper', k=k_secant, x0=l, y0=f_l)

    def bound_forward_noninteger(self, dim_in, x):
        """
        Forward-mode bound propagation for non-integer exponents with x >= 0.

        For 0 < y < 1: concave -> lower=secant, upper=tangent
        For y > 1: convex -> lower=tangent, upper=secant
        """
        y = self.exponent
        l, u = x.lower, x.upper

        # For 0 < y < 1, clamp lower for tangent/derivative computations only
        if 0 < y < 1:
            l_tang = torch.clamp(l, min=1e-9)
        else:
            l_tang = l

        f_l, f_u = self.act_func(l), self.act_func(u)

        # Secant slope: uses original l (well-defined even at 0)
        k_secant = (f_u - f_l) / (u - l).clamp(min=1e-8)

        # Tangent at midpoint (using clamped lower for derivative safety)
        m = (l_tang + u) / 2
        k_tangent = self.d_act_func(m)
        f_m = self.act_func(m)
        b_tangent = f_m - k_tangent * m

        # Secant bias
        b_secant = f_l - k_secant * l

        if 0 < y < 1:
            # Concave: lower=secant, upper=tangent
            kl, bl = k_secant, b_secant
            ku, bu = k_tangent, b_tangent
        else:
            # Convex (y > 1): lower=tangent, upper=secant
            kl, bl = k_tangent, b_tangent
            ku, bu = k_secant, b_secant

        # Propagate linear bounds
        lw = x.lw * kl.unsqueeze(1)
        lb = x.lb * kl + bl
        uw = x.uw * ku.unsqueeze(1)
        ub = x.ub * ku + bu

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        exp = v[1][0]
        h_L, h_U = v[0][0], v[0][1]

        # Only positive exponents are supported
        assert exp > 0, f"Exponent must be positive. Got y={exp}"

        if exp == 1:
            # Identity: pow(x, 1) = x, valid for any x including negative.
            return h_L, h_U

        if exp == int(exp) and int(exp) >= 2:
            # Integer exponent >= 2: existing logic
            exp = int(exp)
            pl, pu = torch.pow(h_L, exp), torch.pow(h_U, exp)
            if exp % 2 == 1:
                return pl, pu
            else:
                pl, pu = torch.min(pl, pu), torch.max(pl, pu)
                mask = 1 - ((h_L < 0) * (h_U > 0)).to(pl.dtype)
                return pl * mask, pu
        else:
            # Non-integer exponent: requires x > 0
            assert h_L.min() >= 0, (
                f"x must be non-negative for non-integer exponent y={exp}. "
                f"Got h_L.min() = {h_L.min()}"
            )
            self.exponent = exp
            # For y > 0, function is always increasing: f(l) <= f(x) <= f(u)
            return torch.pow(h_L, exp), torch.pow(h_U, exp)

    def clamp_interim_bounds(self):
        if self.exponent % 2 == 0:
            self.cstr_lower = self.lower.clamp(min=0)
            self.cstr_upper = self.upper.clamp(min=0)
            self.cstr_interval = (self.cstr_lower, self.cstr_upper)


def dtanh(x):
    return 1 - torch.tanh(x).pow(2)

def dsigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def darctan(x):
    return (x.square() + 1.).reciprocal()

def d2tanh(x):
    return -2 * torch.tanh(x) * (1 - torch.tanh(x).pow(2))

def d2sigmoid(x):
    return dsigmoid(x) * (1 - 2 * torch.sigmoid(x))


class BoundTanh(BoundSShaped):
    """
    BoundTanh is based on the S-shaped BoundSShaped. In the meantime, it works as the
    base class for other globally S-shaped functions such as Sigmoid and Atan.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None,
                 activation=('tanh', torch.tanh, dtanh), precompute=True):
        super().__init__(attr, inputs, output_index, options, activation, precompute)


    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        return super()._init_opt_parameters_impl(size_spec, name_start, num_params=8)

    def build_gradient_node(self, grad_upstream):
        node_grad = TanhGrad()
        grad_input = (grad_upstream, self.inputs[0].forward_value)
        grad_extra_nodes = [self.inputs[0]]
        return [(node_grad, grad_input, grad_extra_nodes)]


class TanhGradOp(Function):
    @staticmethod
    def symbolic(_, preact):
        return _.op('grad::Tanh', preact).setType(preact.type())
    
    @staticmethod
    def forward(ctx, preact):
        return 1 - torch.tanh(preact)**2


class TanhGrad(Module):
    def forward(self, g, preact):
        return g * TanhGradOp.apply(preact).unsqueeze(1)


class BoundTanhGrad(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None,
                 activation=('tanh', dtanh, d2tanh), precompute=True):
        super().__init__(attr, inputs, output_index, options)
        self.requires_input_bounds = [0]
        # The inflection point is where d2f/dx2 = 0.
        self.inflection_point = 0.6585026
        self.func = activation[1]
        self.dfunc = activation[2]
        if precompute:
            self.precompute_relaxation()

    def forward(self, x):
        return self.func(x)

    def interval_propagate(self, *v):
        lower, upper = v[0]
        f_lower = self.func(lower)
        f_upper = self.func(upper)
        next_lower = torch.min(f_lower, f_upper)
        next_upper = torch.max(f_lower, f_upper)
        mask_both = torch.logical_and(lower < 0, upper > 0)
        next_upper[mask_both] = self.func(torch.tensor(0))
        return next_lower, next_upper
    
    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        return self.bound_relax_impl(x)
    
    def precompute_relaxation(self, x_limit=500):
        """
        This function precomputes the tangent lines that will be used as
        the lower/upper bounds for bell-shaped functions.
        Three tensors are precomputed:
        - self.precompute_x: The x values of the upper preactivation bound.
        - self.d_lower: The tangent points of the lower bound.
        - self.d_upper: The tangent points of the upper bound.
        """

        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)

        max_iter = 100
        func, dfunc = self.func, self.dfunc

        logger.debug('Precomputing relaxation for %s (pre-activation limit: %f)',
                     self.__class__.__name__, x_limit)

        def check_lower(upper, d):
            """Given two points upper, d (d <= upper),
            check if the slope at d will be less than f(upper) at upper."""
            k = dfunc(d)
            # Return True if the slope is a lower bound.
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            """Given two points lower, d (d <= lower),
            check if the slope at d will be greater than f(lower) at lower."""
            k = dfunc(d)
            # Return True if the slope is a upper bound.
            return k * (lower - d) + func(d) >= func(lower)

        self.precompute_x = torch.arange(-self.x_limit, self.x_limit + self.step_pre, self.step_pre, device=self.device)
        self.d_lower = torch.zeros_like(self.precompute_x)
        self.d_upper = torch.zeros_like(self.precompute_x)

        # upper point that needs lower precomputed tangent line
        mask_need_d_lower = self.precompute_x >= -self.inflection_point
        upper = self.precompute_x[mask_need_d_lower] 
        # 1. Initial guess, the tangent is at -2*inflection_point (should be between (-inf, -inflection_point))
        r = -self.inflection_point * torch.ones_like(upper)
        l = -2 * self.inflection_point * torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an lower bound at f(upper).
            checked = check_lower(upper, l).int()
            # If the initial guess is not smaller enough, then double it (-2, -4, etc).
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
        # We want to further tighten this bound by moving it closer to upper.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_lower(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to lower bound the function.
        self.d_lower[mask_need_d_lower] = l.clone()

        # upper point that needs upper precomputed tangent line
        mask_need_upper_d = self.precompute_x >= self.inflection_point
        upper = self.precompute_x[mask_need_upper_d]
        # 1. Initial guess, the tangent is at inflection_point/2 (should be between (0, inflection_point))
        r = self.inflection_point * torch.ones_like(upper)
        l = self.inflection_point / 2 * torch.ones_like(upper)
        while True:
            # Check if the tangent line at the guessed point is an upper bound at f(upper).
            checked = check_upper(upper, l).int()
            # If the initial guess is not smaller enough, then reduce it.
            l = checked * l + (1 - checked) * (l / 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an upper bound at f(upper).
        # We want to further tighten this bound by moving it closer to upper.
        for _ in range(max_iter):
            # Binary search.
            m = (l + r) / 2
            checked = check_upper(upper, m).int()
            l = checked * m + (1 - checked) * l
            r = checked * r + (1 - checked) * m
        # At upper, a line with slope l is guaranteed to upper bound the function.
        self.d_upper[mask_need_upper_d] = l.clone()

    def retrieve_from_precompute(self, x, flip=False):
        if not flip:
            if x.max() > self.x_limit:
                warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Take the left endpoint of the interval
            x_indices = torch.searchsorted(self.precompute_x, x, right=True) - 1
            return self.d_lower[x_indices], self.d_upper[x_indices]
        else:
            if x.min() < -self.x_limit:
                warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Take the right endpoint of the interval
            x_indices = torch.searchsorted(self.precompute_x, -x, right=False)
            return -self.d_lower[x_indices], -self.d_upper[x_indices]
            

    def bound_relax_impl(self, x):
        lower, upper = x.lower, x.upper
        func, dfunc = self.func, self.dfunc
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # The tangent line at the midpoint can be a good approximation
        midpoint = (lower + upper) / 2
        k_midpoint = dfunc(midpoint)
        y_midpoint = func(midpoint)

        # If -inflection_point <= lower < upper <= inflection_point,
        # we call it "completely concave" region.
        mask_completely_concave = torch.logical_and(
            lower >= -self.inflection_point,
            upper <= self.inflection_point
        )
        self.add_linear_relaxation(
            mask=mask_completely_concave, type='lower', k=k_direct, x0=lower, y0=y_l)
        self.add_linear_relaxation(
            mask=mask_completely_concave, type='upper', k=k_midpoint, x0=midpoint, y0=y_midpoint)
        
        # From now on, we assume at least one of the bounds is outside the completely concave region.
        # Without loss of generality, we assume upper > inflection_point (indicated by mask_right).
        mask_right = lower + upper >= 0

        dl, du = self.retrieve_from_precompute(upper, flip=False)
        dl_, du_ = self.retrieve_from_precompute(lower, flip=True)

        # Case 1: Similar to a convex function
        mask_case1 = torch.logical_or(
            torch.logical_and(mask_right, lower >= self.inflection_point),
            torch.logical_and(torch.logical_not(mask_right), upper <= -self.inflection_point)
        )
        self.add_linear_relaxation(
            mask=mask_case1, type='upper', k=k_direct, x0=lower, y0=y_l)
        self.add_linear_relaxation(
            mask=mask_case1, type='lower', k=k_midpoint, x0=midpoint, y0=y_midpoint)
        
        # Case 2: Similar to a S-shaped function
        mask_case2_right = torch.logical_and(mask_right, torch.logical_and(
            upper > self.inflection_point, lower < self.inflection_point))
        # The upper tangent point is lineraly interpolated between 0 and du,
        # given lower ranging between -upper and du.
        d_mask_case2_right_upper = du * (lower + upper) / (du + upper)
        k_mask_case2_right_upper = dfunc(d_mask_case2_right_upper)
        y_mask_case2_right_upper = func(d_mask_case2_right_upper)
        self.add_linear_relaxation(
            mask=mask_case2_right, type='upper',
            k=k_mask_case2_right_upper, x0=d_mask_case2_right_upper, y0=y_mask_case2_right_upper)
        # The lower tangent point is found based on lower.
        d_mask_case2_right_lower = (dl_ + upper) / 2
        k_mask_case2_right_lower = dfunc(d_mask_case2_right_lower)
        y_mask_case2_right_lower = func(d_mask_case2_right_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_right, dl_ < upper), type='lower',
            k=k_mask_case2_right_lower, x0=d_mask_case2_right_lower, y0=y_mask_case2_right_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_right, dl_ >= upper), type='lower',
            k=k_direct, x0=lower, y0=y_l)

        mask_case2_left = torch.logical_and(torch.logical_not(mask_right), torch.logical_and(
            lower < -self.inflection_point, upper > -self.inflection_point))
        # The upper tangent point is lineraly interpolated between du_ and 0,
        # given upper ranging between du_ and -lower.
        d_mask_case2_left_upper = du_ * (upper + lower) / (du_ + lower)
        k_mask_case2_left_upper = dfunc(d_mask_case2_left_upper)
        y_mask_case2_left_upper = func(d_mask_case2_left_upper)
        self.add_linear_relaxation(
            mask=mask_case2_left, type='upper',
            k=k_mask_case2_left_upper, x0=d_mask_case2_left_upper, y0=y_mask_case2_left_upper)
        # The lower tangent point is found based on upper.
        d_mask_case2_left_lower = (dl + lower) / 2
        k_mask_case2_left_lower = dfunc(d_mask_case2_left_lower)
        y_mask_case2_left_lower = func(d_mask_case2_left_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_left, dl > lower), type='lower',
            k=k_mask_case2_left_lower, x0=d_mask_case2_left_lower, y0=y_mask_case2_left_lower)
        self.add_linear_relaxation(
            mask=torch.logical_and(mask_case2_left, dl <= lower), type='lower',
            k=k_direct, x0=upper, y0=y_u)
        
        # If the lower and upper bounds are too close, we just use IBP bounds to avoid numerical issues.
        mask_very_close = upper - lower < 1e-6
        if mask_very_close.any():
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_neg), type='lower', k=0, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_neg), type='upper', k=0, x0=upper, y0=y_u)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_pos), type='lower', k=0, x0=upper, y0=y_u)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_pos), type='upper', k=0, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_both), type='lower', k=0, x0=lower, y0=torch.min(y_l, y_u))
            self.add_linear_relaxation(
                mask=torch.logical_and(mask_very_close, self.mask_both), type='upper', k=0, x0=upper, y0=torch.full_like(y_l, func(torch.tensor(0))))


class BoundSigmoid(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         activation=('sigmoid', torch.sigmoid, dsigmoid))
    
    def build_gradient_node(self, grad_upstream):
        node_grad = SigmoidGrad()
        grad_input = (grad_upstream, self.inputs[0].forward_value)
        grad_extra_nodes = [self.inputs[0]]
        return [(node_grad, grad_input, grad_extra_nodes)]


class SigmoidGradOp(Function):
    @staticmethod
    def symbolic(_, preact):
        return _.op('grad::Sigmoid', preact).setType(preact.type())
    
    @staticmethod
    def forward(ctx, preact):
        sigmoid_x = torch.sigmoid(preact)
        return sigmoid_x * (1 - sigmoid_x)


class SigmoidGrad(Module):
    def forward(self, g, preact):
        return g * SigmoidGradOp.apply(preact).unsqueeze(1)


class BoundSigmoidGrad(BoundTanhGrad):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None,
                 activation=('sigmoid', dsigmoid, d2sigmoid), precompute=True):
        super().__init__(attr, inputs, output_index, options, activation, precompute=False)
        self.inflection_point = 1.3169614
        if precompute:
            self.precompute_relaxation()


class BoundAtan(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         activation=('arctan', torch.arctan, darctan))
        self.split_range = (-torch.inf, torch.inf)

    def build_gradient_node(self, grad_upstream):
        node_grad = AtanGrad()
        grad_input = (grad_upstream, self.inputs[0].forward_value)
        grad_extra_nodes = [self.inputs[0]]
        return [(node_grad, grad_input, grad_extra_nodes)]


class AtanGrad(Module):
    def forward(self, g, preact):
        # arctan'(x) = 1 / (1 + x^2)
        return g / (1 + preact.square()).unsqueeze(1)


class BoundTan(BoundAtan):
    """
    The implementation of BoundTan is based on the S-shaped BoundAtan. We use the bounds from its
    inverse function and directly convert the bounds of the inverse function to bounds of the original
    function. This trick allows us to quickly implement bounds on inverse functions.
    """

    def forward(self, x):
        return torch.tan(x)

    def _check_bounds(self, lower, upper):
        # Lower and upper bounds must be within the same [-½π, ½π] region.
        lower_periods = torch.floor((lower + 0.5 * torch.pi) / torch.pi)
        upper_periods = torch.floor((upper + 0.5 * torch.pi) / torch.pi)
        if not torch.allclose(lower_periods, upper_periods):
            print('Tan preactivation lower bounds:\n', lower)
            print('Tan preactivation upper bounds:\n', upper)
            raise ValueError("BoundTan received pre-activation bounds that produce infinity. "
                    "The preactivation bounds are too loose. Try to reduce perturbation region.")
        # Return the period number for each neuron.
        # Period is 0 => bounds are within [-½π, ½π],
        # Period is 1 => bounds are within [-½π + π, ½π + π]
        # Period is -1 => bounds are within [-½π - π, ½π - π]
        return lower_periods

    def _init_masks(self, x):
        # The masks now must consider the periodicity.
        lower = torch.remainder(x.lower + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        upper = torch.remainder(x.upper + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        self.mask_pos = lower >= 0
        self.mask_neg = upper <= 0
        self.mask_both = torch.logical_not(torch.logical_or(self.mask_pos, self.mask_neg))

    def interval_propagate(self, *v):
        # We need to check if the input lower and upper bounds are within the same period.
        # Otherwise the bounds become infinity.
        concrete_lower, concrete_upper = v[0][0], v[0][1]
        self._check_bounds(concrete_lower, concrete_upper)
        return super().interval_propagate(*v)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        periods = self._check_bounds(x.lower, x.upper)
        periods = torch.pi * periods
        # Create a fake x with inversed lower and upper.
        inverse_x = lambda: None
        inverse_x.lower = torch.tan(x.lower)
        inverse_x.upper = torch.tan(x.upper)
        super().bound_relax(inverse_x, init=init, dim_opt=dim_opt)
        # Lower slope, lower bias, upper slope and upper bias are saved to
        # self.lw, self.lb, self.uw, self.ub. We need to reverse them.
        # E.g., y = self.lw * x + self.lb, now becomes x = 1./self.lw * y - self.lb / self.lw
        # Additionally, we need to add the missing ½π periods.
        new_upper_slope = 1. / self.lw
        new_upper_bias = - self.lb / self.lw - periods / self.lw
        new_lower_slope = 1. / self.uw
        new_lower_bias = - self.ub / self.uw - periods / self.uw

        # NaN can happen if lw=0 or uw=0 when the pre-activation bounds are too close
        # Replace the bounds with interval bounds.
        if (self.lw == 0).any():
            mask = self.lw == 0
            new_upper_slope[mask] = 0
            new_upper_bias[mask] = inverse_x.upper[mask]
        if (self.uw == 0).any():
            mask = self.uw == 0
            new_lower_slope[mask] = 0
            new_lower_bias[mask] = inverse_x.lower[mask]

        self.lw = new_lower_slope
        self.lb = new_lower_bias
        self.uw = new_upper_slope
        self.ub = new_upper_bias
