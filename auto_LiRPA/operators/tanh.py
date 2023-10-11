"""BoundTanh and similar ops."""
import warnings
import torch
from .base import *
from .activation_base import BoundOptimizableActivation


def dtanh(x):
    # to avoid bp error when cosh is too large
    # cosh(25.0)**2 > 1e21
    mask = torch.lt(torch.abs(x), 25.0).to(x.dtype)
    cosh = torch.cosh(mask * x + 1 - mask)
    return mask * (1. / cosh.pow(2))

def dsigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def darctan(x):
    return (x.square() + 1.).reciprocal()


class BoundTanh(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None,
                 activation=('tanh', torch.tanh, dtanh), precompute=True):
        super().__init__(attr, inputs, output_index, options)
        if options is None:
            options = {}
        self.splittable = True
        self.ibp_intermediate = True
        self.activation = activation
        self.activation_name = activation[0]
        self.activation_forward = activation[1]
        self.activation_backward = activation[2]
        if precompute:
            self.precompute_relaxation(*activation)
        # TODO make them configurable when implementing a general nonlinear activation.
        # Neurons whose gap between pre-activation bounds is smaller than this
        # threshold will be masked and don't need branching.
        self.split_min_gap = 1e-2 #1e-4
        # Neurons whose pre-activation bounds don't overlap with this range
        # are considered as stable (with values either 0 or 1) and don't need
        # branching.
        self.split_range = (-10, 10)
        # The initialization will be adjusted if the pre-activation bounds are too loose.
        self.loose_threshold = options.get('tanh', {}).get(
            'loose_threshold', None)

    def opt_init(self):
        super().opt_init()
        self.tp_both_lower_init = {}
        self.tp_both_upper_init = {}

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = l.shape
        # Alpha dimension is (8, output_shape, batch, *shape) for Tanh.
        alpha = torch.empty(8, size_spec, *shape, device=l.device)
        alpha.data[:4] = ((l + u) / 2)
        alpha.data[4:6] = self.tp_both_lower_init[name_start]
        alpha.data[6:8] = self.tp_both_upper_init[name_start]
        return alpha

    @torch.no_grad()
    def precompute_relaxation(self, name, func, dfunc, x_limit=500):
        """
        This function precomputes the tangent lines that will be used as
        lower/upper bounds for S-shapes functions.
        """
        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

        logger.debug('Precomputing relaxation for %s (pre-activation limit: %f)',
                     name, x_limit)

        def check_lower(upper, d):
            """Given two points upper, d (d <= upper), check if the slope at d will be less than f(upper) at upper."""
            k = dfunc(d)
            # Return True if the slope is a lower bound.
            return k * (upper - d) + func(d) <= func(upper)

        def check_upper(lower, d):
            """Given two points lower, d (d >= lower), check if the slope at d will be greater than f(lower) at lower."""
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

    def forward(self, x):
        return self.activation_forward(x)

    def bound_relax_impl(self, x, func, dfunc):
        lower, upper = x.lower, x.upper
        y_l, y_u = func(lower), func(upper)
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # Fixed bounds that cannot be optimized. self.mask_neg are the masks for neurons with upper bound <= 0.
        # Upper bound for the case of input lower bound <= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=self.mask_neg, type='upper', k=k_direct, x0=lower, y0=y_l)
        # Lower bound for the case of input upper bound >= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=self.mask_pos, type='lower', k=k_direct, x0=lower, y0=y_l)

        # Indices of neurons with input upper bound >=0, whose optimal slope to lower bound the function was pre-computed.
        # Note that for neurons with also input lower bound >=0, they will be masked later.
        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
            (upper / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        if index.max() >= self.d_lower.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Lookup the lower bound slope from the pre-computed table.
            d_lower = torch.where(
                (index < self.d_lower.numel()).view(lower.shape),
                torch.index_select(
                    self.d_lower, 0, index.clamp(max=self.d_lower.numel() - 1)
                ).view(lower.shape),
                lower,
                # If the pre-activation bounds are too loose, just use IBP.
                # torch.ones_like(index).to(lower) * (-100.)
            ).view(lower.shape)
        else:
            # Lookup the lower bound slope from the pre-computed table.
            d_lower = torch.index_select(
                self.d_lower, 0, index).view(lower.shape)

        # Indices of neurons with lower bound <=0, whose optimal slope to upper
        # bound the function was pre-computed.
        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            (lower / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        if index.max() >= self.d_upper.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Lookup the lower bound slope from the pre-computed table.
            d_upper = torch.where(
                (index < self.d_upper.numel()).view(upper.shape),
                torch.index_select(
                    self.d_upper, 0, index.clamp(max=self.d_upper.numel() - 1)
                ).view(upper.shape),
                upper,
            )
        else:
            d_upper = torch.index_select(
                self.d_upper, 0, index).view(upper.shape)

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
            self.alpha[ns].data[4:6] = torch.min(
                self.alpha[ns][4:6], d_lower)
            self.alpha[ns].data[6:8] = torch.max(
                self.alpha[ns][6:8], d_upper)

            # shape [2, out_c, n, c, h, w].
            tp_pos = self.alpha[ns][0:2]  # For upper bound relaxation
            tp_neg = self.alpha[ns][2:4]  # For lower bound relaxation
            tp_both_lower = self.alpha[ns][4:6]
            tp_both_upper = self.alpha[ns][6:8]

            # No need to use tangent line, when the tangent point is at the left
            # side of the preactivation lower bound. Simply connect the two sides.
            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(lower))
            self.add_linear_relaxation(
                mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct), type='lower',
                k=dfunc(tp_both_lower), x0=tp_both_lower)

            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(upper))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct), type='upper',
                k=dfunc(tp_both_upper), x0=tp_both_upper)

            self.add_linear_relaxation(
                mask=self.mask_neg, type='lower', k=dfunc(tp_neg), x0=tp_neg)
            self.add_linear_relaxation(
                mask=self.mask_pos, type='upper', k=dfunc(tp_pos), x0=tp_pos)
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
                    # tp_both_lower_init[loose] = lower[loose]
                    # tp_both_upper_init[loose] = upper[loose]

                ns = self._start
                self.tp_both_lower_init[ns] = tp_both_lower_init
                self.tp_both_upper_init[ns] = tp_both_upper_init

            # Not optimized (vanilla CROWN bound).
            # Use the middle point slope as the lower/upper bound. Not optimized.
            m = (lower + upper) / 2
            y_m = func(m)
            k = dfunc(m)
            # Lower bound is the middle point slope for the case input upper bound <= 0.
            # Note that the upper bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
            # Upper bound is the middle point slope for the case input lower bound >= 0.
            # Note that the lower bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

            # Now handle the case where input lower bound <=0 and upper bound >= 0.
            # A tangent line starting at d_lower is guaranteed to be a lower bound given the input upper bound.
            k = dfunc(d_lower)
            # Another possibility is to use the direct line as the lower bound, when this direct line does not intersect with f.
            # This is only valid when the slope at the input lower bound has a slope greater than the direct line.
            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(lower))
            self.add_linear_relaxation(mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            # Otherwise we do not use the direct line, we use the d_lower slope.
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct),
                type='lower', k=k, x0=d_lower)

            # Do the same for the upper bound side when input lower bound <=0 and upper bound >= 0.
            k = dfunc(d_upper)
            mask_direct = torch.logical_and(self.mask_both, k_direct < dfunc(upper))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct),
                type='upper', k=k, x0=d_upper)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        self.bound_relax_impl(
            x, self.activation_forward, self.activation_backward)

    def get_split_mask(self, lower, upper, input_index):
        assert input_index == 0
        return torch.logical_and(
            upper - lower >= self.split_min_gap,
            torch.logical_or(upper >= self.split_range[0],
                             lower <= self.split_range[1])
        )


class BoundSigmoid(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         activation=('sigmoid', torch.sigmoid, dsigmoid))


class BoundAtan(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options,
                         activation=('arctan', torch.arctan, darctan))
        self.split_range = (-torch.inf, torch.inf)


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
        super().bound_relax(inverse_x, init=init)
        # Lower slope, lower bias, upper slope and upper bias are saved to
        # self.lw, self.lb, self.uw, self.ub. We need to reverse them.
        # E.g., y = self.lw * x + self.lb, now becomes x = 1./self.lw * y - self.lb / self.lw
        # Additionally, we need to add the missing ½π periods.
        new_upper_slope = 1. / self.lw
        new_upper_bias = - self.lb / self.lw - periods / self.lw
        new_lower_slope = 1. / self.uw
        new_lower_bias = - self.ub / self.uw - periods / self.uw
        self.lw = new_lower_slope
        self.lb = new_lower_bias
        self.uw = new_upper_slope
        self.ub = new_upper_bias

