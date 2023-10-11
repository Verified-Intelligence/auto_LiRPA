"""Unary nonlinearities other than activation functions."""
import torch
import torch.nn.functional as F
from .activation_base import BoundActivation, BoundOptimizableActivation
from .base import *
from .clampmult import multiply_by_A_signs
from .tanh import BoundTanh


# TODO too much code in this class is a duplicate of BoundTanh
class BoundOptimizableNonLinear(BoundTanh):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        # FIXME temporary: precompute=False
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

        self.convex_concave = torch.logical_and(self.mask_both,
            (self.d2_act_func(lower) >= 0))
        self.concave_convex = torch.logical_xor(self.mask_both, self.convex_concave)

    # FIXME
    @torch.no_grad()
    def precompute_relaxation(self, func, dfunc, x_limit = 500):
        return super().precompute_relaxation('nonlinear', func, dfunc, x_limit)

    def generate_d_lower_upper(self, lower, upper):
        # Indices of neurons with input upper bound >=0, whose optimal slope to
        # lower bound the function was pre-computed.
        # Note that for neurons with also input lower bound >=0,
        # they will be masked later.
        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
            (upper / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        # Lookup the lower bound slope from the pre-computed table.
        d_lower = torch.index_select(self.d_lower, 0, index).view(lower.shape)

        # Indices of neurons with lower bound <=0, whose optimal slope to upper
        # bound the function was pre-computed.
        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            (lower / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        d_upper = torch.index_select(self.d_upper, 0, index).view(upper.shape)
        return d_lower, d_upper

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = [size_spec] + list(l.shape)
        alpha = torch.empty(10, *shape, device=l.device)
        alpha.data[:4] = ((l + u) / 2).unsqueeze(0).expand(4, *shape)
        alpha.data[4:6] = self.tp_both_lower_init[name_start].expand(2, *shape)
        alpha.data[6:8] = self.tp_both_upper_init[name_start].expand(2, *shape)
        return alpha

    def bound_relax_impl_sigmoid(self, lb, ub, func, dfunc):
        # When self.x_limit is large enough, torch.tanh(self.x_limit)=1,
        # and thus clipping is valid
        lower = lb
        upper = ub
        y_l, y_u = func(lower), func(upper)

        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # Fixed bounds that cannot be optimized. self.mask_neg are the masks for neurons with upper bound <= 0.
        # Upper bound for the case of input lower bound <= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=self.mask_neg, type='upper', k=k, x0=lower, y0=y_l)
        # Lower bound for the case of input upper bound >= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=self.mask_pos, type='lower', k=k, x0=lower, y0=y_l)

        if self.use_precompute:
            d_lower, d_upper = self.generate_d_lower_upper(lower, upper)
        else:
            d_lower = self.convex_concave * lower + self.concave_convex * upper
            d_upper = self.convex_concave * upper + self.concave_convex * lower

        if self.opt_stage in ['opt', 'reuse']:
            if not hasattr(self, 'alpha'):
                # Raise an error if alpha is not created.
                self._no_bound_parameters()
            ns = self._start

            self.alpha[ns].data[0:2, :] = torch.max(
                torch.min(self.alpha[ns][0:2, :], upper), lower)
            self.alpha[ns].data[2:4, :] = torch.max(
                torch.min(self.alpha[ns][2:4, :], upper), lower)
            self.alpha[ns].data[4:6, :] = (
                self.convex_concave * torch.max(lower, torch.min(self.alpha[ns][4:6, :], d_lower))
                + self.concave_convex * torch.min(upper, torch.max(self.alpha[ns][4:6, :], d_lower)))
            self.alpha[ns].data[6:8, :] = (
                self.convex_concave * torch.min(upper, torch.max(self.alpha[ns][6:8, :], d_upper))
                + self.concave_convex * torch.max(lower, torch.min(self.alpha[ns][6:8, :], d_upper)))

            # shape [2, out_c, n, c, h, w].
            tp_pos = self.alpha[ns][0:2, :]
            tp_neg = self.alpha[ns][2:4, :]
            tp_both_lower = self.alpha[ns][4:6, :]
            tp_both_upper = self.alpha[ns][6:8, :]

            # No need to use tangent line, when the tangent point is at the left
            # side of the preactivation lower bound. Simply connect the two sides.
            mask_direct = torch.logical_or(
                torch.logical_and(self.convex_concave, k_direct < dfunc(lower)),
                torch.logical_and(self.concave_convex, k_direct > dfunc(upper)))
            self.add_linear_relaxation(
                mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct), type='lower',
                k=dfunc(tp_both_lower), x0=tp_both_lower, y0=func(tp_both_lower))

            mask_direct = torch.logical_or(
                torch.logical_and(self.convex_concave, k_direct < dfunc(upper)),
                torch.logical_and(self.concave_convex, k_direct > dfunc(lower)))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct), type='upper',
                k=dfunc(tp_both_upper), x0=tp_both_upper, y0=func(tp_both_upper))

            self.add_linear_relaxation(
                mask=self.mask_neg, type='lower', k=dfunc(tp_neg),
                x0=tp_neg, y0=func(tp_neg))
            self.add_linear_relaxation(
                mask=self.mask_pos, type='upper', k=dfunc(tp_pos),
                x0=tp_pos, y0=func(tp_pos))
        else:
            # Not optimized (vanilla CROWN bound).
            # Use the middle point slope as the lower/upper bound. Not optimized.
            m = (lower + upper) / 2
            y_m = func(m)
            k = dfunc(m)
            # Lower bound is the middle point slope for the case input upper bound <= 0.
            # Note that the upper bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(
                mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
            # Upper bound is the middle point slope for the case input lower bound >= 0.
            # Note that the lower bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(
                mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

            # Now handle the case where input lower bound <=0 and upper bound >= 0.
            # A tangent line starting at d_lower is guaranteed to be a lower bound given the input upper bound.
            k = dfunc(d_lower)
            y0 = func(d_lower)
            if self.opt_stage == 'init':
                # Initialize optimizable slope.
                ns = self._start
                self.tp_both_lower_init[ns] = d_lower.detach()
            # Another possibility is to use the direct line as the lower bound, when this direct line does not intersect with f.
            # This is only valid when the slope at the input lower bound has a slope greater than the direct line.
            mask_direct = torch.logical_or(
                torch.logical_and(self.convex_concave, k_direct < dfunc(lower)),
                torch.logical_and(self.concave_convex, k_direct > dfunc(upper)))
            self.add_linear_relaxation(
                mask=mask_direct, type='lower', k=k_direct, x0=lower, y0=y_l)
            # Otherwise we do not use the direct line, we use the d_lower slope.
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct),
                type='lower', k=k, x0=d_lower, y0=y0)

            # Do the same for the upper bound side when input lower bound <=0 and upper bound >= 0.
            k = dfunc(d_upper)
            y0 = func(d_upper)
            if self.opt_stage == 'init':
                ns = self._start
                self.tp_both_upper_init[ns] = d_upper.detach()
                self.tmp_lower = lb.detach()
                self.tmp_upper = ub.detach()
            mask_direct = torch.logical_or(
                torch.logical_and(self.convex_concave, k_direct < dfunc(upper)),
                torch.logical_and(self.concave_convex, k_direct > dfunc(lower)))
            self.add_linear_relaxation(
                mask=mask_direct, type='upper', k=k_direct, x0=lower, y0=y_l)
            self.add_linear_relaxation(
                mask=torch.logical_xor(self.mask_both, mask_direct),
                type='upper', k=k, x0=d_upper, y0=y0)

    def generate_inflections(self, lb, ub):
        raise NotImplementedError

    def bound_relax_impl(self, lb, ub):
        raise NotImplementedError

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        lb = x.lower
        ub = x.upper
        self.generate_inflections(lb, ub)
        self.branch_input_domain(lb, ub)
        self.bound_relax_impl_sigmoid(lb, ub, self.act_func, self.d_act_func)
        lower_slope, lower_bias, upper_slope, upper_bias = self.bound_relax_impl(lb, ub)
        self.lw = self.lw * self.sigmoid_like_mask + self.branch_mask * lower_slope
        self.lb = self.lb * self.sigmoid_like_mask + self.branch_mask * lower_bias
        self.uw = self.uw * self.sigmoid_like_mask + self.branch_mask * upper_slope
        self.ub = self.ub * self.sigmoid_like_mask + self.branch_mask * upper_bias


class BoundExp(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.options = options.get('exp', {})
        self.max_input = 0

    def forward(self, x):
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(x, dim=-1, keepdim=True)[0].detach()
            return torch.exp(x - self.max_input)
        return torch.exp(x)

    def interval_propagate(self, *v):
        assert len(v) == 1
        # unary monotonous functions only
        h_L, h_U = v[0]
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(h_U, dim=-1, keepdim=True)[0]
            h_L, h_U = h_L - self.max_input, h_U - self.max_input
        else:
            self.max_input = 0
        return torch.exp(h_L), torch.exp(h_U)

    def bound_forward(self, dim_in, x):
        m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)

        exp_l, exp_m, exp_u = torch.exp(x.lower), torch.exp(m), torch.exp(x.upper)

        kl = exp_m
        lw = x.lw * kl.unsqueeze(1)
        lb = kl * (x.lb - m + 1)

        ku = (exp_u - exp_l) / (x.upper - x.lower + epsilon)
        uw = x.uw * ku.unsqueeze(1)
        ub = x.ub * ku - ku * x.lower + exp_l

        return LinearBound(lw, lb, uw, ub)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        # Special case when computing log_softmax (FIXME: find a better solution, this trigger condition is not reliable).
        if self.loss_fusion and last_lA is None and last_uA is not None and torch.min(
                last_uA) >= 0 and x.from_input:
            # Adding an extra bias term to the input. This is equivalent to adding a constant and subtract layer before exp.
            # Note that we also need to adjust the bias term at the end.
            if self.options == 'no-detach':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0]
            elif self.options != 'no-max-input':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0].detach()
            else:
                self.max_input = 0
            adjusted_lower = x.lower - self.max_input
            adjusted_upper = x.upper - self.max_input
            # relaxation for upper bound only (used in loss fusion)
            exp_l, exp_u = torch.exp(adjusted_lower), torch.exp(adjusted_upper)
            k = (exp_u - exp_l) / (adjusted_upper - adjusted_lower).clamp(min=1e-8)
            if k.requires_grad:
                k = k.clamp(min=1e-8)
            uA = last_uA * k.unsqueeze(0)
            ubias = last_uA * (-adjusted_lower * k + exp_l).unsqueeze(0)

            if ubias.ndim > 2:
                ubias = torch.sum(ubias, dim=tuple(range(2, ubias.ndim)))
            # Also adjust the missing ubias term.
            if uA.ndim > self.max_input.ndim:
                A = torch.sum(uA, dim=tuple(range(self.max_input.ndim, uA.ndim)))
            else:
                A = uA

            # These should hold true in loss fusion
            assert self.batch_dim == 0
            assert A.shape[0] == 1

            batch_size = A.shape[1]
            ubias -= (A.reshape(batch_size, -1) * self.max_input.reshape(batch_size, -1)).sum(dim=-1).unsqueeze(0)
            return [(None, uA)], 0, ubias
        else:
            return super().bound_backward(last_lA, last_uA, x, **kwargs)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)
        min_val = -1e9
        l, u = x.lower.clamp(min=min_val), x.upper.clamp(min=min_val)
        if self.opt_stage in ['opt', 'reuse']:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], x.lower), x.upper)
            m = torch.min(self.alpha[self._start], x.lower + 0.99)
        else:
            m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)
        exp_l, exp_m, exp_u = torch.exp(x.lower), torch.exp(m), torch.exp(x.upper)
        k = exp_m
        self.add_linear_relaxation(mask=None, type='lower', k=k, x0=m, y0=exp_m)
        k = (exp_u - exp_l) / (u - l).clamp(min=1e-8)
        self.add_linear_relaxation(mask=None, type='upper', k=k, x0=l, y0=exp_l)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(2, size_spec, *l.shape, device=l.device)
        alpha.data[:2] = (l + u) / 2
        return alpha


class BoundLog(BoundActivation):

    def forward(self, x):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            return torch.logsumexp(self.inputs[0].inputs[0].inputs[0].forward_value, dim=-1)
        return torch.log(x.clamp(min=epsilon))

    def bound_relax(self, x, init=False):
        if init:
            self.init_linear_relaxation(x)
        rl, ru = self.forward(x.lower), self.forward(x.upper)
        ku = (ru - rl) / (x.upper - x.lower + epsilon)
        self.add_linear_relaxation(mask=None, type='lower', k=ku, x0=x.lower, y0=rl)
        m = (x.lower + x.upper) / 2
        k = torch.reciprocal(m)
        rm = self.forward(m)
        self.add_linear_relaxation(mask=None, type='upper', k=k, x0=m, y0=rm)

    def interval_propagate(self, *v):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            par = self.inputs[0].inputs[0].inputs[0]
            lower = torch.logsumexp(par.lower, dim=-1)
            upper = torch.logsumexp(par.upper, dim=-1)
            return lower, upper
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        A, lbias, ubias = super().bound_backward(last_lA, last_uA, x)
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            assert A[0][0] is None
            exp_module = self.inputs[0].inputs[0]
            ubias = ubias + self.get_bias(A[0][1], exp_module.max_input.squeeze(-1))
        return A, lbias, ubias


class BoundPow(BoundOptimizableNonLinear):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.ibp_intermediate = False
        self.use_precompute = True
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
            # Indices of neurons with input upper bound >=0, whose optimal slope to lower bound the function was pre-computed.
            # Note that for neurons with also input lower bound >=0, they will be masked later.
            index = torch.max(
                torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
                (upper / self.step_pre).to(torch.long).reshape(-1)
            ) + 1
            # Lookup the lower bound slope from the pre-computed table.
            d_upper = torch.index_select(self.d_upper, 0, index).view(lower.shape)

            # Indices of neurons with lower bound <=0, whose optimal slope to upper bound the function was pre-computed.
            index = torch.max(
                torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
                (lower / -self.step_pre).to(torch.long).reshape(-1)
            ) + 1
            d_lower = torch.index_select(self.d_lower, 0, index).view(upper.shape)
            return d_lower, d_upper
        else:
            return torch.zeros_like(upper), torch.zeros_like(upper)

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        shape = [size_spec] + list(l.shape)
        alpha = torch.empty(10, *shape, device=l.device)
        alpha.data[:4] = ((l + u) / 2).unsqueeze(0).expand(4, *shape)
        alpha.data[4:6] = self.tp_both_lower_init[name_start].expand(2, *shape)
        alpha.data[6:8] = self.tp_both_upper_init[name_start].expand(2, *shape)
        alpha.data[8:10] = torch.zeros(2, *shape)
        return alpha

    @torch.no_grad()
    def precompute_relaxation(self, func, dfunc, x_limit = 500):
        """
        This function precomputes the tangent lines that will be used as lower/upper bounds for S-shapes functions.
        """
        self.x_limit = x_limit
        self.step_pre = 0.01
        self.num_points_pre = int(self.x_limit / self.step_pre)
        max_iter = 100

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
            checked = check_upper(upper, l).int()
            # If the initial guess is not smaller enough, then double it (-2, -4, etc).
            l = checked * l + (1 - checked) * (l * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
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
        # Given an lower bound point (<=0), find a line that is guaranteed to be an upper bound of this function.
        lower = -self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device)
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

    def bound_relax_impl(self, lb, ub):
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


class BoundReciprocal(BoundOptimizableActivation):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True

    def forward(self, x):
        return torch.reciprocal(x)

    def interval_propagate(self, *v):
        h_L = v[0][0].to(dtype=torch.get_default_dtype())
        h_U = v[0][1].to(dtype=torch.get_default_dtype())
        assert h_L.min() > 0, 'Only positive values are supported in BoundReciprocal'
        return torch.reciprocal(h_U), torch.reciprocal(h_L)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)

        assert x.lower.min() > 0

        ku = -1. / (x.lower * x.upper)
        self.add_linear_relaxation(mask=None, type='upper', k=ku, x0=x.lower)

        if self.opt_stage in ['opt', 'reuse']:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], x.lower), x.upper)
            mid = self.alpha[self._start].clamp(min=0.01)
        else:
            mid = (x.lower + x.upper) / 2

        self.add_linear_relaxation(
            mask=None, type='lower', k=-1./(mid**2), x0=mid)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(2, size_spec, *l.shape, device=l.device)
        alpha.data[:2] = (l + u) / 2
        return alpha


class BoundSqrt(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_prior_constraint = True
        self.has_constraint = True

    def forward(self, x):
        return torch.sqrt(x)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)

        if self.opt_stage in ['opt', 'reuse']:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], x.lower), x.upper)
            mid = self.alpha[self._start]
        else:
            mid = (x.lower + x.upper) / 2
        k = 0.5 / self.forward(mid)
        self.add_linear_relaxation(mask=None, type='upper', k=k, x0=mid)

        sqrt_l = self.forward(x.lower)
        sqrt_u = self.forward(x.upper)
        k = (sqrt_u - sqrt_l) / (x.upper - x.lower).clamp(min=1e-8)
        self.add_linear_relaxation(mask=None, type='lower', k=k, x0=x.lower)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        if self.use_prior_constraint and self.check_constraint_available(x):
            if hasattr(x, 'cstr_interval'):
                del x.cstr_interval
                del x.cstr_lower
                del x.cstr_upper

            x_l, x_u = self._ibp_constraint(x, delete_bounds_after_use=True)
            x_u = torch.max(x_u, x_l + 1e-8)
        return super().bound_backward(last_lA, last_uA, x, **kwargs)

    def clamp_interim_bounds(self):
        self.cstr_lower = self.lower.clamp(min=0)
        self.cstr_upper = self.upper.clamp(min=0)
        self.cstr_interval = (self.cstr_lower, self.cstr_upper)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(2, size_spec, *l.shape, device=l.device)
        alpha.data[:2] = (l + u) / 2
        return alpha


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


class BoundMinMax(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.options = options
        self.requires_input_bounds = [0, 1]
        self.op = None

    def _init_opt_parameters_impl(self, size_spec, name_start):
        """Implementation of init_opt_parameters for each start_node."""
        l = self.inputs[0].lower
        # Alpha dimension is (8, output_shape, batch, *shape) for Tanh.
        return torch.ones_like(l).unsqueeze(0).repeat(2, *[1] * l.ndim)

    def clip_alpha(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0., 1.)

    def forward(self, x, y):
        if self.op == 'max':
            return torch.max(x, y)
        elif self.op == 'min':
            return torch.min(x, y)
        else:
            raise NotImplementedError

    def _backward_relaxation(self, last_lA, last_uA, x, y, start_node):
        lb_x = x.lower
        ub_x = x.upper
        lb_y = y.lower
        ub_y = y.upper

        ub_x = torch.max(ub_x, lb_x + 1e-8)
        ub_y = torch.max(ub_y, lb_y + 1e-8)

        if self.opt_stage in ['opt', 'reuse']:
            selected_alpha = self.alpha[start_node.name]
            alpha_u = selected_alpha[0].squeeze(0)
            alpha_l = selected_alpha[1].squeeze(0)
        else:
            alpha_u = alpha_l = 1

        # Generate masks for stable and unstable neurons
        # Neurons are stable when x, y bounds fall in z=x or z=y plane
        x_mask = (lb_x >= ub_y).requires_grad_(False).to(lb_x.dtype)
        y_mask = (lb_y >= ub_x).requires_grad_(False).to(lb_y.dtype)
        no_mask = (1. - x_mask) * (1. - y_mask)

        # Calculate dx, dy, b coefficients according to https://www.overleaf.com/read/dbyyfpjhhwbk
        if self.op == 'max':
            upper_dx = x_mask + no_mask * (
                (ub_y - ub_x) / (alpha_u * (lb_x - ub_x)))
            upper_dy = y_mask + no_mask * (
                (alpha_u - 1) * (ub_y - ub_x)) / (alpha_u * (ub_y - lb_y))
            upper_b = no_mask * (
                ub_x - (ub_x * (ub_y - ub_x)) / (alpha_u * (lb_x - ub_x))
                - ((alpha_u - 1) * (ub_y - ub_x) * lb_y) / (
                    alpha_u * (ub_y - lb_y)))
            lower_dx = x_mask + no_mask * alpha_l
            lower_dy = y_mask + no_mask * (1 - alpha_l)
            lower_b = None
        elif self.op == 'min':
            lower_dx = y_mask + no_mask * (
                (lb_x - lb_y) / (alpha_u * (lb_x - ub_x)))
            lower_dy = x_mask + no_mask * (
                (alpha_u - 1) * (lb_x - lb_y)) / (alpha_u * (ub_y - lb_y))
            lower_b = no_mask * (
                lb_y - (ub_x * (lb_x - lb_y)) / (alpha_u * (lb_x - ub_x))
                - ((alpha_u - 1) * (lb_x - lb_y) * lb_y) / (
                    alpha_u * (ub_y - lb_y)))
            upper_dx = y_mask + no_mask * alpha_l
            upper_dy = x_mask + no_mask * (1 - alpha_l)
            upper_b = None
        else:
            raise NotImplementedError

        upper_dx = upper_dx.unsqueeze(0)
        upper_dy = upper_dy.unsqueeze(0)
        lower_dx = lower_dx.unsqueeze(0)
        lower_dy = lower_dy.unsqueeze(0)
        if upper_b is not None:
            upper_b = upper_b.unsqueeze(0)
        if lower_b is not None:
            lower_b = lower_b.unsqueeze(0)

        return upper_dx, upper_dy, upper_b, lower_dx, lower_dy, lower_b

    def bound_backward(self, last_lA, last_uA, x=None, y=None, start_shape=None,
                       start_node=None, **kwargs):
        # Get element-wise CROWN linear relaxations.
        upper_dx, upper_dy, upper_b, lower_dx, lower_dy, lower_b = \
            self._backward_relaxation(last_lA, last_uA, x, y, start_node)

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
        upper_dx = maybe_unfold_patches(upper_dx, last_lA if last_lA is not None else last_uA)
        upper_dy = maybe_unfold_patches(upper_dy, last_lA if last_lA is not None else last_uA)
        lower_dx = maybe_unfold_patches(lower_dx, last_lA if last_lA is not None else last_uA)
        lower_dy = maybe_unfold_patches(lower_dy, last_lA if last_lA is not None else last_uA)
        upper_b = maybe_unfold_patches(upper_b, last_lA if last_lA is not None else last_uA)
        lower_b = maybe_unfold_patches(lower_b, last_lA if last_lA is not None else last_uA)

        uAx, ubias = _bound_oneside(last_uA, upper_dx, lower_dx, upper_b, lower_b)
        uAy, ubias = _bound_oneside(last_uA, upper_dy, lower_dy, upper_b, lower_b)
        lAx, lbias = _bound_oneside(last_lA, lower_dx, upper_dx, lower_b, upper_b)
        lAy, lbias = _bound_oneside(last_lA, lower_dy, upper_dy, lower_b, upper_b)

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


class BoundGELU(BoundOptimizableNonLinear):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.ibp_intermediate = False
        self.use_precompute = True
        self.act_func = F.gelu
        def d_act_func(x):
            return 0.5 * (1 + torch.erf(x / np.sqrt(2))) + x * torch.exp(-0.5 * x ** 2) / np.sqrt(2 * torch.pi)
        self.d_act_func = d_act_func
        def d2_act_func(x):
            return 2 * torch.exp(-0.5 * x ** 2) / np.sqrt(2 * torch.pi) - x ** 2 * torch.exp(-0.5 * x ** 2) / np.sqrt(2 * torch.pi)
        self.d2_act_func = d2_act_func
        self.precompute_relaxation('gelu', self.act_func, self.d_act_func)

    def _init_masks(self, x):
        lower = x.lower
        upper = x.upper
        self.mask_left_pos = torch.logical_and(lower >= -np.sqrt(2), upper <= 0)
        self.mask_left_neg = upper <= -np.sqrt(2)
        self.mask_left = torch.logical_xor(upper <= 0,
                torch.logical_or(self.mask_left_pos, self.mask_left_neg))

        self.mask_right_pos = lower >= np.sqrt(2)
        self.mask_right_neg = torch.logical_and(upper <= np.sqrt(2), lower >= 0)
        self.mask_right = torch.logical_xor(lower >= 0,
                torch.logical_or(self.mask_right_pos, self.mask_right_neg))

        self.mask_2 = torch.logical_and(torch.logical_and(upper > 0, upper <= np.sqrt(2)),
                    torch.logical_and(lower < 0, lower >= -np.sqrt(2)))
        self.mask_left_3 = torch.logical_and(lower < -np.sqrt(2), torch.logical_and(
            upper > 0, upper <= np.sqrt(2)))
        self.mask_right_3 = torch.logical_and(upper > np.sqrt(2), torch.logical_and(
            lower < 0, lower >= -np.sqrt(2)))
        self.mask_4 = torch.logical_and(lower < -np.sqrt(2), upper > np.sqrt(2))
        self.mask_both = torch.logical_or(self.mask_2, torch.logical_or(self.mask_4,
                    torch.logical_or(self.mask_left_3, self.mask_right_3)))

    @torch.no_grad()
    def precompute_relaxation(self, name, func, dfunc, x_limit=1000):
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
        upper = self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device) + np.sqrt(2)
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
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
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
        # Given an lower bound point (<=0), find a line that is guaranteed to be an upper bound of this function.
        lower = (-self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device) + np.sqrt(2)).clamp(min=0.01)
        l = torch.zeros_like(upper) + np.sqrt(2)
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

        upper = -self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device) - np.sqrt(2)
        r = torch.zeros_like(upper) - 0.7517916
        # Initial guess, the tangent line is at -1.
        l = torch.zeros_like(upper) - np.sqrt(2)
        while True:
            checked = check_lower(upper, r).int()
            r = checked * r + (1 - checked) * (r * 2)
            if checked.sum() == l.numel():
                break
        # Now we have starting point at l, its tangent line is guaranteed to be an lower bound at f(upper).
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
        # Given an lower bound point (<=0), find a line that is guaranteed to be an upper bound of this function.
        lower = (self.step_pre * torch.arange(0, self.num_points_pre + 5, device=self.device) - np.sqrt(2)).clamp(max=0)
        l = torch.zeros_like(upper) - x_limit
        r = torch.zeros_like(upper) - np.sqrt(2)
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
        # k_direct is the slope of the line directly connect (lower, func(lower)), (upper, func(upper)).
        k_direct = k = (y_u - y_l) / (upper - lower).clamp(min=1e-8)

        # Fixed bounds that cannot be optimized. self.mask_neg are the masks for neurons with upper bound <= 0.
        # Upper bound for the case of input lower bound <= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=torch.logical_or(torch.logical_or(self.mask_left_pos,
                    self.mask_right_neg), self.mask_both), type='upper', k=k_direct, x0=lower, y0=y_l)
        # Lower bound for the case of input upper bound >= 0, is always the direct line.
        self.add_linear_relaxation(
            mask=torch.logical_or(self.mask_left_neg,
                    self.mask_right_pos), type='lower', k=k_direct, x0=lower, y0=y_l)

        # Indices of neurons with input upper bound >=0, whose optimal slope to lower bound the function was pre-computed.
        # Note that for neurons with also input lower bound >=0, they will be masked later.
        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=upper.device),
            ((upper - np.sqrt(2)) / self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        if index.max() >= self.d_lower_right.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Lookup the lower bound slope from the pre-computed table.
            d_lower_right = torch.where(
                (index < self.d_lower_right.numel()).view(lower.shape),
                torch.index_select(
                    self.d_lower_right, 0, index.clamp(max=self.d_lower_right.numel() - 1)
                ).view(lower.shape),
                lower,
                # If the pre-activation bounds are too loose, just use IBP.
                # torch.ones_like(index).to(lower) * (-100.)
            )
        else:
            # Lookup the lower bound slope from the pre-computed table.
            d_lower_right = torch.index_select(
                self.d_lower_right, 0, index).view(lower.shape)

        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            ((lower + np.sqrt(2)) / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        if index.max() >= self.d_lower_left.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Lookup the lower bound slope from the pre-computed table.
            d_lower_left = torch.where(
                (index < self.d_lower_left.numel()).view(upper.shape),
                torch.index_select(
                    self.d_lower_left, 0, index.clamp(max=self.d_lower_left.numel() - 1)
                ).view(lower.shape),
                upper,
            ).view(lower.shape)
        else:
            # Lookup the lower bound slope from the pre-computed table.
            d_lower_left = torch.index_select(
                self.d_lower_left, 0, index).view(lower.shape)

        index = torch.max(
            torch.zeros(lower.numel(), dtype=torch.long, device=lower.device),
            ((lower - np.sqrt(2)) / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        if index.max() >= self.d_upper_right.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Lookup the lower bound slope from the pre-computed table.
            d_upper_right = torch.where(
                (index < self.d_upper_right.numel()).view(upper.shape),
                torch.index_select(
                    self.d_upper_right, 0, index.clamp(max=self.d_upper_right.numel() - 1)
                ).view(upper.shape),
                upper,
            )
        else:
            d_upper_right = torch.index_select(
                self.d_upper_right, 0, index).view(upper.shape)

        index = torch.max(
            torch.zeros(upper.numel(), dtype=torch.long, device=lower.device),
            ((upper + np.sqrt(2)) / -self.step_pre).to(torch.long).reshape(-1)
        ) + 1
        if index.max() >= self.d_upper_left.numel():
            warnings.warn(f'Pre-activation bounds are too loose for {self}')
            # Lookup the lower bound slope from the pre-computed table.
            d_upper_left = torch.where(
                (index < self.d_upper_left.numel()).view(upper.shape),
                torch.index_select(
                    self.d_upper_left, 0, index.clamp(max=self.d_upper_left.numel() - 1)
                ).view(upper.shape),
                upper,
            )
        else:
            d_upper_left = torch.index_select(
                self.d_upper_left, 0, index).view(upper.shape)

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
            # Note that the upper bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=torch.logical_or(torch.logical_or(self.mask_left_pos, self.mask_right_neg),
                    self.mask_2), type='lower', k=k, x0=m, y0=y_m)
            # Upper bound is the middle point slope for the case input lower bound >= 0.
            # Note that the lower bound in this case is the direct line between (lower, func(lower)) and (upper, func(upper)).
            self.add_linear_relaxation(mask=torch.logical_or(self.mask_right_pos,
                    self.mask_left_neg), type='upper', k=k, x0=m, y0=y_m)

            # Now handle the case where input lower bound <=0 and upper bound >= 0.
            # A tangent line starting at d_lower is guaranteed to be a lower bound given the input upper bound.
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