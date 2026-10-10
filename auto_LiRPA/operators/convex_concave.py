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
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""Nonlinear functions that are either convex or convave within the entire domain."""
import torch
from torch.nn import Module
from .base import *
from .activation_base import BoundActivation, BoundOptimizableActivation


class BoundLog(BoundActivation):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.range_l = 1e-6

    @staticmethod
    def _check_real_domain(lower):
        if (lower <= 0).any():
            raise OperatorDomainError(
                'log',
                'input bounds require a strictly positive lower bound',
            )

    def forward(self, x):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            return torch.logsumexp(self.inputs[0].inputs[0].inputs[0].forward_value, dim=-1)
        return torch.log(x)

    def bound_relax(self, x, init=False):
        self._check_real_domain(x.lower)
        self.require_finite_relaxation_bounds(x.lower, x.upper, operator='log')
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
            return Interval(lower, upper)
        self._check_real_domain(v[0].lower)
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        A, lbias, ubias = super().bound_backward(last_lA, last_uA, x)
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            assert A[0][0] is None
            exp_module = self.inputs[0].inputs[0]
            ubias = ubias + self.get_bias(A[0][1], exp_module.max_input.squeeze(-1))
        return A, lbias, ubias


class BoundSqrt(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_prior_constraint = True
        self.has_constraint = True
        self.range_l = 1e-6

    @staticmethod
    def _check_real_domain(lower):
        if (lower < 0).any():
            raise OperatorDomainError(
                'sqrt',
                'input bounds require a non-negative lower bound',
            )

    def forward(self, x):
        return torch.sqrt(x)

    def bound_relax(self, x, init=False, dim_opt=None):
        self._check_real_domain(x.lower)
        self.require_finite_relaxation_bounds(x.lower, x.upper, operator='sqrt')
        if init:
            self.init_linear_relaxation(x, dim_opt)

        if self.opt_stage in ['opt', 'reuse']:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], x.lower), x.upper)
            mid = self.alpha[self._start]
        else:
            mid = (x.lower + x.upper) / 2
        # The square-root derivative is infinite at zero, but a zero-width
        # interval at zero is represented exactly by the constant line y = 0.
        fixed_zero = (x.lower == 0) & (x.upper == 0)
        safe_mid = torch.where(fixed_zero, torch.ones_like(mid), mid)
        k = 0.5 / self.forward(safe_mid)
        k = torch.where(fixed_zero, torch.zeros_like(k), k)
        mid_value = self.forward(safe_mid)
        mid_value = torch.where(
            fixed_zero, torch.zeros_like(mid_value), mid_value)
        self.add_linear_relaxation(
            mask=None, type='upper', k=k, x0=mid, y0=mid_value)

        sqrt_l = self.forward(x.lower)
        sqrt_u = self.forward(x.upper)
        k = (sqrt_u - sqrt_l) / (x.upper - x.lower).clamp(min=1e-8)
        self.add_linear_relaxation(mask=None, type='lower', k=k, x0=x.lower)

    def interval_propagate(self, *v):
        self._check_real_domain(v[0].lower)
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        if self.use_prior_constraint and self.check_constraint_available(x):
            if hasattr(x, 'cstr_interval'):
                del x.cstr_interval
                del x.cstr_lower
                del x.cstr_upper

            constraint = self._ibp_constraint(
                x, delete_bounds_after_use=True)
            x_l, x_u = constraint.lower, constraint.upper
            x_u = torch.max(x_u, x_l + 1e-8)
        return super().bound_backward(last_lA, last_uA, x, **kwargs)

    def clamp_interim_bounds(self):
        self.cstr_lower = self.lower.clamp(min=0)
        self.cstr_upper = self.upper.clamp(min=0)
        self.cstr_interval = Interval(
            self.cstr_lower, self.cstr_upper)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(
            2, size_spec, *l.shape, dtype=l.dtype, device=l.device)
        alpha.data[:2] = (l + u) / 2
        return alpha


class BoundReciprocal(BoundOptimizableActivation):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True
        self.range_l = 1e-6

    def forward(self, x):
        return torch.reciprocal(x)

    def interval_propagate(self, *v):
        h_L = v[0].lower
        h_U = v[0].upper
        lower = torch.reciprocal(h_U)
        upper = torch.reciprocal(h_L)

        # reciprocal(h_U), reciprocal(h_L) already handle intervals strictly
        # on either side of zero and [0, h_U], since reciprocal(+0) is +inf.
        # Correct the remaining singular cases: crossing zero needs the full
        # real hull, approaching zero from below needs a -inf lower bound, and
        # the zero-only interval is conservatively treated as fully unbounded.
        crosses_zero = (h_L < 0) & (h_U > 0)
        negative_to_zero = (h_L < 0) & (h_U == 0)
        zero_only = (h_L == 0) & (h_U == 0)

        lower = torch.where(
            crosses_zero | negative_to_zero | zero_only,
            torch.full_like(lower, -torch.inf),
            lower)
        upper = torch.where(
            crosses_zero | zero_only,
            torch.full_like(upper, torch.inf),
            upper)
        return Interval(lower, upper)

    def bound_relax(self, x, init=False, dim_opt=None):
        if ((x.lower <= 0) & (x.upper >= 0)).any():
            raise OperatorDomainError(
                'reciprocal',
                'linear bound propagation input interval contains zero',
            )
        positive = x.lower > 0
        negative = x.upper < 0

        # Use the finite endpoint to choose a tangent for one-sided unbounded
        # intervals. Their asymptotic relaxations are assigned below.
        unbounded_lower = torch.isneginf(x.lower)
        unbounded_upper = torch.isposinf(x.upper)
        finite_lower = torch.where(
            unbounded_lower, x.upper, x.lower)
        finite_upper = torch.where(
            unbounded_upper, x.lower, x.upper)
        self.require_finite_relaxation_bounds(
            finite_lower, finite_upper, operator='reciprocal')
        if init:
            self.init_linear_relaxation(x, dim_opt)

        # Reciprocal is convex on (0, +inf) and concave on (-inf, 0), so the
        # secant and tangent exchange their upper/lower roles across zero.
        secant_k = -1. / (x.lower * x.upper)
        secant_x0 = torch.where(positive, x.lower, x.upper)
        self.add_linear_relaxation(
            mask=positive, type='upper', k=secant_k, x0=secant_x0)
        self.add_linear_relaxation(
            mask=negative, type='lower', k=secant_k, x0=secant_x0)

        if self.opt_stage in ['opt', 'reuse']:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], finite_lower), finite_upper)
            mid = torch.where(
                positive,
                self.alpha[self._start].clamp(min=0.01),
                self.alpha[self._start].clamp(max=-0.01),
            )
        else:
            mid = (finite_lower + finite_upper) / 2

        tangent_k = -1. / (mid**2)
        self.add_linear_relaxation(
            mask=positive, type='lower', k=tangent_k, x0=mid)
        self.add_linear_relaxation(
            mask=negative, type='upper', k=tangent_k, x0=mid)

        self.lw[..., unbounded_upper] = 0
        self.lb[..., unbounded_upper] = 0
        self.uw[..., unbounded_lower] = 0
        self.ub[..., unbounded_lower] = 0

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(
            2, size_spec, *l.shape, dtype=l.dtype, device=l.device)
        finite_lower = torch.where(torch.isneginf(l), u, l)
        finite_upper = torch.where(torch.isposinf(u), l, u)
        alpha.data[:2] = (finite_lower + finite_upper) / 2
        return alpha

    def build_gradient_node(self, grad_upstream):
        return [(ReciprocalGrad(), (grad_upstream, self.inputs[0].forward_value), [self.inputs[0]])]


class ReciprocalGrad(Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, x):
        # partial derivative of 1/x is -1/x^2
        return -g / torch.square(x).unsqueeze(1)


class BoundExp(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if options is None:
            options = {}
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
        h_L, h_U = v[0].lower, v[0].upper
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(h_U, dim=-1, keepdim=True)[0]
            h_L, h_U = h_L - self.max_input, h_U - self.max_input
        else:
            self.max_input = 0
        return Interval(torch.exp(h_L), torch.exp(h_U))

    def bound_forward(self, dim_in, x):
        self.require_finite_relaxation_bounds(x.lower, x.upper, operator='exp')
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
                last_uA) >= 0 and x.batch_dim != -1:
            self.require_finite_relaxation_bounds(x.lower, x.upper, operator='exp')
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
            As, lbias, ubias = super().bound_backward(last_lA, last_uA, x, **kwargs)
            lA, uA = As[0]
            lA, lbias = self._check_nan(lA, lbias, last_lA, 0)
            uA, ubias = self._check_nan(uA, ubias, last_uA, torch.inf)
            return [(lA, uA)], lbias, ubias

    def _check_nan(self, A, bias, last_A, const_bound):
        """Check for NaN caused by 0 in A and inf in lw/lb/uw/ub.
        It can happen if the pre-activation bounds are very loose for exp.
        """
        if A is None:
            return A, bias
        if bias.isnan().any():
            # These assertions ensure that 0 is in A and inf is in lw/lb/uw/ub
            assert not last_A.isnan().any()
            assert not last_A.isinf().any()
            assert not self.lw.isnan().any()
            assert not self.uw.isnan().any()
            assert not self.lb.isnan().any()
            assert not self.ub.isnan().any()
            A_ = A.view(-1, *A.shape[2:]).clone()
            bias_ = bias.view(-1).clone()
            mask = bias_.isnan()
            A_[mask] = 0
            assert (last_A >= 0).all()
            bias_[mask] = const_bound if (last_A != 0).any() else 0.
            A = A_.view(A.shape)
            bias = bias_.view(bias.shape)
        return A, bias

    def bound_relax(self, x, init=False, dim_opt=None):
        self.require_finite_relaxation_bounds(x.lower, x.upper, operator='exp')
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
        alpha = torch.empty(
            2, size_spec, *l.shape, dtype=l.dtype, device=l.device)
        alpha.data[:2] = (l + u) / 2
        return alpha

    def build_gradient_node(self, grad_upstream):
        if self.loss_fusion:
            raise NotImplementedError('Gradient computation for exp with loss fusion is not supported.')
        return [(ExpGrad(), (grad_upstream, self.inputs[0].forward_value), [self.inputs[0]])]


class ExpGrad(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, g, preact):
        # exp'(x) = exp(x)
        return g * torch.exp(preact).unsqueeze(1)
