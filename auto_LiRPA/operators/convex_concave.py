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
"""Nonlinear functions that are either convex or convave within the entire domain."""
import torch
from .base import *
from .activation_base import BoundActivation, BoundOptimizableActivation


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
