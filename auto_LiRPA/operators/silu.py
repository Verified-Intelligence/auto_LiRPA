#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
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
"""SiLU support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_base import BoundActivation


_SILU_X_MIN = -1.2784645427610737
_SILU_INFLECTION = 2.3993572805154675


def silu(x):
    return x * torch.sigmoid(x)


def dsilu(x):
    """First derivative of SiLU."""
    sig = torch.sigmoid(x)
    return sig * (1 + x * (1 - sig))


def d2silu(x):
    """Second derivative of SiLU."""
    sig = torch.sigmoid(x)
    return sig * (1 - sig) * (2 + x * (1 - 2 * sig))


class BoundSiLU(BoundActivation):
    """
    SiLU as an atomic unary activation.

    This atomic version focuses on two things:
    1. exact interval propagation using the internal minimum point;
    2. tighter single-line relaxations on pure convex/concave intervals, with
       a conservative fallback on mixed-curvature intervals.
    """
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True
        self.ibp_intermediate = True
        self.activation_name = 'SiLU'
        self.x_min = _SILU_X_MIN
        self.inflection_left = -_SILU_INFLECTION
        self.inflection_right = _SILU_INFLECTION
        self.inflections = [self.inflection_left, self.inflection_right]
        self.extremes = [self.x_min]
        self.split_min_gap = 1e-2
        self.split_range = (self.inflection_left, self.inflection_right)

    def forward(self, x):
        return F.silu(x)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        y_l = self.forward(h_L)
        y_u = self.forward(h_U)

        lower = torch.minimum(y_l, y_u)
        upper = torch.maximum(y_l, y_u)

        x_min = torch.full_like(h_L, self.x_min)
        y_min = self.forward(x_min)
        cross_min = torch.logical_and(h_L < self.x_min, h_U > self.x_min)
        lower = torch.where(cross_min, torch.minimum(lower, y_min), lower)
        return lower, upper

    def get_split_mask(self, lower, upper, input_index):
        del input_index
        return torch.logical_and(
            upper - lower >= self.split_min_gap,
            torch.logical_and(
                upper >= self.split_range[0],
                lower <= self.split_range[1],
            ),
        )

    def _segment_bounds(self, lower, upper):
        bounds = []
        left_l = lower
        left_u = min(upper, self.inflection_left)
        if left_l <= left_u:
            bounds.append((left_l, left_u, False))

        mid_l = max(lower, self.inflection_left)
        mid_u = min(upper, self.inflection_right)
        if mid_l <= mid_u:
            bounds.append((mid_l, mid_u, True))

        right_l = max(lower, self.inflection_right)
        right_u = upper
        if right_l <= right_u:
            bounds.append((right_l, right_u, False))
        return bounds

    def _is_convex_interval(self, lower, upper):
        return lower >= self.inflection_left and upper <= self.inflection_right

    def _is_concave_interval(self, lower, upper):
        return upper <= self.inflection_left or lower >= self.inflection_right

    def _find_derivative_root(self, left, right, target, increasing, steps=60):
        if right - left < 1e-10:
            return left
        lo, hi = left, right
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            value = dsilu(torch.tensor(mid, device=self.device)).item()
            if increasing:
                if value < target:
                    lo = mid
                else:
                    hi = mid
            else:
                if value > target:
                    lo = mid
                else:
                    hi = mid
        return 0.5 * (lo + hi)

    def _parallel_tangent_intercept(self, lower, upper, slope, increasing):
        tangent_point = self._find_derivative_root(
            lower, upper, slope, increasing=increasing)
        x_tensor = torch.tensor(tangent_point, device=self.device)
        return (silu(x_tensor) - slope * x_tensor).item()

    def _derivative_range(self, lower, upper):
        candidates = [lower, upper]
        if lower <= self.inflection_left <= upper:
            candidates.append(self.inflection_left)
        if lower <= self.inflection_right <= upper:
            candidates.append(self.inflection_right)
        values = []
        for x in candidates:
            values.append(dsilu(torch.tensor(x, device=self.device)).item())
        return min(values), max(values)

    def _phi_extrema(self, lower, upper, slope):
        candidates = [lower, upper]
        for seg_l, seg_u, increasing in self._segment_bounds(lower, upper):
            dl = dsilu(torch.tensor(seg_l, device=self.device)).item()
            du = dsilu(torch.tensor(seg_u, device=self.device)).item()
            low_d, high_d = (dl, du) if increasing else (du, dl)
            if low_d - 1e-10 <= slope <= high_d + 1e-10:
                root = self._find_derivative_root(seg_l, seg_u, slope, increasing)
                candidates.append(root)

        phi_vals = []
        for x in candidates:
            x_tensor = torch.tensor(x, device=self.device)
            phi_vals.append((silu(x_tensor) - slope * x_tensor).item())
        return min(phi_vals), max(phi_vals)

    def _optimize_generic_slope(self, lower, upper, initial_slope):
        deriv_l, deriv_u = self._derivative_range(lower, upper)
        lo = min(deriv_l, deriv_u)
        hi = max(deriv_l, deriv_u)
        if hi - lo < 1e-10:
            min_phi, max_phi = self._phi_extrema(lower, upper, initial_slope)
            return initial_slope, min_phi, max_phi

        best_k = initial_slope
        best_min_phi, best_max_phi = self._phi_extrema(lower, upper, initial_slope)
        best_gap = best_max_phi - best_min_phi

        search_lo, search_hi = lo, hi
        for num_steps in (33, 33):
            step = (search_hi - search_lo) / (num_steps - 1)
            for idx in range(num_steps):
                k = search_lo + idx * step
                min_phi, max_phi = self._phi_extrema(lower, upper, k)
                gap = max_phi - min_phi
                if gap < best_gap:
                    best_gap = gap
                    best_k = k
                    best_min_phi = min_phi
                    best_max_phi = max_phi
            radius = max(step * 2, 1e-4)
            search_lo = max(lo, best_k - radius)
            search_hi = min(hi, best_k + radius)

        return best_k, best_min_phi, best_max_phi

    def bound_relax(self, x, init=False):
        if init:
            self.init_linear_relaxation(x)

        lower = x.lower.reshape(-1)
        upper = x.upper.reshape(-1)
        lw = torch.empty_like(lower)
        lb = torch.empty_like(lower)
        uw = torch.empty_like(lower)
        ub = torch.empty_like(lower)

        for i in range(lower.numel()):
            l = lower[i].item()
            u = upper[i].item()
            if abs(u - l) < 1e-10:
                y = silu(torch.tensor(l, device=self.device)).item()
                lw[i] = uw[i] = 0.0
                lb[i] = ub[i] = y
                continue

            y_l = silu(torch.tensor(l, device=self.device)).item()
            y_u = silu(torch.tensor(u, device=self.device)).item()
            slope = (y_u - y_l) / (u - l)

            if self._is_convex_interval(l, u):
                secant_bias = y_l - slope * l
                tangent_bias = self._parallel_tangent_intercept(
                    l, u, slope, increasing=True)
                lw[i] = slope
                lb[i] = tangent_bias
                uw[i] = slope
                ub[i] = secant_bias
            elif self._is_concave_interval(l, u):
                secant_bias = y_l - slope * l
                tangent_bias = self._parallel_tangent_intercept(
                    l, u, slope, increasing=False)
                lw[i] = slope
                lb[i] = secant_bias
                uw[i] = slope
                ub[i] = tangent_bias
            else:
                slope, min_phi, max_phi = self._optimize_generic_slope(l, u, slope)
                lw[i] = slope
                uw[i] = slope
                lb[i] = min_phi
                ub[i] = max_phi

        self.lw = lw.view_as(x.lower)
        self.uw = uw.view_as(x.upper)
        self.lb = lb.view_as(x.lower)
        self.ub = ub.view_as(x.upper)


class SiLUOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op('custom::SiLU', x).setType(x.type())

    @staticmethod
    def forward(ctx, x):
        return F.silu(x)


class SiLU(nn.Module):
    def forward(self, x):
        return SiLUOp.apply(x)
