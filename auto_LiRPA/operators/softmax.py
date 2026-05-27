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
""" Softmax """
from .base import *


class BoundSoftmaxImpl(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis
        assert self.axis == int(self.axis)

    def forward(self, x):
        max_x = torch.max(x, dim=self.axis).values
        x = torch.exp(x - max_x.unsqueeze(self.axis))
        s = torch.sum(x, dim=self.axis, keepdim=True)
        return x / s


class BoundSoftmax(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.option = options.get('softmax', 'lse')
        if self.option == 'complex':
            self.complex = True
        elif self.option == 'lse':
            self.requires_input_bounds = [0]
        else:
            self.max_input = 30

    def forward(self, x):
        assert self.axis == int(self.axis)
        if self.option == 'complex':
            self.input = (x,)
            self.model = BoundSoftmaxImpl(self.axis)
            self.model.device = self.device
            return self.model(x)
        else:
            return F.softmax(x, dim=self.axis)

    def interval_propagate(self, *v):
        assert self.option != 'complex'
        assert self.perturbed
        h_L, h_U = v[0]
        shift = h_U.max(dim=self.axis, keepdim=True).values
        exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
        lower = exp_L / (torch.sum(exp_U, dim=self.axis, keepdim=True)
                         - exp_U + exp_L + epsilon)
        upper = exp_U / (torch.sum(exp_L, dim=self.axis, keepdim=True)
                         - exp_L + exp_U + epsilon)
        return lower, upper

    def _softmax_lse_lower(self, lb_p, ub_p, x0_p):
        """Lower bound Jacobian and bias via LSE (Eqs. 39-40).

        L_k(x) = e^{x_k} / SE(x; l, u) is convex, so its tangent at x0
        gives a valid affine lower bound.
        """
        # Shift by max(u) for stability.
        max_u = ub_p.max(dim=-1, keepdim=True)[0]
        u_s = ub_p - max_u
        l_s = lb_p - max_u
        x0_s = x0_p - max_u

        diff = u_s - l_s
        # Chordal slope and intercept of exp on [l_i, u_i].
        kappa = torch.where(
            diff > 1e-5,
            (torch.exp(u_s) - torch.exp(l_s)) / diff,
            torch.exp(u_s))
        beta = torch.where(
            diff > 1e-5,
            (u_s * torch.exp(l_s) - l_s * torch.exp(u_s)) / diff,
            torch.exp(u_s) * (1 - u_s))

        se_x0 = (kappa * x0_s + beta).sum(dim=-1, keepdim=True)
        L_x0 = torch.exp(x0_s) / se_x0
        V_k = L_x0 / se_x0

        # OPTIMIZED: Return implicit components (D, u, v) instead of full Jacobian
        # J_L = diag(L_x0) - outer(V_k, kappa)
        J_L_comp = (L_x0, V_k, kappa)

        # OPTIMIZED: Calculate b_L without full matrix multiplication
        # b_L = L_x0 - J_L @ x0_p
        dot_kappa_x0 = (kappa * x0_p).sum(dim=-1, keepdim=True)
        b_L = L_x0 - (L_x0 * x0_p) + (V_k * dot_kappa_x0)

        return J_L_comp, b_L

    def _softmax_lse_upper(self, lb_p, ub_p, x0_p, sm_x0, max_x0, sum_exp_x0, eps):
        """Upper bound Jacobian and bias via LSE (Eqs. 46-47).

        U_k(x) = a_k - c_k (log SE(x) - x_k) is concave, so its tangent
        at x0 gives a valid affine upper bound.
        """
        # p_under_k and p_bar_k (constant probability bounds).
        max_u = ub_p.max(dim=-1, keepdim=True)[0]
        u_s = ub_p - max_u
        l_s = lb_p - max_u
        sum_exp_u = torch.exp(u_s).sum(dim=-1, keepdim=True)
        p_under = torch.exp(l_s) / (
            torch.exp(l_s) - torch.exp(u_s) + sum_exp_u + eps)

        max_l = lb_p.max(dim=-1, keepdim=True)[0]
        u_sl = ub_p - max_l
        l_sl = lb_p - max_l
        sum_exp_l = torch.exp(l_sl).sum(dim=-1, keepdim=True)
        p_bar = torch.exp(u_sl) / (
            torch.exp(u_sl) - torch.exp(l_sl) + sum_exp_l + eps)

        p_diff = p_bar - p_under
        log_p_bar = torch.log(p_bar + eps)
        log_p_under = torch.log(p_under + eps)
        log_p_diff = log_p_bar - log_p_under

        c_k = torch.where(p_diff > 1e-5, p_diff / log_p_diff, p_bar)
        a_k = torch.where(
            p_diff > 1e-5,
            (p_under * log_p_bar - p_bar * log_p_under) / log_p_diff,
            p_bar * (1 - log_p_bar))

        # OPTIMIZED: Return implicit components (D, u, v) instead of full Jacobian
        # J_U = diag(c_k) - outer(c_k, sm_x0)
        J_U_comp = (c_k, c_k, sm_x0)

        lse_x0 = max_x0 + torch.log(sum_exp_x0 + eps)
        U_x0 = a_k - c_k * (lse_x0 - x0_p)

        # OPTIMIZED: Calculate b_U without full matrix multiplication
        # b_U = U_x0 - J_U @ x0_p
        dot_sm_x0 = (sm_x0 * x0_p).sum(dim=-1, keepdim=True)
        b_U = U_x0 - (c_k * x0_p) + (c_k * dot_sm_x0)

        return J_U_comp, b_U

    @staticmethod
    def _move_dim_to_last(t, dim, ndim):
        if t is None:
            return None
        return t.transpose(dim, -1) if dim != ndim - 1 else t

    @staticmethod
    def _restore_dim(t, A_dim, dim, ndim):
        if t is None:
            return None
        return t.transpose(A_dim, -1) if dim != ndim - 1 else t

    @staticmethod
    def _apply_jacobian_components(A, D, u, v):
        """Computes A @ (diag(D) - outer(u, v)) implicitly in O(N) memory."""
        # A has shape (spec, batch, ..., K), while D/u/v have shape
        # (batch, ..., K). Add the missing leading spec dimension explicitly
        # so broadcasting works for arbitrary batch sizes.
        term1 = A * D.unsqueeze(0)
        Au = (A * u.unsqueeze(0)).sum(dim=-1, keepdim=True)
        term2 = Au * v.unsqueeze(0)
        return term1 - term2

    def _propagate(self, last_A, J_pos_comp, J_neg_comp, b_pos, b_neg, dim, A_dim, ndim):
        """Sign-split A and propagate through local implicit Jacobians."""
        if last_A is None:
            return None, 0

        A_p = self._move_dim_to_last(last_A, A_dim, last_A.ndim)
        A_pos = A_p.clamp(min=0)
        A_neg = A_p.clamp(max=0)

        # Apply implicit matrix multiplication
        next_A_pos = self._apply_jacobian_components(A_pos, *J_pos_comp)
        next_A_neg = self._apply_jacobian_components(A_neg, *J_neg_comp)
        
        next_A = next_A_pos + next_A_neg
        next_A = self._restore_dim(next_A, A_dim, dim, ndim)

        # Bias propagation uses simple dot products. b_pos/b_neg do not have
        # the leading spec dimension present in A_pos/A_neg, so add it
        # explicitly to ensure correct broadcasting for batch > 1.
        next_bias = ((A_pos * b_pos.unsqueeze(0)).sum(dim=-1)
                     + (A_neg * b_neg.unsqueeze(0)).sum(dim=-1))
        
        if next_bias.ndim > 2:
            next_bias = next_bias.flatten(2).sum(dim=-1)
            
        return next_A, next_bias

    def bound_backward(self, last_lA, last_uA, *x, start_node=None, **kwargs):
        """CROWN backward via LSE decomposition (Wei et al., AISTATS 2023)."""
        lb = x[0].lower
        ub = x[0].upper

        ndim = lb.ndim
        dim = self.axis if self.axis >= 0 else ndim + self.axis
        A_dim = dim if dim < 0 else dim + 1

        lb_p = self._move_dim_to_last(lb, dim, ndim)
        ub_p = self._move_dim_to_last(ub, dim, ndim)

        eps = 1e-8

        # Tangent point; TODO: make x0 optimizable (alpha-CROWN style).
        x0_p = (lb_p + ub_p) / 2.0

        # softmax(x0) at the tangent point.
        max_x0 = x0_p.max(dim=-1, keepdim=True)[0]
        x0_shifted = x0_p - max_x0
        exp_x0 = torch.exp(x0_shifted)
        sum_exp_x0 = exp_x0.sum(dim=-1, keepdim=True)
        sm_x0 = exp_x0 / sum_exp_x0

        J_L_comp, b_L = self._softmax_lse_lower(lb_p, ub_p, x0_p)
        J_U_comp, b_U = self._softmax_lse_upper(
            lb_p, ub_p, x0_p, sm_x0, max_x0, sum_exp_x0, eps)

        # CROWN: lower uses (J_L, J_U), upper uses (J_U, J_L).
        lA, lbias = self._propagate(
            last_lA, J_L_comp, J_U_comp, b_L, b_U, dim, A_dim, ndim)
        uA, ubias = self._propagate(
            last_uA, J_U_comp, J_L_comp, b_U, b_L, dim, A_dim, ndim)

        return [(lA, uA)], lbias, ubias
