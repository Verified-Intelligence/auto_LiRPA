#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
""" Logical operators"""
from .base import *


class BoundWhere(Bound):
    def forward(self, condition, x, y):
        return torch.where(condition.to(torch.bool), x, y)

    def interval_propagate(self, *v):
        (cond_l, cond_u), (xl, xu), (yl, yu) = v[0], v[1], v[2]
        if not self.is_input_perturbed(0):
            # Fixed condition: select the corresponding branch bounds elementwise.
            c = cond_l.to(torch.bool)
            return torch.where(c, xl, yl), torch.where(c, xu, yu)
        # Perturbed condition: ``where`` treats any non-zero as true (numpy/minijax
        # semantics). Sound interval rule: where the condition interval excludes 0
        # the branch is determined; where it contains 0 either branch is possible,
        # so take the elementwise union of the two branch intervals.
        cl, cu = cond_l, cond_u
        true_mask = (cl > 0) | (cu < 0)          # interval excludes 0 -> definitely x
        false_mask = (cl == 0) & (cu == 0)       # exactly zero -> definitely y
        out_l = torch.where(true_mask, xl, torch.where(false_mask, yl, torch.minimum(xl, yl)))
        out_u = torch.where(true_mask, xu, torch.where(false_mask, yu, torch.maximum(xu, yu)))
        return out_l, out_u

    def bound_backward(self, last_lA, last_uA, condition, x, y, **kwargs):
        # NOTE: CROWN (linear relaxation) still only supports a fixed condition.
        # interval_propagate (IBP) above handles a perturbed condition via the
        # branch union; milestone2 only uses where through IBP-mode bounds, so
        # this path is not hit. Bounding a perturbed-condition where with CROWN
        # would need a linear relaxation of the selection and is not implemented.
        assert torch.allclose(condition.lower.float(), condition.upper.float())
        assert self.from_input
        mask = condition.lower.float()

        def _bound_oneside(last_A):
            if last_A is None:
                return None, None
            assert last_A.ndim > 1
            A_x = self.broadcast_backward(mask.unsqueeze(0) * last_A, x)
            A_y = self.broadcast_backward((1 - mask).unsqueeze(0) * last_A, y)
            return A_x, A_y

        lA_x, lA_y = _bound_oneside(last_lA)
        uA_x, uA_y = _bound_oneside(last_uA)

        return [(None, None), (lA_x, uA_x), (lA_y, uA_y)], 0, 0

class BoundNot(Bound):
    def forward(self, x):
        return x.logical_not()


class BoundEqual(Bound):
    def forward(self, x, y):
        return x == y
