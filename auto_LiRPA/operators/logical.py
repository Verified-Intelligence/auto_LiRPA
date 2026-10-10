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
""" Logical operators"""
from .base import *


class BoundWhere(Bound):
    def forward(self, condition, x, y):
        return torch.where(condition.to(torch.bool), x, y)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(0)
        condition = v[0].lower
        return Interval(
            torch.where(condition, v[1].lower, v[2].lower),
            torch.where(condition, v[1].upper, v[2].upper))

    def bound_backward(self, last_lA, last_uA, condition, x, y, **kwargs):
        assert torch.allclose(condition.lower, condition.upper)
        assert self.batch_dim != -1
        mask = condition.lower

        def _bound_oneside(last_A):
            if last_A is None:
                return None, None
            assert last_A.ndim > 1
            mask_for_A = mask.to(dtype=last_A.dtype)
            A_x = self.broadcast_backward(mask_for_A.unsqueeze(0) * last_A, x)
            A_y = self.broadcast_backward(
                (1 - mask_for_A).unsqueeze(0) * last_A, y)
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
