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
"""BoundTile"""
from torch.nn import Module
from .base import *

class BoundTile(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
    
    def forward(self, x, repeats):
        return x.repeat(repeats.tolist())

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        assert not self.is_input_perturbed(1)
        repeats = x[1].value

        def _bound_oneside(A):
            if A is None:
                return None
            # block_shape: (specs, d1/r1, r1, d2/r2, r2, ..., dn/rn, rn)
            # Reshaping A to block_shape and sum along the "r" dimensions
            # is equivalent to summing up all block fragments of A.
            block_shape = [A.shape[0]]
            axes_to_sum = []
            for i in range(len(repeats)):
                block_shape.append(A.size(i + 1) // repeats[i].item())
                block_shape.append(repeats[i].item())
                axes_to_sum.append(2 * i + 2)
            reshaped_A = A.reshape(*block_shape)
            next_A = reshaped_A.sum(dim=axes_to_sum)
            return next_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, *x):
        assert (x[1].lb == x[1].ub).all(), "repeats should be constant."
        repeats = x[1].lb.tolist()
        assert repeats[0] == 1, "shouldn't repeat on the batch dimension."
        # lb and ub have the same shape as x, so we repeat then with "repeats"
        lb = x[0].lb.repeat(repeats)
        ub = x[0].ub.repeat(repeats)
        # lw and uw have shape (batch_size, input_dim, *shape_of_the_current_layer)
        # so we need to repeat them with "repeats" as well, but we need to
        # insert 1 at the second position to keep the input dimension unchanged.
        repeats.insert(1, 1)
        lw = x[0].lw.repeat(repeats)
        uw = x[0].uw.repeat(repeats)
        return LinearBound(lw, lb, uw, ub)
