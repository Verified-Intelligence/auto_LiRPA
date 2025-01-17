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

# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class BoundSoftmax(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.option = options.get('softmax', 'complex')
        if self.option == 'complex':
            self.complex = True
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
        lower = exp_L / (torch.sum(exp_U, dim=self.axis, keepdim=True) - exp_U + exp_L + epsilon)
        upper = exp_U / (torch.sum(exp_L, dim=self.axis, keepdim=True) - exp_L + exp_U + epsilon)
        return lower, upper
