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
import torch
from torch.nn import Module
from .base import Bound
from ..utils import prod


class JacobianOP(torch.autograd.Function):
    @staticmethod
    def symbolic(g, output, input):
        return g.op('grad::jacobian', output, input).setType(output.type())

    @staticmethod
    def forward(ctx, output, input):
        output_ = output.flatten(1)
        return torch.zeros(
            output.shape[0], output_.shape[-1], *input.shape[1:],
            device=output.device)


class BoundJacobianOP(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, output, input):
        return JacobianOP.apply(output, input)


class BoundJacobianInit(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.never_perturbed = True

    def forward(self, x):
        dim = prod(x.shape[1:])
        eye = torch.eye(dim, device=x.device, requires_grad=x.requires_grad)
        eye = eye.unsqueeze(0).expand(
            x.shape[0], -1, -1
        ).view(x.shape[0], dim, *x.shape[1:])
        return eye


class GradNorm(Module):
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def forward(self, grad):
        grad = grad.view(grad.size(0), -1)
        if self.norm == 1:
            # torch.norm is not supported in auto_LiRPA yet
            # use simpler operators for now
            return grad.abs().sum(dim=-1, keepdim=True)
        elif self.norm == 2:
            return (grad * grad).sum(dim=-1)
        else:
            raise NotImplementedError(self.norm)
