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
import torch
import torch.nn as nn

class CrossEntropyWrapper(nn.Module):
    def __init__(self, model):
        super(CrossEntropyWrapper, self).__init__()
        self.model = model

    def forward(self, x, labels):
        y = self.model(x)
        logits = y - torch.gather(y, dim=-1, index=labels.unsqueeze(-1))
        return torch.exp(logits).sum(dim=-1, keepdim=True)

class CrossEntropyWrapperMultiInput(nn.Module):
    def __init__(self, model):
        super(CrossEntropyWrapperMultiInput, self).__init__()
        self.model = model

    def forward(self, labels, *x):
        y = self.model(*x)
        logits = y - torch.gather(y, dim=-1, index=labels.unsqueeze(-1))
        return torch.exp(logits).sum(dim=-1, keepdim=True)