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
""" Leaf nodes (indepedent nodes in the auto_LiRPA paper).

Including input, parameter, buffer, etc."""

from itertools import chain
from .base import *


class BoundInput(Bound):
    def __init__(self, ori_name, value, perturbation=None, input_index=None, options=None, attr=None):
        super().__init__(options=options, attr=attr)
        self.ori_name = ori_name
        self.value = value
        self.perturbation = perturbation
        self.from_input = True
        self.input_index = input_index
        self.no_jacobian = True

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        # Update perturbed property based on the perturbation set.
        if key == "perturbation":
            if self.perturbation is not None:
                self.perturbed = True
            else:
                self.perturbed = False

    def forward(self):
        return self.value

    def bound_forward(self, dim_in):
        assert 0

    def bound_backward(self, last_lA, last_uA, **kwargs):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

    def interval_propagate(self, *v):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

class BoundParams(BoundInput):
    def __init__(self, ori_name, value, perturbation=None, options=None, attr=None):
        super().__init__(ori_name, None, perturbation, attr=attr)
        self.register_parameter('param', value)
        self.from_input = False
        self.initializing = False

    def register_parameter(self, name, param):
        """Override register_parameter() hook to register only needed parameters."""
        if name == 'param':
            return super().register_parameter(name, param)
        else:
            # Just register it as a normal property of class.
            object.__setattr__(self, name, param)

    def init(self, initializing=False):
        self.initializing = initializing

    def forward(self):
        if self.initializing:
            return self.param_init.requires_grad_(self.training)
        else:
            return self.param.requires_grad_(self.training)

class BoundBuffers(BoundInput):
    def __init__(self, ori_name, value, perturbation=None, options=None, attr=None):
        super().__init__(ori_name, None, perturbation, attr=attr)
        self.register_buffer('buffer', value.clone().detach())
        # BoundBuffers are like constants and they are by default not from inputs.
        # The "has_batchdim" was a hack that will forcibly set BoundBuffer to be
        # from inputs, to workaround buffers with a batch size dimension. This is
        # not needed in most cases now.
        if 'buffers' in options and 'has_batchdim' in options['buffers']:
            warnings.warn('The "has_batchdim" option for BoundBuffers is deprecated.'
                          ' It may be removed from the next release.')
        self.from_input = options.get('buffers', {}).get('has_batchdim', False)

    def forward(self):
        return self.buffer
