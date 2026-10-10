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
""" Leaf nodes (indepedent nodes in the auto_LiRPA paper).

Including input, parameter, buffer, etc."""

from itertools import chain
from .base import *
from ..bounded_tensor import BoundedParameter, BoundedTensor


class BoundInput(Bound):
    def __init__(self, ori_name, value, perturbation=None,
                 model_input_index=None, options=None, attr=None):
        super().__init__(options=options, attr=attr)
        self.ori_name = ori_name
        self.value = value
        self.perturbation = perturbation
        self.input_dependent = True
        self.batch_dim = 0
        # Position in the flattened original model inputs used for binding.
        # Model inputs hold an integer; parameters and buffers hold None.
        self.model_input_index = model_input_index
        self.blocks_jacobian = True

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

    def bind_input(self, value):
        """Bind a runtime model input and invalidate its concrete cache."""
        self.value = value
        self.forward_value = None
        if isinstance(value, (BoundedTensor, BoundedParameter)):
            self.perturbation = value.ptb
        else:
            self.perturbation = None

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
        if options is None:
            options = {}
        self.auto_requires_grad = options.get("param", {}).get("auto_requires_grad", True)
        self.input_dependent = False
        self.batch_dim = -1

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
        param = self.param
        if self.auto_requires_grad:
            param = param.requires_grad_(self.training)
        return param

class BoundBuffers(BoundInput):
    def __init__(self, ori_name, value, perturbation=None, options=None, attr=None):
        super().__init__(ori_name, None, perturbation, attr=attr)
        self.register_buffer('buffer', value.clone().detach())
        self.input_dependent = False
        self.batch_dim = -1
        # has_batchdim marks a batch-shaped buffer layout.
        if 'buffers' in options and 'has_batchdim' in options['buffers']:
            warnings.warn('The "has_batchdim" option for BoundBuffers is deprecated.'
                          ' It may be removed from the next release.')
        if options.get('buffers', {}).get('has_batchdim', False):
            self.batch_dim = 0

    def forward(self):
        return self.buffer
