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
""" Reduce operators"""
from .base import *


class BoundReduce(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr.get('axes', None)
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True
        self.use_default_ibp = True

    def _parse_input_and_axis(self, *x):
        if len(x) > 1:
            assert not self.is_input_perturbed(1)
            self.axis = tuple(item.item() for item in tuple(x[1]))
        self.axis = self.make_axis_non_negative(self.axis)
        return x[0]

    def _return_bound_backward(self, lA, uA):
        return [(lA, uA)] + [(None, None)] * (len(self.inputs) - 1), 0, 0


class BoundReduceMax(BoundReduce):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        """Assume that the indexes with the maximum values are not perturbed.
        This generally doesn't hold true, but can still be used for the input shift
        in Softmax of Transformers."""
        self.fixed_max_index = options.get('fixed_reducemax_index', False)

    def _parse_input_and_axis(self, *x):
        x = super()._parse_input_and_axis(*x)
        # for torch.max, `dim` must be an int
        if isinstance(self.axis, tuple):
            assert len(self.axis) == 1
            self.axis = self.axis[0]
        return x

    def forward(self, *x):
        x = self._parse_input_and_axis(*x)
        res = torch.max(x, dim=self.axis, keepdim=self.keepdim)
        self.indices = res.indices
        return res.values

    def bound_backward(self, last_lA, last_uA, *args, **kwargs):
        if self.fixed_max_index:
            def _bound_oneside(last_A):
                if last_A is None:
                    return None
                indices = self.indices.unsqueeze(0)
                if not self.keepdim:
                    assert (self.from_input)
                    last_A = last_A.unsqueeze(self.axis + 1)
                    indices = indices.unsqueeze(self.axis + 1)
                shape = list(last_A.shape)
                shape[self.axis + 1] *= self.input_shape[self.axis]
                A = torch.zeros(shape, device=last_A.device)
                indices = indices.expand(*last_A.shape)
                A.scatter_(dim=self.axis + 1, index=indices, src=last_A)
                return A

            return self._return_bound_backward(_bound_oneside(last_lA),
                                               _bound_oneside(last_uA))
        else:
            raise NotImplementedError(
                '`bound_backward` for BoundReduceMax with perturbed maximum'
                'indexes is not implemented.')


class BoundReduceMin(BoundReduceMax):
    def forward(self, *x):
        x = self._parse_input_and_axis(*x)
        res = torch.min(x, dim=self.axis, keepdim=self.keepdim)
        self.indices = res.indices
        return res.values


class BoundReduceMean(BoundReduce):
    def forward(self, *x):
        x = self._parse_input_and_axis(*x)
        return torch.mean(x, dim=self.axis, keepdim=self.keepdim)

    def bound_backward(self, last_lA, last_uA, *args, **kwargs):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                shape = list(last_A.shape)
                size_axis = self.input_shape[axis]
                shape[axis + 1] *= size_axis
                last_A = last_A.expand(*shape) / size_axis
            return last_A

        return self._return_bound_backward(_bound_oneside(last_lA),
                                           _bound_oneside(last_uA))

    def bound_forward(self, dim_in, x, *args):
        assert self.keepdim
        assert len(self.axis) == 1
        axis = self.make_axis_non_negative(self.axis[0])
        assert (axis > 0)
        size = self.input_shape[axis]
        lw = x.lw.sum(dim=axis + 1, keepdim=True) / size
        lb = x.lb.sum(dim=axis, keepdim=True) / size
        uw = x.uw.sum(dim=axis + 1, keepdim=True) / size
        ub = x.ub.sum(dim=axis, keepdim=True) / size
        return LinearBound(lw, lb, uw, ub)


class BoundReduceSum(BoundReduce):
    def forward(self, *x):
        x = self._parse_input_and_axis(*x)
        if self.axis is not None:
            return torch.sum(x, dim=self.axis, keepdim=self.keepdim)
        else:
            return torch.sum(x)

    def bound_backward(self, last_lA, last_uA, x, *args, **kwargs):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                shape = list(last_A.shape)
                shape[axis + 1] *= self.input_shape[axis]
                last_A = last_A.expand(*shape)
            return last_A

        return self._return_bound_backward(_bound_oneside(last_lA),
                                           _bound_oneside(last_uA))

    def bound_forward(self, dim_in, x, *args):
        assert len(self.axis) == 1
        axis = self.make_axis_non_negative(self.axis[0])
        assert axis > 0
        lw = x.lw.sum(dim=axis + 1, keepdim=self.keepdim)
        lb = x.lb.sum(dim=axis, keepdim=self.keepdim)
        uw = x.uw.sum(dim=axis + 1, keepdim=self.keepdim)
        ub = x.ub.sum(dim=axis, keepdim=self.keepdim)
        return LinearBound(lw, lb, uw, ub)
