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
""" Reduce operators"""
from .base import *
from torch.nn import Module


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
        
    def build_gradient_node(self, grad_upstream):
        if self.fixed_max_index:
            node_grad = ReduceMaxGrad(self.axis, self.keepdim, self.input_shape, self.indices)
            return [(node_grad, (grad_upstream,), [])]
        else:
            raise NotImplementedError(
                '`build_gradient_node` for BoundReduceMax with perturbed maximum'
                'indexes is not implemented.')


class ReduceMaxGrad(Module):
    def __init__(self, axis, keepdim, input_shape, indices):
        super().__init__()
        self.axis = axis
        self.keepdim = keepdim
        self.input_shape = input_shape
        self.indices = indices.unsqueeze(0)

    def forward(self, grad_last):
        # Only keep the gradient at the maximum index
        # The gradient at other indices is 0
        # If keepdim is False, add a singleton dimension at the specified axis
        if not self.keepdim:
            grad_last = grad_last.unsqueeze(self.axis + 1)
            indices = self.indices.unsqueeze(self.axis + 1)
        else:
            indices = self.indices
            assert grad_last.shape[self.axis + 1] == 1
        # Calculate the target dimension size at axis + 1
        new_dim = self.input_shape[self.axis]
        # Create the output tensor shape
        new_shape = list(grad_last.shape)
        new_shape[self.axis + 1] = new_dim

        ########################################################################
        # TODO: The following lines are equivalent to:
        #
        # grad = torch.zeros(new_shape, device=grad_last.device)
        # indices = indices.expand(*grad_last.shape)
        # grad.scatter_(dim=self.axis + 1, index=indices, src=grad_last)
        #
        # But auto_LiRPA does not support scatter_ yet.
        # So we use a workaround to avoid using scatter_.
        ########################################################################

        # Expand indices to match the target shape,
        # filling axis + 1 with new_dim
        indices_expanded = indices.expand(
            *grad_last.shape[:self.axis + 1],
            new_dim,
            *grad_last.shape[self.axis + 2:]
            ).to(grad_last.device)
        # Create a coordinate tensor for comparison along axis + 1
        coord_shape = [1] * grad_last.dim()
        coord_shape[self.axis + 1] = new_dim
        coord = torch.arange(new_dim, device=grad_last.device).view(*coord_shape)
        # Create a binary mask where 1 indicates the desired position for each gradient
        mask = (coord == indices_expanded).type_as(grad_last)
        # Expand grad_last to match the target shape for element-wise multiplication
        grad_last_expanded = grad_last.expand(
            *grad_last.shape[:self.axis + 1],
            new_dim,
            *grad_last.shape[self.axis + 2:])
        # Use the mask to retain values only at the correct positions
        grad = mask * grad_last_expanded
        return grad


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
            shape = list(last_A.shape)
            shape[2:] = self.input_shape[1:]
            # We perform expansion as in BoundReduceSum. 
            # and divide the product of the sizes of the reduced dimensions.
            last_A = last_A.expand(*shape) / np.prod(np.take(self.input_shape, self.axis))
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
            # last_A.shape = [num_spec, batch_size, ..., dim_size_1 (1), ...]
            shape = list(last_A.shape)
            # self.input_shape = [batch_size_original, ..., dim_size_1_before_reduction, ...]
            # we expand last_A with keeping its batch_size instead of that from self.input_shape.
            shape[2:] = self.input_shape[1:]
            # For reduced dims, their dim_size will be expanded from 1 to the original size.
            # For non-reduced dims, their dim_size will be unchanged.
            last_A = last_A.expand(*shape)
            return last_A

        return self._return_bound_backward(_bound_oneside(last_lA),
                                           _bound_oneside(last_uA))

    def bound_forward(self, dim_in, x, *args):
        # Handle possibly multiple axes
        axes = [self.make_axis_non_negative(ax) for ax in self.axis]
        # Ensure all axes are greater than 0 (not batch dimension)
        assert all(ax > 0 for ax in axes)
        # For lw/uw, need to shift by 1 due to an extra leading dimension (num_spec)
        lw = x.lw.sum(dim=[ax + 1 for ax in axes], keepdim=self.keepdim)
        lb = x.lb.sum(dim=axes, keepdim=self.keepdim)
        uw = x.uw.sum(dim=[ax + 1 for ax in axes], keepdim=self.keepdim)
        ub = x.ub.sum(dim=axes, keepdim=self.keepdim)
        return LinearBound(lw, lb, uw, ub)

    def build_gradient_node(self, grad_upstream):
        node_grad = ReduceSumGrad(self.axis, self.keepdim, self.input_shape)
        return [(node_grad, (grad_upstream,), [])]
        

class ReduceSumGrad(Module):
    def __init__(self, axis, keepdim, input_shape):
        super().__init__()
        self.axis = axis
        self.keepdim = keepdim
        self.input_shape = input_shape
    
    def forward(self, grad_last):
        grad_new = grad_last.clone()
        if not self.keepdim:
            for axis in self.axis:
                if axis > 0:
                    grad_new = grad_new.unsqueeze(axis + 1)
        # For ReduceSum, ∂y/∂x = 1, so we just need to expand the gradient
        # along each axis that is reduced.
        shape = list(grad_new.shape)
        shape[2:] = self.input_shape[1:]
        grad_new = grad_new.expand(*shape)
        return grad_new
