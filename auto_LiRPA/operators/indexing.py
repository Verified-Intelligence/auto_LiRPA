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
from .base import *
from ..patches import Patches, patches_to_matrix
from torch.nn import Module


class BoundGather(Bound):
    def __init__(self, attr, x, output_index, options):
        super().__init__(attr, x, output_index, options)
        self.axis = attr['axis'] if 'axis' in attr else 0

    def forward(self, x, indices):
        self.indices = indices
        if self.axis == -1:
            self.axis = len(x.shape) - 1
        # BoundShape.shape() will return values on cpu only
        x = x.to(self.indices.device)
        if indices.ndim == 0:
            if indices == -1:
                self.indices = x.shape[self.axis] + indices
            return torch.index_select(x, dim=self.axis, index=self.indices).squeeze(self.axis)
        elif indices.ndim == 1:
            if self.axis == 0:
                assert not self.perturbed
            # `index_select` requires `indices` to be a 1-D tensor
            return torch.index_select(x, dim=self.axis, index=indices)

        raise ValueError('Unsupported shapes in Gather: '
                         f'data {x.shape}, indices {indices.shape}, '
                         f'axis {self.axis}')

    def bound_backward(self, last_lA, last_uA, *args, **kwargs):
        assert self.from_input

        def _expand_A_with_zeros(A, axis, idx, max_axis_size):
            # Need to recreate A with three parts: before the gathered element, gathered element, and after gathered element.
            tensors = []
            if idx < 0:
                idx = max_axis_size + idx
            if idx > 0:
                shape_pre = list(A.shape)
                shape_pre[axis] *= idx
                # Create the same shape as A, except for the dimension to be gathered.
                tensors.append(torch.zeros(shape_pre, device=A.device))
            # The gathered element itself, in the middle.
            tensors.append(A)
            if max_axis_size - idx - 1 > 0:
                shape_next = list(A.shape)
                shape_next[axis] *= max_axis_size - idx - 1
                # Create the rest part of A.
                tensors.append(torch.zeros(shape_next, device=A.device))
            # Concatenate all three parts together.
            return torch.cat(tensors, dim=axis)

        def _bound_oneside(A):
            if A is None:
                return None

            if isinstance(A, torch.Tensor):
                if self.indices.ndim == 0:
                    A = A.unsqueeze(self.axis + 1)
                    idx = int(self.indices)
                    return _expand_A_with_zeros(A, self.axis + 1, idx, self.input_shape[self.axis])
                else:
                    shape = list(A.shape)
                    final_A = torch.zeros(*shape[:self.axis + 1], self.input_shape[self.axis], *shape[self.axis + 2:], device=A.device)
                    idx = self.indices.view([*[1]*(self.axis+1), -1, *[1]*len(shape[self.axis + 2:])])
                    idx = idx.repeat([*A.shape[:self.axis+1], 1, *A.shape[self.axis+2:]])
                    final_A.scatter_add_(dim=self.axis+1, index=idx, src=A)
                    return final_A
            elif isinstance(A, Patches):
                if self.indices.ndim == 0:
                    idx = int(self.indices)
                    assert len(self.input_shape) == 4 and self.axis == 1, "Gather is only supported on the channel dimension for Patches mode."
                    # For gather in the channel dimension, we only need to deal with the in_c dimension (-3) in patches.
                    patches = A.patches
                    # -3 is the in_c dimension.
                    new_patches = _expand_A_with_zeros(patches, axis=-3, idx=idx, max_axis_size=self.input_shape[self.axis])
                    return A.create_similar(new_patches)
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f'Unknown last_A type {type(A)}')

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, indices):
        assert self.indices.numel() == 1 and self.indices.ndim <= 1 and (self.indices >= 0).all()
        if isinstance(x, torch.Size):
            lw = uw = torch.zeros(dim_in, device=self.device)
            lb = ub = torch.index_select(
                torch.tensor(x, device=self.device),
                dim=self.axis, index=self.indices).squeeze(self.axis)
        else:
            axis = self.axis + 1
            lw = torch.index_select(x.lw, dim=self.axis + 1, index=self.indices)
            uw = torch.index_select(x.uw, dim=self.axis + 1, index=self.indices)
            lb = torch.index_select(x.lb, dim=self.axis, index=self.indices)
            ub = torch.index_select(x.ub, dim=self.axis, index=self.indices)
            if self.indices.ndim == 0:
                lw = lw.squeeze(axis)
                uw = uw.squeeze(axis)
                lb = lb.squeeze(self.axis)
                ub = ub.squeeze(self.axis)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        return self.forward(v[0][0], v[1][0]), self.forward(v[0][1], v[1][0])

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0], v[1])

    def build_gradient_node(self, grad_upstream):
        return [(GatherGrad(self.axis, self.indices, self.input_shape), (grad_upstream,), []), None]


class GatherGrad(Module):
    def __init__(self, axis, indices, input_shape):
        super().__init__()
        self.axis = axis
        self.indices = indices
        self.input_shape = input_shape
    
    def forward(self, grad_last):
        # TODO: It's better to use scatter_add_ instead of cat.
        # This is a workaround for the fact that scatter_add_ does not support negative indices.

        # Scalar indices case (ndim == 0)
        if self.indices.ndim == 0:
            grad_unsq = grad_last.unsqueeze(self.axis)
            
            # Get the scalar index and adjust if negative.
            idx = int(self.indices)
            if idx < 0:
                idx = self.input_shape[self.axis] + idx
            
            # Build the gradient by concatenating three parts along self.axis:
            tensors = []
            # 1. Zeros block before the gathered element (if idx > 0)
            if idx > 0:
                shape_pre = list(grad_unsq.shape)
                shape_pre[self.axis] = idx  # pre-block has size idx along self.axis
                zeros_pre = torch.zeros(shape_pre, dtype=grad_last.dtype, device=grad_last.device)
                tensors.append(zeros_pre)
            
            # 2. The gathered gradient slice (already in grad_unsq)
            tensors.append(grad_unsq)
            
            # 3. Zeros block after the gathered element
            num_after = self.input_shape[self.axis] - idx - 1
            if num_after > 0:
                shape_post = list(grad_unsq.shape)
                shape_post[self.axis] = num_after
                zeros_post = torch.zeros(shape_post, dtype=grad_last.dtype, device=grad_last.device)
                tensors.append(zeros_post)
            
            # Concatenate all parts along self.axis to form the full gradient tensor.
            grad_input = torch.cat(tensors, dim=self.axis)
            return grad_input

        # 1-D indices case (ndim == 1)
        elif self.indices.ndim == 1:
            grad_slices = []
            # Iterate over each position in the original input along self.axis.
            for i in range(self.input_shape[self.axis]):
                # matching: tensor of indices (in grad_last) where the gathered index equals i.
                matching = (self.indices == i).nonzero(as_tuple=False).squeeze(-1)
                
                if matching.numel() == 0:
                    # No matching index: create a zeros slice with the same shape as one slice of grad_last.
                    slice_shape = list(grad_last.shape)
                    slice_shape[self.axis] = 1  # single slice along self.axis
                    grad_slice = torch.zeros(slice_shape, dtype=grad_last.dtype, device=grad_last.device)
                else:
                    # There are one or more matching positions.
                    # For each matching index j, extract the corresponding slice from grad_last.
                    slice_list = []
                    for j in matching.tolist():
                        # Build slicing object：select all elements, but at self.axis take index j.
                        slicer = [slice(None)] * grad_last.dim()
                        slicer[self.axis] = j
                        # Extract the slice and add back the missing dimension.
                        slice_j = grad_last[tuple(slicer)].unsqueeze(self.axis)
                        slice_list.append(slice_j)
                    # Concatenate all slices along self.axis; if there are duplicates, sum them.
                    cat_slices = torch.cat(slice_list, dim=self.axis)
                    # Sum along self.axis to accumulate contributions from duplicate indices.
                    grad_slice = cat_slices.sum(dim=self.axis, keepdim=True)
                # Append the slice corresponding to position i.
                grad_slices.append(grad_slice)
            
            # Concatenate all slices in order along self.axis to form the final gradient tensor.
            grad_input = torch.cat(grad_slices, dim=self.axis)
            return grad_input

        else:
            raise ValueError("Unsupported indices dimensions in gradient for Gather")


class BoundGatherElements(Bound):
    def __init__(self, attr, input, output_index, options):
        super().__init__(attr, input, output_index, options)
        self.axis = attr['axis']

    def forward(self, x, index):
        self.index = index
        return torch.gather(x, dim=self.axis, index=index)

    def bound_backward(self, last_lA, last_uA, x, index, **kwargs):
        assert self.from_input

        dim = self._get_dim()

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(
                last_A.shape[0], last_A.shape[1], *x.output_shape[1:], device=last_A.device)
            A.scatter_(
                dim=dim + 1,
                index=self.index.unsqueeze(0).repeat(A.shape[0], *([1] * (A.ndim - 1))),
                src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        return self.forward(v[0][0], v[1][0]), \
               self.forward(v[0][1], v[1][1])

    def bound_forward(self, dim_in, x, index):
        assert self.axis != 0
        dim = self._get_dim()
        return LinearBound(
            torch.gather(x.lw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.lb, dim=dim, index=self.index),
            torch.gather(x.uw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.ub, dim=dim, index=self.index))

    def _get_dim(self):
        dim = self.axis
        if dim < 0:
            dim = len(self.output_shape) + dim
        return dim
