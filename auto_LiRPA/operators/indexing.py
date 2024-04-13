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
from .base import *
from ..patches import Patches, patches_to_matrix


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
                indices = x.shape[self.axis] + indices
            return torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
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
        assert self.indices.numel() == 1 and self.indices.ndim <= 1 # TODO

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
