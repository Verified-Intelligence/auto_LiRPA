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
""" Constant operators, including operators that are usually fixed nodes and not perturbed """
from .base import *


class BoundConstant(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.value = attr['value'].to(self.device)
        self.forward_value = self.value
        self.use_default_ibp = True
        self.blocks_jacobian = True

    def __repr__(self):
        if self.value.numel() == 1:
            return f'BoundConstant(name={self.name}, value={self.value})'
        else:
            return super().__repr__()

    def forward(self):
        return self.value.to(self.device)

    def bound_backward(self, last_lA, last_uA, **kwargs):
        def _bound_oneside(A):
            if A is None:
                return 0.0

            if type(A) == Tensor:
                if A.ndim > 2:
                    A = torch.sum(A, dim=list(range(2, A.ndim)))
            elif type(A) == Patches:
                assert A.padding == 0 or A.padding == (0, 0, 0, 0) or self.value == 0  # FIXME (09/19): adding padding here.
                patches_reshape = torch.sum(A.patches, dim=(-1, -2, -3)) * self.value.to(self.device)
                # Expected shape for bias is (spec, batch, out_h, out_w) or (unstable_size, batch)
                return patches_reshape

            return A * self.value.to(self.device)

        lbias = _bound_oneside(last_lA)
        ubias = _bound_oneside(last_uA)
        return [], lbias, ubias

    def bound_forward(self, dim_in):
        lw = uw = torch.zeros(
            dim_in, dtype=self.value.dtype, device=self.device)
        lb = ub = self.value
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.value


class BoundPrimConstant(Bound):
    def forward(self):
        return torch.tensor([], device=self.device)


class BoundConstantOfShape(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.fill_value = attr['value'].to(self.device)
        self.blocks_jacobian = True

    def forward(self, x):
        self.x = x
        self.batch_dim = 0
        return self.fill_value.expand(*list(x))

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        if last_lA is not None:
            lower_sum_b = last_lA * self.fill_value
            while lower_sum_b.ndim > 2:
                lower_sum_b = torch.sum(lower_sum_b, dim=-1)
        else:
            lower_sum_b = 0

        if last_uA is not None:
            upper_sum_b = last_uA * self.fill_value
            while upper_sum_b.ndim > 2:
                upper_sum_b = torch.sum(upper_sum_b, dim=-1)
        else:
            upper_sum_b = 0

        return [(None, None)], lower_sum_b, upper_sum_b

    def bound_forward(self, dim_in, x):
        assert (len(self.x) >= 1)
        lb = ub = torch.ones(
            self.output_shape, dtype=self.fill_value.dtype,
            device=self.device) * self.fill_value
        lw = uw = torch.zeros(
            self.x[0], dim_in, *self.x[1:], dtype=self.fill_value.dtype,
            device=self.device)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        self.x = v[0].lower
        value = torch.ones(
            tuple(v[0].lower), dtype=self.fill_value.dtype,
            device=self.device) * self.fill_value
        return Interval(value, value)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v)


class BoundRange(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.device = attr['device']

    def forward(self, start, end, step):
        if start.dtype == end.dtype == step.dtype == torch.int64:
            return torch.arange(start, end, step, dtype=torch.int64, device=self.device)
        else:
            return torch.arange(start, end, step, device=self.device)


class BoundATenDiag(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.device = attr['device']

    def forward(self, x, diagonal=0):
        return torch.diag(x, diagonal=diagonal)

    def interval_propagate(self, *v):
        if self.is_input_perturbed(1):
            raise OperatorCapabilityError(
                self, 'IBP requires a fixed diagonal offset')
        lower = torch.diag(v[0].lower, v[1].lower)
        return Interval(
            lower, torch.diag(v[0].upper, v[1].lower))


class BoundATenDiagonal(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.device = attr['device']

    def forward(self, x, offset=0, dim1=0, dim2=1):
        return torch.diagonal(x, offset=offset, dim1=dim1, dim2=dim2)

    def interval_propagate(self, *v):
        if any(self.is_input_perturbed(i) for i in range(1, len(v))):
            raise OperatorCapabilityError(
                self, 'IBP requires fixed diagonal dimensions and offset')
        params = (v[1].lower, v[2].lower, v[3].lower)
        lower = torch.diagonal(v[0].lower, *params)
        return Interval(lower, torch.diagonal(v[0].upper, *params))

    def bound_backward(self, last_lA, last_uA, *args, **kwargs):
        for i in range(1, 4):
            assert isinstance(self.inputs[i], BoundConstant)

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(*last_A.shape[:2], *self.inputs[0].output_shape[1:]).to(last_A)
            dim1 = self.inputs[2].forward_value
            dim2 = self.inputs[3].forward_value
            assert dim1 != 0 and dim2 != 0
            if dim1 > 0:
                dim1 += 1
            if dim2 > 0:
                dim2 += 1
            A = torch.diagonal_scatter(
                A, last_A,
                offset=self.inputs[1].forward_value, dim1=dim1, dim2=dim2)
            return A

        return ([(_bound_oneside(last_lA), _bound_oneside(last_uA))]
                + [(None, None)] * 3), 0, 0
