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
from torch.nn import Module
from .base import *
from ..patches import Patches, patches_to_matrix
from .linear import BoundLinear
from .constant import BoundConstant


class BoundReshape(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # It can be set to `view`, so that `view` instead of `reshape` will be used.
        self.option = options.get('reshape', 'reshape')

    def forward(self, x, shape):
        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = prod(x.shape) // int(prod(shape[:i]) * prod(shape[(i + 1):]))
        self.shape = shape
        if self.option == 'view':
            return x.contiguous().view(shape)
        else:
            return x.reshape(shape)

    def bound_backward(self, last_lA, last_uA, x, shape, **kwargs):
        def _bound_oneside(A):
            if A is None:
                return None
            if type(A) == Patches:
                if type(self.inputs[0]) == BoundLinear:
                    # Save the shape and it will be converted to matrix in Linear layer.
                    return A.create_similar(input_shape=self.output_shape)
                if A.unstable_idx is None:
                    patches = A.patches
                    # non-sparse: [batch, out_dim, out_c, out_H, out_W, out_dim, in_c, H, W]
                    # [batch, out_dim*out_c, out_H, out_W, out_dim*in_c, H, W]
                    # expected next_A shape [batch, spec, in_c, in_H , in_W].
                    next_A = patches_to_matrix(
                        patches, [
                            self.input_shape[0]*self.input_shape[1],
                            patches.shape[-3],
                            int(math.sqrt(self.input_shape[-1]//A.patches.shape[-3])),
                            int(math.sqrt(self.input_shape[-1]//A.patches.shape[-3]))],
                        A.stride, A.padding)
                else:
                    # sparse: [spec, batch, in_c, patch_H, patch_W] (specs depends on the number of unstable neurons).
                    patches = A.patches
                    # expected next_A shape [batch, spec, input_c, in_H, in_W].
                    next_A = patches_to_matrix(patches, [
                        self.input_shape[0],
                        patches.shape[-3],
                        int(math.sqrt(self.input_shape[-1]//patches.shape[-3])),
                        int(math.sqrt(self.input_shape[-1]//patches.shape[-3]))],
                        A.stride, A.padding, output_shape=A.output_shape,
                        unstable_idx=A.unstable_idx)
                # Reshape it to [batch, spec, *input_shape]  (input_shape is the shape before Reshape operation).
                next_A = next_A.reshape(A.shape[1], -1, *self.input_shape[1:])
                return next_A.transpose(0,1)
            else:
                return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])
        #FIXME check reshape or view
        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, shape):
        batch_size = x.lw.shape[0]
        lw = x.lw.reshape(batch_size, dim_in, *self.shape[1:])
        uw = x.uw.reshape(batch_size, dim_in, *self.shape[1:])
        lb = x.lb.reshape(batch_size, *self.shape[1:])
        ub = x.ub.reshape(batch_size, *self.shape[1:])
        return LinearBound(lw, lb, uw, ub)

    def bound_dynamic_forward(self, x, shape, max_dim=None, offset=0):
        w = x.lw.reshape(x.lw.shape[0], x.lw.shape[1], *self.shape[1:])
        b = x.lb.reshape(x.lb.shape[0], *self.shape[1:])
        return LinearBound(w, b, w, b, x_L=x.x_L, x_U=x.x_U, tot_dim=x.tot_dim)

    def interval_propagate(self, *v):
        return Interval.make_interval(
            self.forward(v[0][0], v[1][0]),
            self.forward(v[0][1], v[1][0]), v[0])

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor):
            self.solver_vars = self.forward(*v)
            return
        gvar_array = np.array(v[0])
        gvar_array = gvar_array.reshape(v[1].detach().cpu().numpy())[0]
        self.solver_vars = gvar_array.tolist()

    def build_gradient_node(self, grad_upstream):
        node_grad = ReshapeGrad()
        grad_input = (grad_upstream, self.inputs[0].forward_value)
        return [(node_grad, grad_input, [])]


class BoundUnsqueeze(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        if 'axes' in attr:
            self.axes = attr['axes']
            assert len(self.axes) == 1
            self.axes = self.axes[0]
        else:
            self.axes = None

    def forward(self, *x):
        data = x[0]
        if self.axes is not None:
            axes = self.axes
        else:
            axes = x[1].item()
        return data.unsqueeze(axes)

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        if self.axes is not None:
            axes = self.make_axis_non_negative(self.axes, 'output')
        else:
            axes = self.make_axis_non_negative(x[1].value.item(), 'output')
        if axes == 0:
            raise ValueError("Unsqueezing with axes == 0 is not allowed")
        else:
            def squeeze_A(last_A):
                if type(last_A) == Patches:
                    return Patches(
                        last_A.patches.squeeze(axes - 5),
                        last_A.stride, last_A.padding, last_A.shape,
                        last_A.identity, last_A.unstable_idx, last_A.output_shape)
                elif last_A is not None:
                    return last_A.squeeze(axes + 1)
                else:
                    return None
            lA = squeeze_A(last_lA)
            uA = squeeze_A(last_uA)
            return [(lA, uA), (None, None)], 0, 0

    def bound_forward(self, dim_in, *x):
        if self.axes is not None:
            axes = self.axes
        else:
            axes = self.make_axis_non_negative(x[1].lb.item(), 'output')
        x = x[0]
        if len(self.input_shape) == 0:
            lw, lb = x.lw.unsqueeze(1), x.lb.unsqueeze(0)
            uw, ub = x.uw.unsqueeze(1), x.ub.unsqueeze(0)
        else:
            lw, lb = x.lw.unsqueeze(axes + 1), x.lb.unsqueeze(axes)
            uw, ub = x.uw.unsqueeze(axes + 1), x.ub.unsqueeze(axes)
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0])


class BoundSqueeze(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        if 'axes' in attr:
            self.axes = attr['axes']
            assert len(self.axes) == 1
            self.axes = self.axes[0]
        else:
            self.axes = None

    def forward(self, *x):
        data = x[0]
        if self.axes is not None:
            axes = self.axes
        else:
            axes = x[1].item()
        return data.squeeze(axes)

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        if self.axes is not None:
            axes = self.axes
        else:
            axes = self.make_axis_non_negative(x[1].value.item(), 'output')
        if axes == 0:
            raise ValueError("Squeezing with axes == 0 is not allowed")
        return [(last_lA.unsqueeze(axes + 1) if last_lA is not None else None,
                 last_uA.unsqueeze(axes + 1) if last_uA is not None else None),
                (None, None)], 0, 0

    def bound_forward(self, dim_in, *x):
        if self.axes is not None:
            axes = self.axes
        else:
            axes = self.make_axis_non_negative(x[1].lb.item(), 'output')
        x = x[0]
        return LinearBound(
            x.lw.squeeze(axes + 1),
            x.lb.squeeze(axes),
            x.uw.squeeze(axes + 1),
            x.ub.squeeze(axes)
        )

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0])


class BoundFlatten(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.axis = attr['axis']

    def forward(self, x):
        return torch.flatten(x, self.axis)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        def _bound_oneside(A):
            if A is None:
                return None
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])
        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_dynamic_forward(self, x, max_dim=None, offset=0):
        w = torch.flatten(x.lw, self.axis + 1)
        b = torch.flatten(x.lb, self.axis)
        return LinearBound(w, b, w, b, x_L=x.x_L, x_U=x.x_U, tot_dim=x.tot_dim)

    def bound_forward(self, dim_in, x):
        self.axis = self.make_axis_non_negative(self.axis)
        assert self.axis > 0
        return LinearBound(
            torch.flatten(x.lw, self.axis + 1),
            torch.flatten(x.lb, self.axis),
            torch.flatten(x.uw, self.axis + 1),
            torch.flatten(x.ub, self.axis),
        )

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., v[0] input shape (16, 8, 8) => output shape (1024,)
        self.solver_vars = np.array(v[0]).reshape(-1).tolist()
        model.update()

    def build_gradient_node(self, grad_upstream):
        node_grad = ReshapeGrad()
        grad_input = (grad_upstream, self.inputs[0].forward_value)
        return [(node_grad, grad_input, [])]


class ReshapeGrad(Module):
    def forward(self, grad_last, inp):
        if grad_last.numel() == inp.numel():
            return grad_last.reshape(grad_last.shape[0], *inp.shape[1:])
        else:
            return grad_last.reshape(*grad_last.shape[:2], *inp.shape[1:])


class BoundTranspose(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.perm = attr['perm']
        self.perm_inv_inc_one = [-1] * (len(self.perm) + 1)
        self.perm_inv_inc_one[0] = 0
        for i in range(len(self.perm)):
            self.perm_inv_inc_one[self.perm[i] + 1] = i + 1
        self.use_default_ibp = True
        self.ibp_intermediate = True

    def forward(self, x):
        return x.permute(*self.perm)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return last_A.permute(self.perm_inv_inc_one)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        if self.input_shape[0] != 1:
            perm = [0] + [(p + 1) for p in self.perm]
        else:
            assert (self.perm[0] == 0)
            perm = [0, 1] + [(p + 1) for p in self.perm[1:]]
        lw, lb = x.lw.permute(*perm), x.lb.permute(self.perm)
        uw, ub = x.uw.permute(*perm), x.ub.permute(self.perm)

        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)

    def build_gradient_node(self, grad_upstream):
        node_grad = TransposeGrad(self.perm_inv_inc_one)
        grad_input = (grad_upstream,)
        return [(node_grad, grad_input, [])]


class TransposeGrad(Module):
    def __init__(self, perm_inv):
        super().__init__()
        self.perm_inv = perm_inv

    def forward(self, grad_last):
        return grad_last.permute(*self.perm_inv)
