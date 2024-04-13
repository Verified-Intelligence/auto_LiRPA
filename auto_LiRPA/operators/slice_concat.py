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
""" Shape operators """
from torch.nn import Module
from torch.autograd import Function
from .base import *
from ..patches import Patches
from .constant import BoundConstant


class BoundConcat(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.IBP_rets = None

    def forward(self, *x):  # x is a list of tensors
        x = [(item if isinstance(item, Tensor) else torch.tensor(item)) for item in x]
        self.input_size = [item.shape[self.axis] for item in x]
        self.axis = self.make_axis_non_negative(self.axis)
        return torch.cat(x, dim=int(self.axis))

    def interval_propagate(self, *v):
        norms = []
        eps = []
        # Collect perturbation information for all inputs.
        for i, _v in enumerate(v):
            if self.is_input_perturbed(i):
                n, e = Interval.get_perturbation(_v)
                norms.append(n)
                eps.append(e)
            else:
                norms.append(None)
                eps.append(0.0)
        eps = np.array(eps)
        # Supporting two cases: all inputs are Linf norm, or all inputs are L2 norm perturbed.
        # Some inputs can be constants without perturbations.
        all_inf = all(map(lambda x: x is None or x == torch.inf, norms))
        all_2 = all(map(lambda x: x is None or x == 2, norms))

        h_L = [_v[0] for _v in v]
        h_U = [_v[1] for _v in v]
        if all_inf:
            # Simply returns a tuple. Every subtensor has its own lower and upper bounds.
            return self.forward(*h_L), self.forward(*h_U)
        elif all_2:
            # Sum the L2 norm over all subtensors, and use that value as the new L2 norm.
            # This will be an over-approximation of the original perturbation (we can prove it).
            max_eps = np.sqrt(np.sum(eps * eps))
            # For L2 norm perturbed inputs, lb=ub and for constants lb=ub. Just propagate one object.
            r = self.forward(*h_L)
            ptb = PerturbationLpNorm(norm=2, eps=max_eps)
            return Interval(r, r, ptb=ptb)
        else:
            raise RuntimeError(f"BoundConcat does not support inputs with norm {norms}")

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        self.axis = self.make_axis_non_negative(self.axis, 'output')
        assert self.axis > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if isinstance(last_A, torch.Tensor):
                return torch.split(last_A, self.input_size, dim=self.axis + 1)
            elif isinstance(last_A, Patches):
                assert len(self.input_shape) == 4 and self.axis == 1, "Split channel dimension is supported; others are unimplemented."
                # Patches shape can be [out_c, batch, out_h, out_w, in_c, patch_h, patch_w]
                # Or [spec, batch, in_c, patch_h, patch_w]  (sparse)
                new_patches = torch.split(last_A.patches, self.input_size, dim=-3)  # split the in_c dimension is easy.
                return [last_A.create_similar(p) for p in new_patches]
            else:
                raise RuntimeError(f'Unsupported type for last_A: {type(last_A)}')

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)
        if uA is None:
            return [(lA[i] if lA is not None else None, None) for i in range(len(lA))], 0, 0
        if lA is None:
            return [(None, uA[i] if uA is not None else None) for i in range(len(uA))], 0, 0
        return [(lA[i], uA[i]) for i in range(len(lA))], 0, 0

    def bound_forward(self, dim_in, *x):
        self.axis = self.make_axis_non_negative(self.axis)
        assert (self.axis == 0 and not self.from_input or self.from_input)
        lw = torch.cat([item.lw for item in x], dim=self.axis + 1)
        lb = torch.cat([item.lb for item in x], dim=self.axis)
        uw = torch.cat([item.uw for item in x], dim=self.axis + 1)
        ub = torch.cat([item.ub for item in x], dim=self.axis)
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)

    def build_gradient_node(self, grad_upstream):
        ret = []
        for i in range(len(self.inputs)):
            node_grad = ConcatGrad(self.axis, i)
            grad_input = (grad_upstream, ) + tuple(inp.forward_value for inp in self.inputs)
            ret.append((node_grad, grad_input, []))
        return ret


BoundConcatFromSequence = BoundConcat


class BoundSlice(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.start = attr["starts"][0] if "starts" in attr else None
        self.end = attr["ends"][0] if "ends" in attr else None
        self.axes = attr["axes"][0] if "axes" in attr else None
        self.use_default_ibp = False

    def __repr__(self):
        attrs = {}
        if (len(self.inputs) == 5
            and all(isinstance(item, BoundConstant) and item.value.numel() == 1
                    for item in self.inputs[1:])):
            attrs['start'] = self.inputs[1].value.item()
            attrs['end'] = self.inputs[2].value.item()
            attrs['axes'] = self.inputs[3].value.item()
            attrs['step'] = self.inputs[4].value.item()
        return super().__repr__(attrs)

    def _fixup_params(self, shape, start, end, axes, steps):
        if start < 0:
            start += shape[axes]
        if end < 0:
            if end == -9223372036854775807:  # -inf in ONNX
                end = 0  # only possible when step == -1
            else:
                end += shape[axes]
        if steps == -1:
            start, end = end, start + 1  # TODO: more test more negative step size.
        end = min(end, shape[axes])
        return start, end

    # Older Pytorch version only passes steps as input.
    def forward(self, x, start=None, end=None, axes=None, steps=1):
        start = self.start if start is None else start
        end = self.end if end is None else end
        axes = self.axes if axes is None else axes
        assert (steps == 1 or steps == -1) and axes == int(axes) and start == int(start) and end == int(end)
        shape = x.shape if isinstance(x, Tensor) else [len(x)]
        start, end = self._fixup_params(shape, start, end, axes, steps)
        final = torch.narrow(x, dim=int(axes), start=int(start), length=int(end - start))
        if steps == -1:
            final = torch.flip(final, dims=tuple(axes))
        return final

    def interval_propagate(self, *v):
        lb = tuple(map(lambda x:x[0],v))
        ub = tuple(map(lambda x:x[1],v))
        return Interval.make_interval(self.forward(*lb), self.forward(*ub))

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        def _bound_oneside(A, start, end, axes, steps):
            if A is None:
                return None
            if isinstance(A, torch.Tensor):
                # Reuse the batch and spec dimension of A, and replace other shapes with input.
                A_shape = A.shape[:2] + self.input_shape[1:]
                new_A = torch.zeros(size=A_shape, device=A.device,
                                    requires_grad=A.requires_grad)
                # Fill part of the new_A based on start, end, axes and steps.
                # Skip the spec dimension at the front (axes + 1).
                dim = axes if axes < 0 else axes + 1
                indices = torch.arange(start, end, device=A.device)
                new_A = torch.index_copy(new_A, dim=dim, index=indices, source=A)
            elif isinstance(A, Patches):
                assert A.unstable_idx is None
                assert len(self.input_shape) == 4 and axes == 1, "Slice is only supported on channel dimension."
                patches = A.patches
                # patches shape is [out_c, batch, out_h, out_w, in_c, patch_h, patch_w].
                new_patches_shape = patches.shape[:4] + (self.input_shape[1], ) + patches.shape[-2:]
                new_patches = torch.zeros(
                    size=new_patches_shape, device=patches.device,
                    requires_grad=patches.requires_grad)
                indices = torch.arange(start, end, device=patches.device)
                new_patches = torch.index_copy(new_patches, dim=-3, index=indices, source=patches)
                # Only the in_c dimension is changed.
                new_A = A.create_similar(new_patches)
            else:
                raise ValueError(f'Unsupport A type {type(A)}')
            return new_A

        start, end, axes = x[1].value.item(), x[2].value.item(), x[3].value.item()
        steps = x[4].value.item() if len(x) == 5 else 1  # If step is not specified, it is 1.
        # Other step size untested, do not enable for now.
        assert steps == 1 and axes == int(axes) and start == int(start) and end == int(end)
        start, end = self._fixup_params(self.input_shape, start, end, axes, steps)
        # Find the original shape of A.
        lA = _bound_oneside(last_lA, start, end, axes, steps)
        uA = _bound_oneside(last_uA, start, end, axes, steps)
        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], 0, 0

    def bound_forward(self, dim_in, *inputs):
        assert len(inputs) == 5 or len(inputs) == 4
        start = inputs[1].lb.item()
        end = inputs[2].lb.item()
        axis = self.make_axis_non_negative(inputs[3].lb.item())
        assert axis > 0, "Slicing along the batch dimension is not supported yet"
        steps = inputs[4].lb.item() if len(inputs) == 5 else 1  # If step is not specified, it is 1.
        assert steps in [1, -1]
        x = inputs[0]
        shape = x.lb.shape
        start, end = self._fixup_params(shape, start, end, axis, steps)
        lw = torch.narrow(x.lw, dim=axis+1, start=start, length=end - start)
        uw = torch.narrow(x.uw, dim=axis+1, start=start, length=end - start)
        lb = torch.narrow(x.lb, dim=axis, start=start, length=end - start)
        ub = torch.narrow(x.ub, dim=axis, start=start, length=end - start)
        if steps == -1:
            lw = torch.flip(lw, dims=tuple(axis+1))
            uw = torch.flip(uw, dims=tuple(axis+1))
            lb = torch.flip(lb, dims=tuple(axis))
            ub = torch.flip(ub, dims=tuple(axis))
        return LinearBound(lw, lb, uw, ub)

    def build_gradient_node(self, grad_upstream):
        assert len(self.inputs) == 5
        start = self.inputs[1].value.item()
        end = self.inputs[2].value.item()
        axes = self.inputs[3].value.item()
        steps = self.inputs[4].value.item()
        assert steps == 1
        node_grad = SliceGrad(start, end, axes, steps)
        grad_input = (grad_upstream, self.inputs[0].forward_value)
        return [(node_grad, grad_input, [])]


class BoundSplit(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.use_default_ibp = True
        if 'split' in attr:
            self.split = attr['split']
        else:
            self.split = None

    def forward(self, *x):
        data = x[0]
        split = self.split if self.split is not None else x[1].tolist()
        if self.axis == -1:
            self.axis = len(data.shape) - 1
        return torch.split(data, split, dim=self.axis)[self.output_index]

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        assert self.axis > 0
        split = self.split if self.split is not None else x[1].value.tolist()
        pre = sum(split[:self.output_index])
        suc = sum(split[(self.output_index + 1):])

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = []
            if pre > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], pre, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            A.append(last_A)
            if suc > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], suc, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            return torch.cat(A, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, *x):
        assert self.axis > 0 and self.from_input
        split = self.split if self.split is not None else x[1].lb.tolist()
        x = x[0]
        lw = torch.split(x.lw, split, dim=self.axis + 1)[self.output_index]
        uw = torch.split(x.uw, split, dim=self.axis + 1)[self.output_index]
        lb = torch.split(x.lb, split, dim=self.axis)[self.output_index]
        ub = torch.split(x.ub, split, dim=self.axis)[self.output_index]
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0])


def slice_grad(x, input_shape, start, end, axes, steps):
    assert steps == 1
    assert axes > 0
    out = torch.zeros(*x.shape[:2], *input_shape[1:]).to(x)
    end = min(end, input_shape[axes])
    index = torch.arange(start, end, device=x.device)
    # Make index.ndim == x.ndim
    index = index.view(
        *((1,) * (axes + 1)),
        end - start,
        *((1,) * (x.ndim - axes - 2)))
    # Make index.shape == x.shape
    index = index.repeat(
        *x.shape[:axes + 1],
        1,
        *x.shape[axes + 2:]
    )
    out.scatter_(axes + 1, index, x)
    return out


class SliceGradOp(Function):
    """ Local gradient of BoundSlice.

    Not including multiplication with gradients from other layers.
    """
    @staticmethod
    def symbolic(_, grad_last, input, start=None, end=None, axes=None, steps=1):
        return _.op(
            'grad::Slice', grad_last, input,
            start_i=start, end_i=end, axes_i=axes, steps_i=steps
        ).setType(grad_last.type())

    @staticmethod
    def forward(ctx, grad_last, input, start, end, axes, steps):
        return slice_grad(grad_last, input.shape, start, end, axes, steps)


class SliceGrad(Module):
    def __init__(self, start, end, axes, steps):
        super().__init__()
        self.start = start
        self.end = end
        self.axes = axes
        self.steps = steps

    def forward(self, grad_last, input):
        return SliceGradOp.apply(
            grad_last, input,
            self.start, self.end, self.axes, self.steps)


class BoundSliceGrad(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.start = attr['start']
        self.end = attr['end']
        self.axes = attr['axes']
        self.steps = attr['steps']
        self.use_default_ibp = True

    def forward(self, grad_last, input):
        return slice_grad(grad_last, input.shape,
                          self.start, self.end, self.axes, self.steps)

    def bound_backward(self, last_lA, last_uA, *args, **kwargs):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            assert self.axes > 0
            last_A_ = last_A.reshape(-1, *self.inputs[1].output_shape[self.axes:])
            last_A_ = last_A_[:, self.start:self.end]
            last_A = last_A_.reshape(
                *last_A.shape[:self.axes+2], -1,
                *self.inputs[1].output_shape[self.axes+1:])
            return last_A
        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)),
                (None, None)], 0, 0


def concat_grad(x, axis, input_index, *inputs):
    cur = 0
    for i in range(input_index):
        cur += inputs[i].shape[axis]
    x_ = x.reshape(-1, *x.shape[axis + 1:])
    ret = x_[:, cur:cur+inputs[input_index].shape[axis]]
    ret = ret.reshape(*x.shape[:axis + 1], *ret.shape[1:])
    return ret


class ConcatGradOp(Function):
    @staticmethod
    def symbolic(_, grad_last, axis, input_index, *inputs):
        return _.op('grad::Concat', grad_last, *inputs,
                    axis_i=axis, input_index_i=input_index).setType(grad_last.type())

    @staticmethod
    def forward(ctx, grad_last, axis, input_index, *inputs):
        return concat_grad(grad_last, axis, input_index, *inputs)


class ConcatGrad(Module):
    def __init__(self, axis, input_index):
        super().__init__()
        self.input_index = input_index
        self.axis = axis

    def forward(self, grad_last, *input):
        return ConcatGradOp.apply(grad_last, self.axis, self.input_index, *input)


class BoundConcatGrad(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.input_index = attr['input_index']
        self.use_default_ibp = True

    def forward(self, grad_last, *inputs):
        return concat_grad(grad_last, self.axis, self.input_index, *inputs)

    def bound_backward(self, last_lA, last_uA, *args, **kwargs):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            assert self.axis > 0
            start = sum([self.inputs[i + 1].output_shape[self.axis]
                         for i in range(self.input_index)])
            end = start + self.output_shape[self.axis+1]
            shape_behind = self.inputs[0].output_shape[self.axis+1:]
            A = torch.zeros(*last_A.shape[:self.axis+2], *shape_behind)
            A = A.view(-1, *shape_behind)
            A[:, start:end] = last_lA.reshape(-1, *last_A.shape[self.axis+2:])
            A = A.view(*last_A.shape[:self.axis+2], *shape_behind)
            return A

        return ([(_bound_oneside(last_lA), _bound_oneside(last_uA))]
                + [(None, None)] * (len(self.inputs) - 1)), 0, 0