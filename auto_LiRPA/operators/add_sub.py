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
from torch.nn import Module
from .base import *
from .constant import BoundConstant
from .solver_utils import grb


def _broadcast_dynamic_forward(w, lb, reference):
    """Bounds whose coefficients come from a single operand.

    Coefficients are shaped ``[batch, variables, *shape]`` and a bias
    ``[batch, *shape]``, so the missing variable axis is inserted before the
    coefficients are broadcast to the merged output shape.
    """
    merged = (lb.shape[0], w.shape[1], *lb.shape[1:])
    if tuple(w.shape) != merged:
        w = w + torch.zeros_like(lb).unsqueeze(1)
    return LinearBound(w, lb, w, lb, tot_dim=reference.tot_dim)


def _combine_dynamic_forward(x, y, sign):
    """Combine two dynamic forward bounds as ``x + sign * y``.

    Both operands lay their coefficients out over the same variable layout
    (see BoundRelu.bound_dynamic_forward), so a column index denotes the same
    bounded variable on either side and the affine forms combine by linearity
    -- negating one included. An operand reaching fewer columns does not depend
    on the variables the rest hold, so widening it with zero coefficients is
    exact.
    """
    lb = x.lb + sign * y.lb
    if x.lw is None and y.lw is None:
        return LinearBound(None, lb, None, lb, tot_dim=0)
    if x.lw is None:
        return _broadcast_dynamic_forward(sign * y.lw, lb, y)
    if y.lw is None:
        return _broadcast_dynamic_forward(x.lw, lb, x)

    width = max(x.lw.shape[1], y.lw.shape[1])

    def pad_columns(w):
        return w if w.shape[1] == width else torch.cat(
            [w, w.new_zeros((w.shape[0], width - w.shape[1], *w.shape[2:]))],
            dim=1)

    lw = pad_columns(x.lw) + sign * pad_columns(y.lw)
    return LinearBound(lw, lb, lw, lb, tot_dim=max(x.tot_dim, y.tot_dim))


class BoundAdd(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        options = options or {}
        # FIXME: This is not the right way to enable patches mode.
        # Instead we must traverse the graph and determine when patches mode needs to be used.

        self.mode = options.get("conv_mode", "matrix")

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y

    def bound_backward(self, last_lA, last_uA, x, y, **kwargs):
        def _bound_oneside(last_A, w):
            if last_A is None:
                return None
            return self.broadcast_backward(last_A, w)

        uA_x = _bound_oneside(last_uA, x)
        uA_y = _bound_oneside(last_uA, y)
        lA_x = _bound_oneside(last_lA, x)
        lA_y = _bound_oneside(last_lA, y)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_dynamic_forward(self, x, y, max_dim=None, offset=0):
        return _combine_dynamic_forward(x, y, 1)

    def bound_forward(self, dim_in, x, y):
        lb, ub = x.lb + y.lb, x.ub + y.ub

        def add_w(x_w, y_w, x_b, y_b):
            if x_w is None and y_w is None:
                return None
            elif x_w is not None and y_w is not None:
                return x_w + y_w
            elif y_w is None:
                return x_w + torch.zeros_like(y_b)
            else:
                return y_w + torch.zeros_like(x_b)

        lw = add_w(x.lw, y.lw, x.lb, y.lb)
        uw = add_w(x.uw, y.uw, x.ub, y.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        assert (not isinstance(y, Tensor))
        return Interval(
            x.lower + y.lower, x.upper + y.upper)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor) and isinstance(v[1], Tensor):
            # constants if both inputs are tensors
            self.solver_vars = self.forward(v[0], v[1])
            return
        # we have both gurobi vars as inputs
        this_layer_shape = self.output_shape
        gvar_array1 = np.array(v[0])
        if isinstance(v[1], Tensor):
            var2 = v[1].cpu().numpy()
            # flatten to create vars and constrs first
            gvar_array1 = gvar_array1.reshape(-1)
            new_layer_gurobi_vars = []
            for neuron_idx, var1 in enumerate(gvar_array1):
                var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                                   vtype=grb.GRB.CONTINUOUS,
                                   name=f'lay{self.name}_{neuron_idx}')
                model.addConstr(var == (var1 + var2), name=f'lay{self.name}_{neuron_idx}_eq')
                new_layer_gurobi_vars.append(var)
        else:
            gvar_array2 = np.array(v[1])
            assert gvar_array1.shape == gvar_array2.shape and gvar_array1.shape == this_layer_shape[1:]

            # flatten to create vars and constrs first
            gvar_array1 = gvar_array1.reshape(-1)
            gvar_array2 = gvar_array2.reshape(-1)
            new_layer_gurobi_vars = []
            for neuron_idx, (var1, var2) in enumerate(zip(gvar_array1, gvar_array2)):
                var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                                vtype=grb.GRB.CONTINUOUS,
                                name=f'lay{self.name}_{neuron_idx}')
                model.addConstr(var == (var1 + var2), name=f'lay{self.name}_{neuron_idx}_eq')
                new_layer_gurobi_vars.append(var)
        # reshape to the correct list shape of solver vars
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape[1:]).tolist()
        model.update()

    def build_gradient_node(self, grad_upstream):
        if not self.inputs[0].blocks_jacobian:
            grad0_node = AddGrad(self.inputs[0].output_shape if self.inputs[0].batch_dim != -1 else
                                 torch.Size((1,) + self.inputs[0].output_shape))
            grad0 = (grad0_node, (grad_upstream,), [])
        else:
            grad0 = None
        if not self.inputs[1].blocks_jacobian:
            grad1_node = AddGrad(self.inputs[1].output_shape if self.inputs[1].batch_dim != -1 else
                                 torch.Size((1,) + self.inputs[1].output_shape))
            grad1 = (grad1_node, (grad_upstream,), [])
        else:
            grad1 = None
        return [grad0, grad1]


class BoundSub(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # FIXME: This is not the right way to enable patches mode. Instead we must traverse the graph and determine when patches mode needs to be used.
        self.mode = options.get("conv_mode", "matrix")

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y

    def bound_backward(self, last_lA, last_uA, x, y, **kwargs):
        def _bound_oneside(last_A, w, sign=-1):
            if last_A is None:
                return None
            if isinstance(last_A, torch.Tensor):
                return self.broadcast_backward(sign * last_A, w)
            elif isinstance(last_A, Patches):
                if sign == 1:
                    # Patches shape requires no broadcast.
                    return last_A
                else:
                    # Multiply by the sign.
                    return last_A.create_similar(sign * last_A.patches)
            else:
                raise ValueError(f'Unknown last_A type {type(last_A)}')

        uA_x = _bound_oneside(last_uA, x, sign=1)
        uA_y = _bound_oneside(last_uA, y, sign=-1)
        lA_x = _bound_oneside(last_lA, x, sign=1)
        lA_y = _bound_oneside(last_lA, y, sign=-1)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_dynamic_forward(self, x, y, max_dim=None, offset=0):
        return _combine_dynamic_forward(x, y, -1)

    def bound_forward(self, dim_in, x, y):
        lb, ub = x.lb - y.ub, x.ub - y.lb

        def add_w(x_w, y_w, x_b, y_b):
            if x_w is None and y_w is None:
                return None
            elif x_w is not None and y_w is not None:
                return x_w + y_w
            elif y_w is None:
                return x_w + torch.zeros_like(y_b)
            else:
                return y_w + torch.zeros_like(x_b)

        # Some nodes such as BoundConstant does not have uw and lw.
        lw = add_w(x.lw, -y.uw if y.uw is not None else None, x.lb, y.lb)
        uw = add_w(x.uw, -y.lw if y.lw is not None else None, x.ub, y.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        return Interval(
            x.lower - y.upper, x.upper - y.lower)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor) and isinstance(v[1], Tensor):
            # constants if both inputs are tensors
            self.solver_vars = self.forward(v[0], v[1])
            return
        # we have both gurobi vars as inputs
        this_layer_shape = self.output_shape
        gvar_array1 = np.array(v[0])
        gvar_array2 = np.array(v[1])
        assert gvar_array1.shape == gvar_array2.shape and gvar_array1.shape == this_layer_shape[1:]

        # flatten to create vars and constrs first
        gvar_array1 = gvar_array1.reshape(-1)
        gvar_array2 = gvar_array2.reshape(-1)
        new_layer_gurobi_vars = []
        for neuron_idx, (var1, var2) in enumerate(zip(gvar_array1, gvar_array2)):
            var = model.addVar(lb=-float('inf'), ub=float('inf'), obj=0,
                            vtype=grb.GRB.CONTINUOUS,
                            name=f'lay{self.name}_{neuron_idx}')
            model.addConstr(var == (var1 - var2), name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        # reshape to the correct list shape of solver vars
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape[1:]).tolist()
        model.update()

    def build_gradient_node(self, grad_upstream):
        if not self.inputs[0].blocks_jacobian:
            grad_node_0 = AddGrad(self.inputs[0].output_shape if self.inputs[0].batch_dim != -1 else
                                  torch.Size((1,) + self.inputs[0].output_shape), w=1.0)
            grad0 = (grad_node_0, (grad_upstream,), [])
        else:
            grad0 = None
        if not self.inputs[1].blocks_jacobian:
            grad_node_1 = AddGrad(self.inputs[1].output_shape if self.inputs[1].batch_dim != -1 else
                                  torch.Size((1,) + self.inputs[1].output_shape), w=-1.0)
            grad1 = (grad_node_1, (grad_upstream,), [])
        else:
            grad1 = None
        return [grad0, grad1]


class AddGrad(Module):
    def __init__(self, input_shape, w=1.0):
        super().__init__()
        # We need the input shape to handle broadcasting.
        self.input_shape = input_shape
        self.w = w

    def forward(self, grad_last):
        return reduce_broadcast_dims(grad_last * self.w, self.input_shape)
