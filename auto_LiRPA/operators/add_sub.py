from .base import *
from .solver_utils import grb


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
        return x[0] + y[0], x[1] + y[1]

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
            model.addConstr(var == (var1 + var2), name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        # reshape to the correct list shape of solver vars
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape[1:]).tolist()
        model.update()


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

        lw = add_w(x.lw, -y.uw, x.lb, y.lb)
        uw = add_w(x.uw, -y.lw, x.ub, y.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        return x[0] - y[1], x[1] - y[0]

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
