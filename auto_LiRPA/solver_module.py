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
from .bound_ops import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def build_solver_module(self: 'BoundedModule', x=None, C=None, interm_bounds=None,
                        final_node_name=None, model_type="mip", solver_pkg="gurobi", set_input=True):
    r"""build lp/mip solvers in general graph.

    Args:
        x: inputs, a list of BoundedTensor. If set to None, we reuse exisint bounds that
        were previously computed in compute_bounds().
        C (Tensor): The specification matrix that can map the output of the model with an
        additional linear layer. This is usually used for maping the logits output of the
        model to classification margins.
        interm_bounds: if specified, will replace existing intermediate layer bounds.
        Otherwise we reuse exising intermediate bounds.

        final_node_name (String): the name for the target layer to optimize

        solver_pkg (String): the backbone of the solver, default gurobi, also support scipy

    Returns:
        output vars (list): a list of final nodes to optimize
    """
    # self.root_names: list of root node name
    # self.final_name: list of output node name
    # self.final_node: output module
    # <module>.input: a list of input modules of this layer module
    # <module>.solver_vars: a list of gurobi vars of every layer module
    #       list with conv shape if conv layers, otherwise flattened
    # if last layer we need to be careful with:
    #       C: specification matrix
    #       <module>.is_input_perturbed(1)
    if x is not None:
        assert interm_bounds is not None
        # Set the model to use new intermediate layer bounds, ignore the original ones.
        self.set_input(x, interm_bounds=interm_bounds)

    roots = [self[name] for name in self.root_names]

    # create interval ranges for input and other weight parameters
    for i in range(len(roots)):
        # if isinstance(root[i], BoundInput) and not isinstance(root[i], BoundParams):
        if type(roots[i]) is BoundInput:
            # create input vars for gurobi self.model
            if set_input:
                inp_gurobi_vars = self._build_solver_input(roots[i])
        else:
            value = roots[i].forward()
            # regular weights
            roots[i].solver_vars = value

    final = self.final_node() if final_node_name is None else self[final_node_name]

    # backward propagate every layer including last layer
    self._build_solver_general(node=final, C=C, model_type=model_type, solver_pkg=solver_pkg)

    # a list of output solver vars
    return final.solver_vars


def _build_solver_general(self: 'BoundedModule', node: Bound, C=None, model_type="mip",
                          solver_pkg="gurobi"):
    if not hasattr(node, 'solver_vars'):
        if not node.perturbed:
            # if not perturbed, just forward
            node.solver_vars = self.get_forward_value(node)
            return node.solver_vars
        for n in node.inputs:
            self._build_solver_general(n, C=C, model_type=model_type)
        inp = [n_pre.solver_vars for n_pre in node.inputs]
        if C is not None and isinstance(node, BoundLinear) and\
                not node.is_input_perturbed(1) and self.final_name == node.name:
            # when node is the last layer
            # merge the last BoundLinear node with the specification,
            # available when weights of this layer are not perturbed
            solver_vars = node.build_solver(*inp, model=self.solver_model, C=C,
                model_type=model_type, solver_pkg=solver_pkg)
        else:
            solver_vars = node.build_solver(*inp, model=self.solver_model, C=None,
                    model_type=model_type, solver_pkg=solver_pkg)
        # just return output node gurobi vars
        return solver_vars

def _reset_solver_vars(self: 'BoundedModule', node: Bound, iteration=True):
    if hasattr(node, 'solver_vars'):
        del node.solver_vars
    if iteration:
        if hasattr(node, 'inputs'):
            for n in node.inputs:
                self._reset_solver_vars(n)
                
def _reset_solver_model(self: 'BoundedModule'):
    self.solver_model.remove(self.solver_model.getVars())
    self.solver_model.remove(self.solver_model.getConstrs())
    self.solver_model.update()

def _build_solver_input(self: 'BoundedModule', node):
    ## Do the input layer, which is a special case
    assert isinstance(node, BoundInput)
    assert node.perturbation is not None

    if self.solver_model is None:
        self.solver_model = grb.Model()
    # zero var will be shared within the solver model
    zero_var = self.solver_model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    one_var = self.solver_model.addVar(lb=1, ub=1, obj=0, vtype=grb.GRB.CONTINUOUS, name='one')
    neg_one_var = self.solver_model.addVar(lb=-1, ub=-1, obj=0, vtype=grb.GRB.CONTINUOUS, name='neg_one')

    x_L = node.value - node.perturbation.eps if node.perturbation.x_L is None else node.perturbation.x_L
    x_U = node.value + node.perturbation.eps if node.perturbation.x_U is None else node.perturbation.x_U
    x_L = x_L.min(dim=0).values
    x_U = x_U.max(dim=0).values

    input_shape = x_L.shape
    name_array = [f'inp_{idx}' for idx in range(prod(input_shape))]
    inp_gurobi_vars_dict = self.solver_model.addVars(*input_shape, lb=x_L, ub=x_U,
                                                      obj=0, vtype=grb.GRB.CONTINUOUS, name=name_array)

    inp_gurobi_vars = np.empty(input_shape, dtype=object)
    for idx in inp_gurobi_vars_dict:
        inp_gurobi_vars[idx] = inp_gurobi_vars_dict[idx]
    inp_gurobi_vars = inp_gurobi_vars.tolist()
    
    # Flatten the input solver_vars. 
    def flatten(x):
        if isinstance(x, list):
            result = []
            for item in x:
                result.extend(flatten(item))
            return result
        else:
            return [x]

    # Add extra constraints for the inputs if the perturbation norm is not L_inf.
    if node.perturbation.norm != float("inf"):
        if isinstance(inp_gurobi_vars, (list, tuple)):
            flat_inp_gurobi_vars = flatten(inp_gurobi_vars)
        else:
            flat_inp_gurobi_vars = inp_gurobi_vars
        if hasattr(node.value[0], "flatten"):
            flat_node_value = node.value.flatten().tolist()
        else:
            flat_node_value = node.value
        assert len(flat_inp_gurobi_vars) == len(flat_node_value), "The input doesn't match the variables"

        if node.perturbation.norm == 2:
            # For L2 norm, we directly add a quadratic constraint for cplex compatibility.
            # TODO: Compare efficiency with the second method below. If the second method is faster,
            # we should use it for L2 norm by default (when cplex is not used).
            print(f'setup L2 constraint for input with radius {node.perturbation.eps}.')
            quad_expr = grb.QuadExpr()
            for var, val in zip(flat_inp_gurobi_vars, flat_node_value):
                quad_expr.add((var - val) * (var - val))

            self.solver_model.addQConstr(
                quad_expr <= node.perturbation.eps ** 2,
                name="l2_perturbation"
            )
        else:
            print(f'setup Lp constraint for input with radius {node.perturbation.eps}.')
            n = len(flat_inp_gurobi_vars)
            # Create variables to set up the lp constraint.
            # We set input = x0 + delta where delta is under the Lp norm constraint.
            senses = ['='] * n
            delta_vars = self.solver_model.addVars(
                n,
                lb=-grb.GRB.INFINITY,
                ub=grb.GRB.INFINITY,
                name="delta"
            )
            diff = -np.array(flat_node_value)
            vars_list = list(delta_vars.values()) + flat_inp_gurobi_vars
            self.solver_model.update()
            A = np.hstack([np.eye(n), -np.eye(n)])
            # Add constraints input = x0 + delta as delta - input = -x0.
            # Here x0 is "flat_node_value" and input is "flat_inp_gurobi_vars".
            self.solver_model.addMConstr(A, vars_list, senses, diff)
            # Set up the lp constraint here: \| delta \|_p <= eps.
            lp_norm_var = self.solver_model.addVar(
                lb=0, 
                vtype=grb.GRB.CONTINUOUS,
                name="lp_norm"
            )
            self.solver_model.addGenConstrNorm(
                lp_norm_var,
                delta_vars,
                node.perturbation.norm,
                name="lp_norm_constr"
            )
            self.solver_model.addConstr(
                lp_norm_var <= node.perturbation.eps,
                name="lp_perturbation_radius"
            )
    
    node.solver_vars = inp_gurobi_vars
    # Save the gurobi input variables so that we can later extract primal values in input space easily.
    self.input_vars = inp_gurobi_vars
    self.solver_model.update()
    return inp_gurobi_vars

