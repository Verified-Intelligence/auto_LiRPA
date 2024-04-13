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
        value = roots[i].forward()
        # if isinstance(root[i], BoundInput) and not isinstance(root[i], BoundParams):
        if type(roots[i]) is BoundInput:
            # create input vars for gurobi self.model
            if set_input:
                inp_gurobi_vars = self._build_solver_input(roots[i])
        else:
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

def _reset_solver_vars(self: 'BoundedModule', node: Bound):
    if hasattr(node, 'solver_vars'):
        del node.solver_vars
    for n in node.inputs:
        self._reset_solver_vars(n)

def _build_solver_input(self: 'BoundedModule', node):
    ## Do the input layer, which is a special case
    assert isinstance(node, BoundInput)
    assert node.perturbation is not None
    assert node.perturbation.norm == float("inf")

    if self.solver_model is None:
        self.solver_model = grb.Model()
    # zero var will be shared within the solver model
    zero_var = self.solver_model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    one_var = self.solver_model.addVar(lb=1, ub=1, obj=0, vtype=grb.GRB.CONTINUOUS, name='one')
    neg_one_var = self.solver_model.addVar(lb=-1, ub=-1, obj=0, vtype=grb.GRB.CONTINUOUS, name='neg_one')

    x_L = node.value - node.perturbation.eps if node.perturbation.x_L is None else node.perturbation.x_L
    x_U = node.value + node.perturbation.eps if node.perturbation.x_U is None else node.perturbation.x_U
    x_L = x_L.squeeze(0)
    x_U = x_U.squeeze(0)

    # Recursive function to create Gurobi variables
    idx = [0]
    def create_gurobi_vars(x_L, x_U, solver_model):
        if x_L.dim() == 0:
            v = solver_model.addVar(lb=x_L, ub=x_U, obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'inp_{idx[0]}')
            idx[0] += 1
            return v
        else:
            vars_list = []
            for i, (sub_x_L, sub_x_U) in enumerate(zip(x_L, x_U)):
                vars_list.append(create_gurobi_vars(sub_x_L, sub_x_U, solver_model))
            return vars_list

    inp_gurobi_vars = create_gurobi_vars(x_L, x_U, self.solver_model)

    node.solver_vars = inp_gurobi_vars
    # Save the gurobi input variables so that we can later extract primal values in input space easily.
    self.input_vars = inp_gurobi_vars
    self.solver_model.update()
    return inp_gurobi_vars

