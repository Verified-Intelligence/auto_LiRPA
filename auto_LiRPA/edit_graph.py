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
"""Edit the computational graph in BoundedModule."""

from auto_LiRPA.bound_ops import Bound

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


# Make sure the nodes already have `name` and `input_name`
def add_nodes(self: 'BoundedModule', nodes):
    # TODO check duplicate names
    nodes = [(node if isinstance(node, Bound) else node.bound_node)
                for node in nodes]
    self.last_bound_target_name = None
    for node in nodes:
        if node.name in self._modules:
            raise NameError(f'Node with name {node.name} already exists')
        self._modules[node.name] = node
        node.output_name = []
        if len(node.inputs) == 0:
            self.root_names.append(node.name)
    for node in nodes:
        for l_pre in node.inputs:
            l_pre.output_name.append(node.name)
        if (getattr(node, 'has_constraint', False) and
                node.name not in self.layers_with_constraint):
            self.layers_with_constraint.append(node.name)


def add_input_node(
        self: 'BoundedModule', node, model_input_index=None):
    """Add a leaf node and assign its flattened model-input position."""
    if model_input_index is None:
        model_input_index = node.model_input_index
    node.model_input_index = model_input_index
    self.add_nodes([node])
    self.input_name.append(node.name)


def _ancestors_of(self: 'BoundedModule', target):
    """Return ``target`` and every node reachable through its inputs."""
    ancestors = set()
    pending = [target]
    while pending:
        node = pending.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        pending.extend(node.inputs)
    return ancestors


def delete_node(self: 'BoundedModule', node):
    for node_inp in node.inputs:
        node_inp.output_name.pop(node_inp.output_name.index(node.name))
    self._modules.pop(node.name)
    self.bound_ancestors.discard(node)
    self.disabled_split_nodes.discard(node)
    self.last_bound_target_name = None
    if node.name in self.input_name:
        self.input_name.remove(node.name)
    if node.name in self.root_names:
        self.root_names.remove(node.name)
    # TODO Create a list to contain all such lists such as
    # "relus" and "optimizable_activations"
    self.relus = [
        item for item in self.relus if item != node]
    self.optimizable_activations = [
        item for item in self.optimizable_activations if item != node]


def replace_node(self: 'BoundedModule', node_old, node_new):
    assert node_old != node_new
    for node in self.nodes():
        for i in range(len(node.inputs)):
            if node.inputs[i] == node_old:
                node.inputs[i] = node_new
    node_new.output_name += node_old.output_name
    if self.final_name == node_old.name:
        self.final_name = node_new.name
    for i in range(len(self.output_name)):
        if self.output_name[i] == node_old.name:
            self.output_name[i] = node_new.name
    self.delete_node(node_old)
