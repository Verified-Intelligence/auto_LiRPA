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
"""Handle Jacobian bounds."""

import torch
from auto_LiRPA.bound_ops import JacobianOP, GradNorm  # pylint: disable=unused-import
from auto_LiRPA.bound_ops import (
    BoundInput, BoundAdd, BoundRelu, BoundJacobianInit,
    BoundJacobianOP)
from auto_LiRPA.utils import logger, prod
from collections import deque

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _expand_jacobian(self):
    self.jacobian_start_nodes = []
    for node in list(self.nodes()):
        if isinstance(node, BoundJacobianOP):
            self.jacobian_start_nodes.append(node.inputs[0])
            expand_jacobian_node(self, node)
    if self.jacobian_start_nodes:
        # Disable unstable options
        self.bound_opts.update({
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False,
            'sparse_features_alpha': False,
            'sparse_spec_alpha': False,
        })
        for node in self.nodes():
            if isinstance(node, BoundRelu):
                node.use_sparse_spec_alpha = node.use_sparse_features_alpha = False


def expand_jacobian_node(self, jacobian_node):
    logger.info(f'Expanding Jacobian node {jacobian_node}')

    output_node = jacobian_node.inputs[0]
    input_node = jacobian_node.inputs[1]
    batch_size = output_node.output_shape[0]
    output_dim = prod(output_node.output_shape[1:])

    # Gradient values in `grad` may not be accurate. We do not consider gradient
    # accumulation from multiple succeeding nodes. We only want the shapes but
    # not the accurate values.
    grad = {}
    # Dummy values in grad_start
    grad_start = torch.ones(batch_size, output_dim,
                            *output_node.output_shape[1:], device=self.device)
    grad[output_node.name] = grad_start
    input_node_found = False

    # First BFS pass: traverse the graph, count degrees, and build gradient
    # layers.
    # Degrees of nodes.
    degree = {}
    # Original layer for gradient computation.
    node_grad_ori = {}

    degree[output_node.name] = 0
    queue = deque([output_node])
    while len(queue) > 0:
        node = queue.popleft()

        if node == input_node:
            input_node_found = True
            continue
        elif node.no_jacobian:
            continue
        else:
            node_grad_ori[node.name] = node.build_gradient_node(grad[node.name])
            node_grad_ori[node.name] += [None] * (
                len(node.inputs) - len(node_grad_ori[node.name]))

        logger.debug(f'Building gradient node for {node}')
        if not isinstance(node, BoundInput):
            for i in range(len(node.inputs)):
                if node_grad_ori[node.name][i] is None:
                    continue
                grad[node.inputs[i].name] = node_grad_ori[
                    node.name][i][0](*node_grad_ori[node.name][i][1])
                if not node.inputs[i].name in degree:
                    degree[node.inputs[i].name] = 0
                    queue.append(node.inputs[i])
                degree[node.inputs[i].name] += 1

    if not input_node_found:
        raise RuntimeError('Input node not found')

    # Second BFS pass: build the backward computational graph
    grad_node = {}
    initial_name = f'/jacobian{output_node.name}{output_node.name}'
    grad_node[output_node.name] = BoundJacobianInit(inputs=[output_node])
    grad_node[output_node.name].name = initial_name
    self.add_nodes([grad_node[output_node.name]])
    queue = deque([output_node])
    while len(queue) > 0:
        node = queue.popleft()

        if node == input_node:
            self.replace_node(jacobian_node, grad_node[node.name])
            continue
        if node.no_jacobian:
            continue

        logger.debug(f'Converting gradient node for {node}')
        for k in range(len(node.inputs)):
            if node_grad_ori[node.name][k] is None:
                continue
            nodes_op, nodes_in, nodes_out, _ = self._convert_nodes(
                node_grad_ori[node.name][k][0],
                tuple(item.detach()
                      for item in node_grad_ori[node.name][k][1]))
            rename_dict = {}
            assert isinstance(nodes_in[0], BoundInput)
            rename_dict[nodes_in[0].name] = grad_node[node.name].name
            for i in range(1, len(nodes_in)):
                # Assume it's a parameter here
                new_name = f'/jacobian{output_node.name}{node.name}/{k}/params{nodes_in[i].name}'
                rename_dict[nodes_in[i].name] = new_name
            for i in range(len(nodes_op)):
                # intermediate nodes
                if not nodes_op[i].name in rename_dict:
                    new_name = f'/jacobian{output_node.name}{node.name}/{k}/tmp{nodes_op[i].name}'
                    rename_dict[nodes_op[i].name] = new_name
            assert len(nodes_out) == 1
            nodes_out = nodes_out[0]
            rename_dict[nodes_out.name] = f'/jacobian{output_node.name}{node.name}/{k}/output'

            self.rename_nodes(nodes_op, nodes_in, rename_dict)
            input_nodes_replace = (
                [self._modules[nodes_in[0].name]] + node_grad_ori[node.name][k][2])
            for i in range(len(input_nodes_replace)):
                for n in nodes_op:
                    for j in range(len(n.inputs)):
                        if n.inputs[j].name == nodes_in[i].name:
                            n.inputs[j] = input_nodes_replace[i]
            self.add_nodes(nodes_op + nodes_in[len(input_nodes_replace):])

            if node.inputs[k].name in grad_node:
                node_cur = grad_node[node.inputs[k].name]
                node_add = BoundAdd(
                    attr=None, inputs=[node_cur, nodes_out],
                    output_index=0, options={})
                node_add.name = f'{nodes_out.name}/add'
                grad_node[node.inputs[k].name] = node_add
                self.add_nodes([node_add])
            else:
                grad_node[node.inputs[k].name] = nodes_out
            degree[node.inputs[k].name] -= 1
            if degree[node.inputs[k].name] == 0:
                queue.append(node.inputs[k])


def compute_jacobian_bounds(self: 'BoundedModule', x, optimize=True,
                            optimize_output_node=None,
                            bound_lower=True, bound_upper=True):
    """Compute jacobian bounds on the pre-augmented graph (new API)."""

    if isinstance(x, torch.Tensor):
        x = (x,)

    if optimize:
        if optimize_output_node is None:
            if len(self.jacobian_start_nodes) == 1:
                optimize_output_node = self.jacobian_start_nodes[0]
            else:
                raise NotImplementedError(
                    'Multiple Jacobian nodes found.'
                    'An output node for optimizable bounds (optimize_output_node) '
                    'must be specified explicitly')
        self.compute_bounds(
            method='CROWN-Optimized',
            C=None, x=x, bound_upper=False,
            final_node_name=optimize_output_node.name)
        intermediate_bounds = {}
        for node in self._modules.values():
            if node.is_lower_bound_current():
                intermediate_bounds[node.name] = (node.lower, node.upper)
    else:
        intermediate_bounds = None
    lb, ub = self.compute_bounds(
        method='CROWN', x=x,
        bound_lower=bound_lower, bound_upper=bound_upper,
        interm_bounds=intermediate_bounds)
    return lb, ub
