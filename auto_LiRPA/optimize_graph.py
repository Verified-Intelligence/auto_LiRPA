#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Optimize the graph to merge nodes and remove unnecessary ones.

Initial and experimental code only.
"""

from auto_LiRPA.bound_ops import (BoundActivation, BoundMul, BoundSqr, BoundDiv,
                                  BoundPow, BoundReciprocal, BoundBuffers, BoundConstant,
                                  BoundCos, BoundSec, BoundMin, BoundMax, BoundAdd, BoundSub,
                                  BoundRelu)

from auto_LiRPA.utils import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _optimize_graph(self: 'BoundedModule'):
    """Optimize the graph to remove some unnecessary nodes."""
    merge_identical_act(self)
    convert_sqr(self)
    div_to_mul(self)
    merge_sec(self)
    minmax_to_relu(self)

    if self.bound_opts['optimize_graph']['optimizer'] is not None:
        # Use the custom graph optimizer
        self.bound_opts['optimize_graph']['optimizer'](self)

    for node in list(self.nodes()):
        if (not node.output_name
                and node.name != self.final_name
                and node.name not in self.root_names):
            self.delete_node(node)


def _copy_node_properties(new, ref):
    new.output_shape = ref.output_shape
    new.device = ref.device
    new.attr['device'] = ref.attr['device']
    new.batch_dim = ref.batch_dim
    new.from_complex_node = ref.from_complex_node


def merge_sec(model: 'BoundedModule'):
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundReciprocal and type(node.inputs[0]) == BoundCos:
            node_new = BoundSec(inputs=[node.inputs[0].inputs[0]])
            node_new.name = f'{node.inputs[0].name}/sec'
            _copy_node_properties(node_new, node)
            if node_new.name in model._modules:
                node_existing = model._modules[node_new.name]
                assert isinstance(node_existing, BoundSec)
                assert node_existing.inputs[0] == node.inputs[0].inputs[0]
                model.replace_node(node, node_existing)
            else:
                model.add_nodes([node_new])
                model.replace_node(node, node_new)


def div_to_mul(model: 'BoundedModule'):
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundDiv:
            logger.debug('Replacing BoundDiv node: %s', node)
            node_reciprocal = BoundReciprocal(inputs=[node.inputs[1]])
            node_reciprocal.name = f'{node.name}/reciprocal'
            _copy_node_properties(node_reciprocal, node)
            model.add_nodes([node_reciprocal])
            node_mul = BoundMul(inputs=[node.inputs[0], node_reciprocal],
                                options=model.bound_opts)
            node_mul.name = f'{node.name}/mul'
            _copy_node_properties(node_mul, node)
            model.add_nodes([node_mul])
            model.replace_node(node, node_mul)


def convert_sqr(model: 'BoundedModule'):
    """Replace BoundMul or Bound Pow with BoundSqr if applicable.

    1. If the two inputs nodes of a BoundMul node are the same, use BoundSqr.
    2. Pow(x, 2) can be replaced with BoundSqr.
    """
    nodes = list(model.nodes())
    for node in nodes:
        replace = False
        if type(node) == BoundMul and node.inputs[0] == node.inputs[1]:
            replace = True
        elif type(node) == BoundPow:
            if ((isinstance(node.inputs[1], BoundBuffers) and node.inputs[1].buffer == 2) or
                (isinstance(node.inputs[1], BoundConstant) and node.inputs[1].value == 2)):
                replace = True
        if replace:
            node_new = BoundSqr(inputs=[node.inputs[0]])
            node_new.name = f'{node.name}/sqr'
            _copy_node_properties(node_new, node)
            model.add_nodes([node_new])
            logger.debug('Replaceing %s with %s', node, node_new)
            model.replace_node(node, node_new)


def merge_identical_act(model: 'BoundedModule'):
    """Merge identical BoundActivation"""
    nodes = list(model.nodes())
    merged = [False] * len(nodes)
    for i in range(len(nodes)):
        if (not merged[i]
                and isinstance(nodes[i], BoundActivation)
                and len(nodes[i].inputs) == 1):
            for j in range(i + 1, len(nodes)):
                if (not merged[j]
                        and type(nodes[j]) == type(nodes[i])
                        and len(nodes[i].inputs) == 1):
                    if nodes[i].inputs[0] == nodes[j].inputs[0]:
                        logger.debug('Merging node %s to %s', nodes[j], nodes[i])
                        model.replace_node(nodes[j], nodes[i])
                        merged[j] = True


def minmax_to_relu(model: 'BoundedModule'):
    """Replace BoundMinMax with BoundRelu if one of its inputs is constant"""
    nodes = list(model.nodes())
    for node in nodes:
        if type(node) == BoundMax:
            for i, input_node in enumerate(node.inputs):
                if type(input_node) == BoundConstant:
                    logger.debug('Replacing BoundMax node %s', node)
                    # max(x, c) = ReLU(x - c) + c
                    node_sub = BoundSub(inputs=[node.inputs[1-i], input_node],
                                        options=model.bound_opts)
                    node_sub.name = f'{node.name}/sub'
                    _copy_node_properties(node_sub, node)
                    node_relu = BoundRelu(inputs=[node_sub],
                                          options=model.bound_opts)
                    node_relu.name = f'{node.name}/relu'
                    _copy_node_properties(node_relu, node)
                    node_add = BoundAdd(inputs=[node_relu, input_node],
                                        options=model.bound_opts)
                    node_add.name = f'{node.name}/add'
                    _copy_node_properties(node_add, node)
                    model.add_nodes([node_sub, node_relu, node_add])
                    model.replace_node(node, node_add)
                    break
        elif type(node) == BoundMin:
            for i, input_node in enumerate(node.inputs):
                if type(input_node) == BoundConstant:
                    logger.debug('Replacing BoundMin node %s', node)
                    # min(x, c) = -ReLU(c - x) + c
                    node_sub_1 = BoundSub(inputs=[input_node, node.inputs[1-i]],
                                          options=model.bound_opts)
                    node_sub_1.name = f'{node.name}/sub/1'
                    _copy_node_properties(node_sub_1, node)
                    node_relu = BoundRelu(inputs=[node_sub_1],
                                          options=model.bound_opts)
                    node_relu.name = f'{node.name}/relu'
                    _copy_node_properties(node_relu, node)
                    node_sub_2 = BoundSub(inputs=[input_node, node_relu],
                                          options=model.bound_opts)
                    node_sub_2.name = f'{node.name}/sub/2'
                    _copy_node_properties(node_sub_2, node)
                    model.add_nodes([node_sub_1, node_relu, node_sub_2])
                    model.replace_node(node, node_sub_2)
                    break