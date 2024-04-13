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
import torch
from torch.onnx.utils import _optimize_graph
from collections import OrderedDict
from collections import namedtuple
from packaging import version
import re
import os
from .bounded_tensor import BoundedTensor, BoundedParameter
from .utils import logger, unpack_inputs

Node = namedtuple('Node', (
    'name', 'ori_name', 'inputs', 'attr', 'op', 'param', 'input_index',
    'bound_node', 'output_index', 'perturbation'), defaults=(None,) * 10)

def get_node_name(node):
    return node.debugName()

def get_node_attribute(node, attribute_name):
    if hasattr(torch.onnx.symbolic_helper, '_node_get'):
        # Pytorch >= 1.13.
        return torch.onnx.symbolic_helper._node_get(node, attribute_name)
    else:
        # Pytorch <= 1.12. This will call _node_getitem in torch.onnx.utils.
        return node[attribute_name]

def parse_graph(graph, inputs, params):
    input_all = []
    input_used = []
    scope = {}
    for n in graph.inputs():
        input_all.append(n.debugName())
    for n in graph.nodes():
        n_inputs = [get_node_name(i) for i in n.inputs()]
        for inp in n.inputs():
            input_used.append(inp.debugName())
        for out in n.outputs():
            scope[get_node_name(out)] = n.scopeName()
    for node in graph.inputs():
        name = get_node_name(node)
        scope[name] = ''
    for n in graph.outputs():
        name = get_node_name(n)
        if name in input_all:
            # This output node directly comes from an input node with an Op
            input_used.append(n.debugName())

    def name_with_scope(node):
        name = get_node_name(node)
        return '/'.join([scope[name], name])

    nodesOP = []
    for n in graph.nodes():
        attrs = {k: get_node_attribute(n, k) for k in n.attributeNames()}
        n_inputs = [name_with_scope(i) for i in n.inputs()]
        for i, out in enumerate(list(n.outputs())):
            nodesOP.append(Node(**{'name': name_with_scope(out),
                                'op': n.kind(),
                                'inputs': n_inputs,
                                'attr': attrs,
                                'output_index': i,
                                }))

    # filter out input nodes in `graph.inputs()` that are actually used
    nodesIn = []
    used_by_index = []
    for i, n in enumerate(graph.inputs()):
        name = get_node_name(n)
        used = name in input_used
        used_by_index.append(used)
        if used:
            nodesIn.append(n)
    # filter out input nodes in `inputs` that are actually used
    inputs_unpacked = unpack_inputs(inputs)
    assert len(list(graph.inputs())) == len(inputs_unpacked) + len(params)
    inputs = [inputs_unpacked[i] for i in range(len(inputs_unpacked)) if used_by_index[i]]
    # index of the used inputs among all the inputs
    input_index = [i for i in range(len(inputs_unpacked)) if used_by_index[i]]
    # Add a name to all inputs
    inputs = list(zip(["input_{}".format(input_index[i]) for i in range(len(inputs))], inputs))
    # filter out params that are actually used
    params = [params[i] for i in range(len(params)) if used_by_index[i + len(inputs_unpacked)]]
    inputs_and_params = inputs + params
    assert len(nodesIn) == len(inputs_and_params)

    # output nodes of the module
    nodesOut = []
    for n in graph.outputs():
        # we only record names
        nodesOut.append(name_with_scope(n))

    for i, n in enumerate(nodesIn):
        if (isinstance(inputs_and_params[i][1], BoundedTensor) or
                isinstance(inputs_and_params[i][1], BoundedParameter)):
            perturbation = inputs_and_params[i][1].ptb
        else:
            perturbation = None
        if i > 0 and n.type().sizes() != list(inputs_and_params[i][1].size()):
            raise RuntimeError("Input tensor shapes do not much: {} != {}".format(
                n.type().sizes(), list(inputs_and_params[i][1].size())))
        nodesIn[i] = Node(**{'name': name_with_scope(n),
                             'ori_name': inputs_and_params[i][0],
                             'op': 'Parameter',
                             'inputs': [],
                             'attr': str(n.type()),
                             'param': inputs_and_params[i][1] if i >= len(inputs) else None,
                             # index among all the inputs including unused ones
                             'input_index': input_index[i] if i < len(inputs) else None,
                             # Input nodes may have perturbation, if they are wrapped in BoundedTensor or BoundedParameters
                             'perturbation': perturbation, })

    return nodesOP, nodesIn, nodesOut

def _get_jit_params(module, param_exclude, param_include):
    state_dict = torch.jit._unique_state_dict(module, keep_vars=True)

    if param_exclude is not None:
        param_exclude = re.compile(param_exclude)
    if param_include is not None:
        param_include = re.compile(param_include)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if param_exclude is not None and param_exclude.match(k) is not None:
            print(f'\nremove input element {k} from nodesIn\n')
            continue
        if param_include is not None and param_include.match(k) is None:
            continue
        new_state_dict[k] = v

    params = zip(new_state_dict.keys(), new_state_dict.values())

    return params

def get_output_template(out):
    """Construct a template for the module output with `None` representing places
    to be filled with tensor results"""
    if isinstance(out, torch.Tensor):
        return None
    elif isinstance(out, list):
        return list([get_output_template(o) for o in out])
    elif isinstance(out, tuple):
        return tuple([get_output_template(o) for o in out])
    elif isinstance(out, dict):
        template = {}
        for key in out:
            template[key] = get_output_template(out[key])
        return template
    else:
        raise NotImplementedError

def parse_source(node):
    kind = node.kind()
    if hasattr(node, 'sourceRange'):
        source_range_str = node.sourceRange()
        # divide source_range_str by '\n' and drop any lines containing 'torch.nn'
        source_range_str = '\n'.join([line for line in source_range_str.split('\n') if 'torch/nn' not in line])
        match = re.match(r'([^ ]+\.py)\((\d+)\)', source_range_str)
        if match:
            # match.group(1) is the file name
            # match.group(2) is the line number
            return f"{kind}_{os.path.basename(match.group(1)).split('.')[0]}_{match.group(2)}"
    return kind

def update_debug_names(trace_graph):
    visited = []
    for n in trace_graph.nodes():
        for input in n.inputs():
            if input.debugName() not in visited:
                input.setDebugName(f"{input.debugName()}_{parse_source(n)}")
                visited.append(input.debugName())
        for output in n.outputs():
            if output.debugName() not in visited:
                output.setDebugName(f"{output.debugName()}_{parse_source(n)}")
                visited.append(output.debugName())

def parse_module(module, inputs, param_exclude=".*AuxLogits.*", param_include=None):
    params = _get_jit_params(module, param_exclude=param_exclude, param_include=param_include)
    trace, out = torch.jit._get_trace_graph(module, inputs)
    if version.parse(torch.__version__) < version.parse("2.0.0"):
        from torch.onnx.symbolic_helper import _set_opset_version
        _set_opset_version(12)

    logger.debug("Graph before ONNX convertion:")
    logger.debug(trace)

    # Assuming that the first node in the graph is the primary input node.
    # It must have a batch dimension.
    primary_input = get_node_name(next(iter(trace.inputs())))
    trace_graph = _optimize_graph(
        trace, torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        params_dict={},
        input_names=[primary_input],
        dynamic_axes={primary_input: {0: 'batch'}})
    logger.debug('trace_graph: %s', trace_graph)

    if os.environ.get('AUTOLIRPA_DEBUG_NAMES', 0):
        update_debug_names(trace_graph)

    logger.debug("ONNX graph:")
    logger.debug(trace_graph)

    if not isinstance(inputs, tuple):
        inputs = (inputs, )

    nodesOP, nodesIn, nodesOut = parse_graph(trace_graph, tuple(inputs), tuple(params))

    for i in range(len(nodesOP)):
        param_in = OrderedDict()
        for inp in nodesOP[i].inputs:
            for n in nodesIn:
                if inp == n.name:
                    param_in.update({inp:n.param})
        nodesOP[i] = nodesOP[i]._replace(param=param_in)

    template = get_output_template(out)

    return nodesOP, nodesIn, nodesOut, template
