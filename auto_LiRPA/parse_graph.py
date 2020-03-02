import torch
from torchvision import models
from collections import OrderedDict
import re
from collections import namedtuple
from torch.onnx import OperatorExportTypes
from packaging import version

Node = namedtuple('Node', (
    'name', 'inputs', 'attr', 'op', 'param', 'bound_node', 'output_index'))

torch_old = version.parse(torch.__version__) < version.parse("1.2.0")

def replace(name, scope):
    return '/'.join([scope[name], name])

def parse(graph, params, num_inputs):
    # in what scope is each node used as an input
    scope = {}

    for n in graph.nodes():
        if torch_old:
            inputs = [i.uniqueName() for i in n.inputs()]
        else:
            inputs = [i.debugName() for i in n.inputs()]

        for i in range(0, len(inputs)):
            if not inputs[i] in scope:
                scope[inputs[i]] = n.scopeName()

        outputs = list(n.outputs())
        for out in outputs:
            uname = out.uniqueName() if torch_old else out.debugName()
            scope[uname] = n.scopeName()

    nodesOP = []
    nodesIO = []

    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}

        if torch_old:
            inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        else:
            inputs = [replace(i.debugName(), scope) for i in n.inputs()]

        outputs = list(n.outputs())
        for i, out in enumerate(outputs):
            if torch_old:
                uname = out.uniqueName()
            else:
                uname = out.debugName()  

            nodesOP.append(Node(**{'name': replace(uname, scope),
                                'op': n.kind(),
                                'inputs': inputs,
                                'attr': attrs,
                                'param': None,  # will assign parameters later
                                'bound_node': None,
                                'output_index': i}))

    for i, n in enumerate(graph.inputs()):
        uname = n.uniqueName() if torch_old else n.debugName()

        if uname not in scope.keys():
            scope[uname] = 'unused'
            continue

        nodesIO.append(Node(**{'name': replace(uname, scope),
                             'op': 'Parameter',
                             'inputs': [], 
                             'attr': str(n.type()),
                             'param': params[0] if i >= num_inputs else None,
                             'bound_node': None,
                             'output_index': None}))
        params = params[(i >= num_inputs):]

    return nodesOP, nodesIO

def _get_jit_params(module, param_exclude, param_include):
    state_dict = torch.jit._unique_state_dict(module)

    if param_exclude is not None:
        param_exclude = re.compile(param_exclude)
    if param_include is not None:
        param_include = re.compile(param_include)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if param_exclude is not None and param_exclude.match(k) is not None:
            print('\nremove input element {} from NodesIO\n'.format(k))
            continue
        if param_include is not None and param_include.match(k) is None:
            continue

        if "num_batches_tracked" not in k:
            if "weight" in k or "bias" in k or "running_mean" in k or "running_var" in k:
                new_state_dict[k] = v

    params = list(new_state_dict.values()) #[::-1]

    return params, list(new_state_dict.keys()) #[::-1]

def get_graph_params(module, input, param_exclude=".*AuxLogits.*", param_include=None):
    params, weight_names = _get_jit_params(module, param_exclude=param_exclude, param_include=param_include)
    trace, out = torch.jit.get_trace_graph(module, input)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    torch_graph = trace.graph()

    if not isinstance(input, tuple):
        input = (input, )

    nodesOP, nodesIO = parse(torch_graph, params, len(input))

    for i in range(len(nodesOP)):
        param_in = OrderedDict()
        for inp in nodesOP[i].inputs:
            for nIO in nodesIO:
                if inp == nIO.name:
                    param_in.update({inp:nIO.param})
        nodesOP[i] = nodesOP[i]._replace(param=param_in)

    return nodesOP, nodesIO
