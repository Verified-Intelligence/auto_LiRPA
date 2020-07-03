import os
import torch
from collections import OrderedDict
import re
from collections import namedtuple
from torch.onnx import OperatorExportTypes
from packaging import version
from auto_LiRPA.bounded_tensor import BoundedTensor, BoundedParameter
from auto_LiRPA.utils import logger

Node = namedtuple('Node', (
    'name', 'ori_name', 'inputs', 'attr', 'op', 'param', 'bound_node', 'output_index', 'perturbation'))

torch_old = version.parse(torch.__version__) < version.parse("1.2.0")

def replace(name, scope):
    return '/'.join([scope[name], name])

def parse(graph, params):
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
                                'ori_name': '',
                                'op': n.kind(),
                                'inputs': inputs,
                                'attr': attrs,
                                'param': None,  # will assign parameters later
                                'bound_node': None,
                                'output_index': i, 
                                'perturbation': None, }))
            if n.kind() == 'onnx::BatchNormalization': break  # bn layer has some redundant outputs

    # assert len(list(graph.inputs())) == len(params)
    _c = 0
    for i, n in enumerate(graph.inputs()):
        uname = n.uniqueName() if torch_old else n.debugName()

        if uname not in scope.keys():
            scope[uname] = 'unused'
            _c += 1
            continue
            
        # params[i] is a tuple, ("name", Tensor)
        if isinstance(params[i-_c][1], BoundedTensor) or isinstance(params[i-_c][1], BoundedParameter):
            perturbation = params[i-_c][1].ptb
        else:
            perturbation = None
        # print(uname, n.type().sizes(), params[i-_c][0], list(params[i-_c][1].size()))

        if n.type().sizes() != list(params[i-_c][1].size()):
            raise RuntimeError("Input tensor shapes do not much: {} != {}".format(n.type().sizes(), list(params[i][1].size())))
        nodesIO.append(Node(**{'name': replace(uname, scope),
                             'ori_name': params[i-_c][0],
                             'op': 'Parameter',
                             'inputs': [], 
                             'attr': str(n.type()),
                             'param': params[i-_c][1],
                             'bound_node': None,
                             'output_index': None,
                             # Input nodes may have perturbation, if they are wrapped in BoundedTensor or BoundedParameters
                             'perturbation': perturbation, }))

    assert len(list(graph.inputs())) == len(params) + _c

    return nodesOP, nodesIO

def _get_jit_params(module, param_exclude, param_include):
    # TODO: May get some nodes not used in forward()
    state_dict = torch.jit._unique_state_dict(module, keep_vars=True)

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

    params = zip(new_state_dict.keys(), new_state_dict.values())

    return params

def get_graph_params(module, inputs, param_exclude=".*AuxLogits.*", param_include=None):
    params = _get_jit_params(module, param_exclude=param_exclude, param_include=param_include)
    if version.parse(torch.__version__) < version.parse("1.4.0"):
        trace, out = torch.jit.get_trace_graph(module, inputs)
        torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        torch_graph = trace.graph()
    else:
        # _get_trace_graph becomes an internal function in version >= 1.4.0
        trace, out = torch.jit._get_trace_graph(module, inputs)
        # this is not present in older torch
        from torch.onnx.symbolic_helper import _set_opset_version
        if version.parse(torch.__version__) < version.parse("1.5.0"):
            _set_opset_version(11)
        else:
            _set_opset_version(12)
        torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

    if int(os.environ.get('AUTOLIRPA_DEBUG_GRAPH', 0)) > 0:
        print("Graph before ONNX convertion:")
        print(trace)
        print("ONNX graph:")
        print(torch_graph)

    if not isinstance(inputs, tuple):
        inputs = (inputs, )
    # Add a name to all inputs
    inputs = zip(["input_{}".format(i) for i in range(len(inputs))], inputs)
    params = tuple(inputs) + tuple(params)

    nodesOP, nodesIO = parse(torch_graph, params)

    for i in range(len(nodesOP)):
        param_in = OrderedDict()
        for inp in nodesOP[i].inputs:
            for nIO in nodesIO:
                if inp == nIO.name:
                    param_in.update({inp:nIO.param})
        nodesOP[i] = nodesOP[i]._replace(param=param_in)

    return nodesOP, nodesIO
