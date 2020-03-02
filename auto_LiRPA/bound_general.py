import pdb
from auto_LiRPA.bound_ops import *
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_op_map import bound_op_map
from auto_LiRPA.parse_graph import get_graph_params
from auto_LiRPA.utils import logger
from collections import OrderedDict

class BoundGeneral(nn.Module):
    def __init__(self, model, global_input, verbose=False):
        super(BoundGeneral, self).__init__()
        if isinstance(model, BoundGeneral):
            for key in model.__dict__.keys():
                setattr(self, key, getattr(model, key))
            return
        self.verbose = verbose
        self._convert(model, global_input)
        self.choices = nn.ModuleList([_l for _l in self.node_dict.values() if not isinstance(_l, BoundParams)])

    def train(self, mode=True):
        for node in self.nodes:
            node.train(mode=mode)

    def eval(self):
        for node in self.nodes:
            node.eval()     

    def forward(self, *x):
        self._clear(*x)

        degree_in = {}
        queue = []
        for l in self.nodes:
            degree_in[l.name] = len(l.input_name)
            if degree_in[l.name] == 0:
                queue.append(l)
        forward_values = {}

        final_output = None
        while len(queue) > 0:
            l = queue[0]
            queue = queue[1:]

            inp = [forward_values[l_pre] for l_pre in l.input_name]
            for l_pre in l.input_name:
                l.from_input = l.from_input or self.node_dict[l_pre].from_input
            l.forward_value = l.forward(*inp)

            forward_values[l.name] = l.forward_value

            if l.name == self.final_name:
                final_output = l.forward_value
            for l_next in l.output_name:
                degree_in[l_next] -= 1
                if degree_in[l_next] == 0:  # all inputs of this node have already set
                    queue.append(self.node_dict[l_next])

        # assumption: there is only one global output
        return final_output

    def compute_bounds(self, x=None, C=None, ptb=None, IBP=False, forward=False, method="backward"):
        if C is None:
            # C is an identity matrix by default 
            output = self.node_dict[self.final_name].forward_value
            dim_output = output.reshape(output.shape[0], -1).shape[1]
            C = torch.eye(dim_output).to(self.device).unsqueeze(0).repeat(output.shape[0], 1, 1)

        root = [self.node_dict[name] for name in self.root_name]

        for i in range(1, len(root)):
            root[i].lower = root[i].upper = root[i].forward_value
            root[i].interval = (root[i].lower, root[i].upper)

        if IBP:
            root[0].interval = ptb.init_interval(x)
            lower, upper = self._IBP_general(node=self.node_dict[self.final_name], C=C)
        if method is None:
            return lower, upper

        root[0].linear, x = ptb.init_linear(x)
        self._concretize_linear(root[0], ptb, x)
        for i in range(1, len(root)):
            root[i].linear = root[i].lower = root[i].upper = root[i].forward_value
        dim_in = root[0].linear.lw.shape[1]

        for i in range(len(self.nodes)):
            if hasattr(self.nodes[i], 'nonlinear') and self.nodes[i].nonlinear:
                for l_name in self.nodes[i].input_name:
                    node = self.node_dict[l_name]
                    if not hasattr(node, 'lower'):
                        if not node.from_input:
                            # fixed and does not have the batch size dimension
                            node.lower = node.upper = node.forward_value
                            node.linear = self._linear_from_constant(node, dim_in)
                            continue

                        if forward:
                            l, u = self._forward_general(
                                x=x, dim_in=dim_in,
                                ptb=ptb, node=node, concretize=True)
                        else:
                            dim = node.forward_value[0].reshape(-1).shape[0]
                            newC = torch.eye(dim, device=node.forward_value.device)\
                                .unsqueeze(0).repeat(node.forward_value.shape[0], 1, 1)\
                                .view(node.forward_value.shape[0], dim, *node.forward_value.shape[1:])
                            l, u = self._backward_general(x=x, C=newC, ptb=ptb, node=node, root=root) 

        final = self.node_dict[self.final_name]

        if method == "backward":
            return self._backward_general(x=x, C=C, ptb=ptb, node=final, root=root)
        elif method == "forward":
            # TODO: `C` is not supported yet (identity matrix by default)
            l, u = self._forward_general(
                x=x,
                dim_in=root[0].linear.lw.shape[1], ptb=ptb, node=final, concretize=True)
            mask = torch.gt(C, 0.).to(torch.float)
            _l = (torch.bmm(mask * C, l.unsqueeze(-1)) + \
                  torch.bmm((1 - mask) * C, u.unsqueeze(-1))).squeeze(-1)
            _u = (torch.bmm(mask * C, u.unsqueeze(-1)) + \
                  torch.bmm((1 - mask) * C, l.unsqueeze(-1))).squeeze(-1)
            return _l, _u
        else:
            raise NotImplementedError

    def compute_worst_logits(self, ptb=None, x=None, y=None, IBP=False, forward=False, method="backward"):
        logits_l, logits_u = self.compute_bounds(ptb=ptb, x=x, IBP=IBP, forward=forward, method=method)
        num_classes = self.node_dict[self.final_name].forward_value.shape[-1]
        one_hot = F.one_hot(y, num_classes=num_classes).to(torch.float32).to(self.device)
        return logits_l * one_hot + logits_u * (1. - one_hot)

    def _clear(self, *x):
        for l in self.nodes:
            l.from_input = False
            try:
                del(l.forward_value)
            except:
                pass
            try:
                for item in l.linear: del(item)
                delattr(l, 'linear')
            except:
                pass
            try:
                delattr(l, 'lower')
                delattr(l, 'upper')
            except:
                pass
            try:
                delattr(l, 'interval')
            except:
                pass
            for i in range(len(self.root_name)):
                if self.root_name[i] == l.name:
                    l.value = x[i]
                    l.from_input = True   

    def _get_node_input(self, nodesOP, nodesIO, node):
        ret = []
        for i in range(len(node.inputs)):
            found = False
            for op in nodesOP:
                if op.name == node.inputs[i]:
                    ret.append(op.bound_node)
                    break
            if len(ret) == i + 1:
                continue
            for io in nodesIO:
                if io.name == node.inputs[i]:
                    ret.append(io.param)
                    break
            if len(ret) <= i:
                raise ValueError('cannot find inputs of node: {}'.format(node.name))
        return ret

    def _convert_nodes(self, model, global_input):
        global_input_cpu = tuple([i.to("cpu") for i in list(global_input)])
        model.train()
        model.to('cpu')
        nodesOP, nodesIO = get_graph_params(model, global_input_cpu)
        model.to(self.device)
        for i in range(0, len(nodesIO)):
            if nodesIO[i].param is not None:
                nodesIO[i] = nodesIO[i]._replace(param=nodesIO[i].param.to(self.device))

        for n in range(len(nodesOP)):
            attr = nodesOP[n].attr
            inputs = self._get_node_input(nodesOP, nodesIO, nodesOP[n])

            if nodesOP[n].op in bound_op_map:
                if nodesOP[n].op == 'onnx::BatchNormalization':
                    # BatchNormalization node needs model.training flag to set running mean and vars
                    nodesOP[n] = nodesOP[n]._replace(
                        bound_node=bound_op_map[nodesOP[n].op](
                            nodesOP[n].inputs, nodesOP[n].name, attr,
                            inputs, nodesOP[n].output_index,
                            self.device,
                            model.training))
                else:
                    nodesOP[n] = nodesOP[n]._replace(
                        bound_node=bound_op_map[nodesOP[n].op](
                            nodesOP[n].inputs, nodesOP[n].name, attr,
                            inputs, nodesOP[n].output_index, self.device))
            else:
                print(nodesOP[n])
                raise NotImplementedError('Unsupported operation {}'.format(nodesOP[n].op))

            if self.verbose:
                logger.debug('Convert complete for {} with operation: {}'.format(nodesOP[n].name, nodesOP[n].op))

        for i in range(0, len(global_input)):
            nodesIO[i] = nodesIO[i]._replace(param=global_input[i], bound_node=BoundInput(
                nodesIO[i].inputs, nodesIO[i].name, value=global_input[i]))
            nodesIO[i].bound_node.method = 'forward'
        for i in range(len(global_input), len(nodesIO)):
            nodesIO[i] = nodesIO[i]._replace(bound_node=BoundParams(
                nodesIO[i].inputs, nodesIO[i].name,
                value=nodesIO[i].param
            ))
            nodesIO[i].bound_node.method = 'forward'

        return nodesOP, nodesIO

    def _build_graph(self, nodesOP, nodesIO):
        self.nodes = []
        for node in nodesOP + nodesIO:
            assert (node.bound_node is not None)
            self.nodes.append(node.bound_node)
        self.final_name = nodesOP[-1].name
        self.root_name = [node.name for node in nodesIO[:self.num_global_inputs]]
        self.node_dict = {}
        for l in self.nodes:
            self.node_dict[l.name] = l
            l.output_name = []
            if isinstance(l.input_name, str):
                l.input_name = [l.input_name]
        for l in self.nodes:
            for l_pre in l.input_name:
                self.node_dict[l_pre].output_name.append(l.name)

    def _split_complex(self, nodesOP, nodesIO):
        found_complex = False
        for n in range(len(nodesOP)):
            if hasattr(nodesOP[n].bound_node, 'complex') and \
                    nodesOP[n].bound_node.complex:
                found_complex = True
                _nodesOP, _nodesIO = self._convert_nodes(
                    nodesOP[n].bound_node.model, nodesOP[n].bound_node.input)
                name_base = nodesOP[n].name + "/split/"
                rename_dict = {}
                for node in _nodesOP + _nodesIO:
                    rename_dict[node.name] = name_base + node.name

                num_inputs = len(nodesOP[n].bound_node.input)

                # assuming each supported complex operation only has one output
                for i in range(num_inputs):
                    rename_dict[_nodesIO[i].name] = nodesOP[n].inputs[i]
                rename_dict[_nodesOP[-1].name] = nodesOP[n].name

                def rename(node):
                    node = node._replace(name=rename_dict[node.name])
                    node = node._replace(inputs=[rename_dict[name] for name in node.inputs])
                    node.bound_node.name = rename_dict[node.bound_node.name]
                    node.bound_node.input_name = [
                        rename_dict[name] for name in node.bound_node.input_name]
                    return node

                for i in range(len(_nodesOP)):
                    _nodesOP[i] = rename(_nodesOP[i])
                for i in range(len(_nodesIO)):
                    _nodesIO[i] = rename(_nodesIO[i])

                nodesOP = nodesOP[:n] + _nodesOP + nodesOP[(n + 1):]
                nodesIO = nodesIO + _nodesIO[num_inputs:]

                break

        return nodesOP, nodesIO, found_complex

    # convert a Pytorch model to a model with bounds
    def _convert(self, model, global_input):
        if self.verbose:
            logger.info('Converting the model...')

        if not isinstance(global_input, tuple):
            global_input = (global_input,)
        self.num_global_inputs = len(global_input)
        self.device = global_input[0].device

        nodesOP, nodesIO = self._convert_nodes(model, global_input)

        while True:
            self._build_graph(nodesOP, nodesIO)
            self.forward(*global_input)
            nodesOP, nodesIO, found_complex = self._split_complex(nodesOP, nodesIO)
            if not found_complex: break

        for node in self.nodes:
            for p in list(node.named_parameters()):
                self.register_parameter(
                    '{}/{}'.format(node.name, p[0]), p[1])

        logger.debug('NodesOP:')
        for node in nodesOP:
            logger.debug('{}'.format(node._replace(param=None)))
        logger.debug('NodesIO')
        for node in nodesIO:
            logger.debug('{}'.format(node._replace(param=None)))

        if self.verbose:
            logger.info('Model converted to support bounds')

    def _linear_from_constant(self, node, dim_in):
        if node.from_input:
            raise NotImplementedError
        else:
            lw = uw = torch.zeros(dim_in, *node.forward_value.shape).to(node.forward_value.device)
            lb = ub = node.forward_value
            return LinearBound(lw, lb, uw, ub, lb, ub)

    def _concretize_linear(self, node, ptb, x):
        if not isinstance(node.linear, LinearBound) or not node.from_input:
            lower = upper = node.forward_value
        else:
            linear = node.linear
            if len(linear.lw.shape) == 1:
                lA = linear.lw.reshape(1, 1, linear.lw.shape[0])
                uA = linear.uw.reshape(1, 1, linear.lw.shape[0])
            else:
                lA = linear.lw.reshape(
                    linear.lw.shape[0], linear.lw.shape[1], -1).transpose(1, 2)
                uA = linear.uw.reshape(
                    linear.uw.shape[0], linear.uw.shape[1], -1).transpose(1, 2)
            lower = ptb.concretize(x, lA, linear.lb, sign=-1).view(node.forward_value.shape)
            upper = ptb.concretize(x, uA, linear.ub, sign=+1).view(node.forward_value.shape)
        node.linear = node.linear._replace(lower=lower, upper=upper)
        node.lower, node.upper = lower, upper
        return lower, upper

    def _forward_general(self, node=None, ptb=None, x=None, dim_in=None, concretize=False):
        if hasattr(node, 'lower'):
            return node.lower, node.upper
        if hasattr(node, 'linear'):
            if concretize:
                return self._concretize_linear(node, ptb, x)
            return

        for l_pre in node.input_name:
            l = self.node_dict[l_pre]
            if not hasattr(l, 'linear'):
                self._forward_general(x=x, dim_in=dim_in, ptb=ptb, node=l)

        inp = [self.node_dict[l_pre].linear for l_pre in node.input_name]
        node.linear = node.bound_forward(dim_in, *inp)

        if concretize:
            if isinstance(node.linear, LinearBound):
                return self._concretize_linear(node, ptb, x)
            else:
                node.lower = node.upper = node.forward_value
                return node.lower, node.upper

    def _IBP_general(self, node=None, C=None):
        for l_pre in node.input_name:
            l = self.node_dict[l_pre]
            if not hasattr(l, 'interval'):
                self._IBP_general(l)

        inp = [self.node_dict[l_pre].interval for l_pre in node.input_name]
        if C is not None:
            # merge the output node with the specification
            assert (isinstance(node, BoundLinear))
            node.interval = node.interval_propagate(*inp, C=C)
        else:
            node.interval = node.interval_propagate(*inp)
        node.lower, node.upper = node.interval

        return node.interval

    def _backward_general(self, norm=np.inf, x=None, C=None, ptb=None, node=None, root=None):
        logger.debug('Backward from {} {}'.format(node.name, node))

        degree_out = {}
        for l in self.nodes:
            l.bounded = True
            l.lA = l.uA = None
            degree_out[l.name] = 0
        queue = [node]
        while len(queue) > 0:
            l = queue[0]
            queue = queue[1:]
            for l_pre in l.input_name:
                degree_out[l_pre] += 1
                if self.node_dict[l_pre].bounded:
                    self.node_dict[l_pre].bounded = False
                    queue.append(self.node_dict[l_pre])
        node.bounded = True
        node.lA = node.uA = C
        lb = ub = torch.tensor(0.).to(C.device)

        queue = [node]
        while len(queue) > 0:
            l = queue[0]  # backward from l
            queue = queue[1:]
            l.bounded = True

            if l.name in self.root_name or l == root: continue

            for l_pre in l.input_name:
                _l = self.node_dict[l_pre]
                degree_out[l_pre] -= 1
                if degree_out[l_pre] == 0:
                    queue.append(_l)

            if l.uA is not None:
                def add_bound(node, lA, uA):
                    node.lA = lA if node.lA is None else (node.lA + lA)
                    node.uA = uA if node.uA is None else (node.uA + uA)

                logger.debug('Backward at {} {}'.format(l.name, l))

                input_nodes = [self.node_dict[l_name] for l_name in l.input_name]

                if len(l.input_name) == 1:
                    lA, lower_b, uA, upper_b = l.bound_backward(l.lA, l.uA, *input_nodes)
                    A = [(lA, uA)]
                else:
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *input_nodes)
                ub = ub + upper_b
                lb = lb + lower_b

                for i, l_pre in enumerate(l.input_name):
                    _l = self.node_dict[l_pre]
                    add_bound(_l, lA=A[i][0], uA=A[i][1])

        batch_size = C.shape[0]
        output_shape = node.forward_value.shape[1:]
        if node.forward_value.contiguous().view(batch_size, -1).shape[1] != C.shape[1]:
            output_shape = [-1]

        for r in root:
            if r.lA is None: continue
            if isinstance(r.linear, LinearBound):
                uA = r.uA.reshape(batch_size, r.uA.shape[1], -1).matmul(
                    r.linear.uw.view(batch_size, r.linear.uw.shape[1], -1).transpose(1, 2))
                ub = ub + r.uA.reshape(batch_size, r.uA.shape[1], -1).matmul(
                    r.linear.ub.view(batch_size, -1, 1)).squeeze(-1)

                lA = r.lA.reshape(batch_size, r.lA.shape[1], -1).matmul(
                    r.linear.lw.view(batch_size, r.linear.lw.shape[1], -1).transpose(1, 2))
                lb = lb + r.lA.reshape(batch_size, r.lA.shape[1], -1).matmul(
                    r.linear.lb.view(batch_size, -1, 1)).squeeze(-1)

                lb = lb + ptb.concretize(x, lA, torch.zeros_like(lb), sign=-1)
                ub = ub + ptb.concretize(x, uA, torch.zeros_like(ub), sign=+1)
            else:
                lb = lb + r.lA.reshape(batch_size, r.lA.shape[1], -1).matmul(
                    r.forward_value.view(batch_size, -1, 1)).squeeze(-1)
                ub = ub + r.uA.reshape(batch_size, r.uA.shape[1], -1).matmul(
                    r.forward_value.view(batch_size, -1, 1)).squeeze(-1)

        node.lower = lb.view(batch_size, *output_shape)
        node.upper = ub.view(batch_size, *output_shape)
        return node.lower, node.upper

    # TODO merge this function with `compute_bounds`
    def weights_full_backward_range(self, norm=np.inf, x=None, eps=None, C=None,
                            ptb=None, IBP=False, w_eps=None):
        root = [self.node_dict[name] for name in self.root_name]

        # CROWN propagation for all rest nodes
        # outer loop, starting from the 2nd node until we reach the output node
        # assign the weights concretized bounds: w +- eps
        _tmp_counter = 0
        for i in range(len(self.nodes)):
            if hasattr(self.nodes[i], 'weight'):
                self.nodes[i].weight_upper = self.nodes[i].weight + w_eps[_tmp_counter]
                self.nodes[i].weight_lower = self.nodes[i].weight - w_eps[_tmp_counter]
                self.nodes[i].weight.eps = w_eps[_tmp_counter]
                _tmp_counter += 1
        root[0].upper = root[0].value + eps
        root[0].lower = root[0].value - eps
        root[0].eps = eps

        if IBP:
            raise NotImplementedError
        else:

            for i in range(1, len(root)):
                root[i].linear = root[i].forward_value

            for i in range(len(self.nodes)):
                if not isinstance(self.nodes[i], (BoundParams, BoundInput,)):
                    for l_name in self.nodes[i].input_name:
                        node = self.node_dict[l_name]
                        if not hasattr(node, 'lower'):
                            # if node.method == "backward":
                            if len(node.forward_value.shape) > 0 and node.forward_value.shape[0] == x.shape[0]:
                                dim = node.forward_value[0].reshape(-1).shape[0]
                            else:
                                dim = node.forward_value.reshape(-1).shape[0]

                            newC = torch.eye(dim, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)

                            if len(node.forward_value.shape) > 0 and \
                                    node.forward_value.shape[0] == x.shape[0]:  # this is not accurate
                                newC = newC.view(x.shape[0], dim, *node.forward_value.shape[1:])
                            else:
                                newC = newC.view(x.shape[0], dim, *node.forward_value.shape)
                            l, u = self.weights_backward_general(x=x, C=newC, ptb=ptb, node=node, root=root)
                            node.lower, node.upper = l, u

        final = self.node_dict[self.final_name]

        # get the final node bound with spec C
        return self.weights_backward_general(x=x, C=C, ptb=ptb, node=final, root=root)

    # TODO merge this with `_backward_general`
    def weights_backward_general(self, norm=np.inf, x=None, eps=None, C=None, 
                                ptb=None, node=None, root=None):
        assert (len(root) == 1)
        root = root[0]

        torch.cuda.empty_cache()

        logger.debug('Backward from {} {}'.format(node.name, node))

        degree_out = {}
        for l in self.nodes:
            l.bounded = True
            l.lA = l.uA = None
            degree_out[l.name] = 0
        queue = [node]
        while len(queue) > 0:
            l = queue[0]
            queue = queue[1:]
            for l_pre in l.input_name:
                degree_out[l_pre] += 1
                if self.node_dict[l_pre].bounded:
                    self.node_dict[l_pre].bounded = False
                    queue.append(self.node_dict[l_pre])
        node.bounded = True
        node.uA = C
        node.lA = C
        upper_sum_b = lower_sum_b = torch.tensor(0.).to(C.device)

        queue = [node]
        nodes_perturb_list = []
        while len(queue) > 0:
            l = queue[0]
            queue = queue[1:]
            l.bounded = True

            if l in self.root_name or l == root: continue

            for l_pre in l.input_name:
                _l = self.node_dict[l_pre]
                degree_out[l_pre] -= 1
                if degree_out[l_pre] == 0:
                    queue.append(_l)

            if l.uA is not None:
                def add_bound(node, uA, lA):
                    node.uA = uA if node.uA is None else (node.uA + uA)
                    node.lA = lA if node.lA is None else (node.lA + lA)

                logger.debug('Backward at {} {}'.format(l.name, l))

                if len(l.input_name) == 1:
                    input_node = self.node_dict[l.input_name[0]]
                    if hasattr(l, 'nonlinear') and l.nonlinear is True:
                        lA, lower_b, uA, upper_b = l.bound_backward(l.lA, l.uA, input_node)
                        A = [(uA, lA)]
                    else:
                        [(lA_x, uA_x), (lA_y, uA_y)], upper_b, lower_b = l.two_bounds_backward(l.lA, l.uA, input_node, l)
                        A = [(lA_x, uA_x)] # y is weights, x is input
                        l.weight.lA_y, l.weight.uA_y = lA_y, uA_y
                        nodes_perturb_list.append(l.weight)
                else:
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA)
                upper_sum_b = upper_sum_b + upper_b
                lower_sum_b = lower_sum_b + lower_b

                for i, l_pre in enumerate(l.input_name):
                    _l = self.node_dict[l_pre]
                    add_bound(_l, uA=A[i][0], lA=A[i][1])

        batch_size = C.shape[0]
        output_shape = node.forward_value.shape[1:]
        if node.forward_value.contiguous().view(batch_size, -1).shape[1] != C.shape[1]:
            output_shape = [-1]

        if node.from_input:
            lb = ptb.concretize_2bounds(x, root.lA, lower_sum_b, sign=-1, y=nodes_perturb_list)
            ub = ptb.concretize_2bounds(x, root.uA, upper_sum_b, sign=+1, y=nodes_perturb_list)
        else:
            lb, ub = lower_sum_b.reshape(-1), upper_sum_b.reshape(-1)

        return lb.view(batch_size, *output_shape), ub.view(batch_size, *output_shape)
