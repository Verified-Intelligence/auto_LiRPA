import pdb, time
from auto_LiRPA.bound_ops import *
from auto_LiRPA.bounded_tensor import BoundedTensor, BoundedParameter
from auto_LiRPA.bound_op_map import bound_op_map
from auto_LiRPA.parse_graph import get_graph_params
from auto_LiRPA.utils import logger
from collections import OrderedDict

class BoundedModule(nn.Module):
    def __init__(self, model, global_input, verbose=False, bound_opts=None, device='cpu'):
        super(BoundedModule, self).__init__()
        if isinstance(model, BoundedModule):
            for key in model.__dict__.keys():
                setattr(self, key, getattr(model, key))
            return
        self.verbose = verbose
        self.bound_opts = bound_opts
        self.device = device
        if device == 'cpu':
            # in case that the device argument is missed
            logger.info('Using CPU for the BoundedModule')
        self._convert(model, global_input)
        # self.choices = nn.ModuleList([_l for _l in self.node_dict.values() if not isinstance(_l, BoundParams)])

    def train(self, mode=True):
        for node in self.nodes:
            node.train(mode=mode)

    def eval(self):
        for node in self.nodes:
            node.eval()     

    def forward(self, *x):
        self._clear(*x)

        for n in self.nodes:
            # should not use `isinstance` here since subclass `BoundParam` should not be included
            if type(n) == BoundInput:
                assert (len(x) <= len(self.root_name))
                for i in range(len(x)):
                    if self.root_name[i] == n.name:
                        n.value = x[i]
                        if isinstance(x[i], (BoundedTensor, BoundedParameter)):
                            n.perturbation = x[i].ptb
                        else:
                            n.perturbation = None
                    n.from_input = True   

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

        return final_output

    def _clear(self, *x):
        for l in self.nodes:
            l.from_input = False
            if hasattr(l, 'forward_value'):
                del(l.forward_value)
            if hasattr(l, 'linear'):
                if isinstance(l.linear, tuple):
                    for item in l.linear: 
                        del(item)
                delattr(l, 'linear')
            if hasattr(l, 'lower'):
                delattr(l, 'lower')
            if hasattr(l, 'upper'):
                delattr(l, 'upper')
            if hasattr(l, 'interval'):
                delattr(l, 'interval')

    def _get_node_input(self, nodesOP, nodesIO, node):
        ret = []
        ori_names = []
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
                    ori_names.append(io.ori_name)
                    break
            if len(ret) <= i:
                raise ValueError('cannot find inputs of node: {}'.format(node.name))
        return ret, ori_names

    def _convert_nodes(self, model, global_input):
        global_input_cpu = tuple([i.to('cpu') for i in list(global_input)])
        model.train()
        model.to('cpu')
        nodesOP, nodesIO = get_graph_params(model, global_input_cpu)
        model.to(self.device)
        for i in range(0, len(nodesIO)):
            if nodesIO[i].param is not None:
                nodesIO[i] = nodesIO[i]._replace(param=nodesIO[i].param.to(self.device))

        for n in range(len(nodesOP)):
            attr = nodesOP[n].attr
            inputs, ori_names = self._get_node_input(nodesOP, nodesIO, nodesOP[n])

            if nodesOP[n].op in bound_op_map:
                if nodesOP[n].op == 'onnx::BatchNormalization':
                    # BatchNormalization node needs model.training flag to set running mean and vars
                    nodesOP[n] = nodesOP[n]._replace(
                        bound_node=bound_op_map[nodesOP[n].op](
                            nodesOP[n].inputs, nodesOP[n].name, None, attr,
                            inputs, nodesOP[n].output_index, self.device, model.training))
                elif nodesOP[n].op == 'onnx::Relu':
                    nodesOP[n] = nodesOP[n]._replace(
                        bound_node=bound_op_map[nodesOP[n].op](
                            nodesOP[n].inputs, nodesOP[n].name, None, attr,
                            inputs, nodesOP[n].output_index, self.device, self.bound_opts))
                else:
                    nodesOP[n] = nodesOP[n]._replace(
                        bound_node=bound_op_map[nodesOP[n].op](
                            nodesOP[n].inputs, nodesOP[n].name, None, attr,
                            inputs, nodesOP[n].output_index, self.device))
            else:
                print(nodesOP[n])
                raise NotImplementedError('Unsupported operation {}'.format(nodesOP[n].op))

            if self.verbose:
                logger.debug('Convert complete for {} with operation: {}'.format(nodesOP[n].name, nodesOP[n].op))

        for i in range(0, len(global_input)):
            nodesIO[i] = nodesIO[i]._replace(param=global_input[i], bound_node=BoundInput(
                nodesIO[i].inputs, nodesIO[i].name, nodesIO[i].ori_name,
                value=global_input[i], perturbation=nodesIO[i].perturbation))
        for i in range(len(global_input), len(nodesIO)):
            nodesIO[i] = nodesIO[i]._replace(bound_node=BoundParams(
                nodesIO[i].inputs, nodesIO[i].name, nodesIO[i].ori_name,
                value=nodesIO[i].param, perturbation=nodesIO[i].perturbation))

        return nodesOP, nodesIO

    def _build_graph(self, nodesOP, nodesIO):
        self.nodes = []
        for node in nodesOP + nodesIO:
            assert (node.bound_node is not None)
            self.nodes.append(node.bound_node)
        self.final_name = nodesOP[-1].name
        self.root_name = [node.name for node in nodesIO]
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
                name_base = nodesOP[n].name + '/split/'
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

        nodesOP, nodesIO = self._convert_nodes(model, global_input)
        global_input = tuple([i.to(self.device) for i in global_input])

        while True:
            self._build_graph(nodesOP, nodesIO)
            self.forward(*global_input)
            nodesOP, nodesIO, found_complex = self._split_complex(nodesOP, nodesIO)
            if not found_complex: break

        for node in self.nodes:
            for p in list(node.named_parameters()):
                if node.ori_name not in self._parameters:
                    # For parameter or input nodes, use their original name directly
                    self._parameters[node.ori_name] = p[1]

        logger.debug('NodesOP:')
        for node in nodesOP:
            logger.debug('{}'.format(node._replace(param=None)))
        logger.debug('NodesIO')
        for node in nodesIO:
            logger.debug('{}'.format(node._replace(param=None)))

        if self.verbose:
            logger.info('Model converted to support bounds')

    def compute_bounds(self, aux=None, C=None, IBP=False, forward=False, method='backward', bound_lower=True,
                       bound_upper=True, reuse_ibp=False):
        if IBP and method is None and reuse_ibp:
            # directly return the previously saved ibp bounds
            return self.ibp_lower, self.ibp_upper
        if method == 'forward':
            forward = True
        root = [self.node_dict[name] for name in self.root_name]     
        batch_size = root[0].forward_value.shape[0]
        dim_in = 0
        for i in range(len(root)):
            if root[i].perturbation is not None:
                root[i].interval = root[i].perturbation.init_interval(root[i].forward_value, aux=aux)
                root[i].linear, root[i].center, root[i].aux = root[i].perturbation.init_linear(
                    root[i].forward_value, aux=aux, forward=forward)
                root[i].from_input = True
                if forward:
                    root[i].dim = root[i].linear.lw.shape[1]
                    dim_in += root[i].dim
                if isinstance(root[i].interval, Hull):
                    root[i].lower, root[i].upper = root[i].interval.lower, root[i].interval.upper
                else:
                    root[i].lower, root[i].upper = root[i].interval
            else:
                root[i].interval = (root[i].forward_value, root[i].forward_value)                
                root[i].lower = root[i].upper = root[i].forward_value

        if forward:
            self._init_forward(root, dim_in)

        if C is None:
            # C is an identity matrix by default 
            output = self.node_dict[self.final_name].forward_value
            dim_output = output.reshape(output.shape[0], -1).shape[1]
            C = torch.eye(dim_output).to(self.device).unsqueeze(0).repeat(output.shape[0], 1, 1)

        if IBP:
            lower, upper = self._IBP_general(node=self.node_dict[self.final_name], C=C)
            self.ibp_lower, self.ibp_upper = lower, upper

        if method is None:
            return self.ibp_lower, self.ibp_upper    

        # check whether weights perturb and set nonlinear for the BoundMatMul operation
        for n in self.nodes:
            if isinstance(n, (BoundLinear, BoundConv2d, BoundBatchNorm2d)):
                for l_name in n.input_name[1:]:
                    node = self.node_dict[l_name]
                    if hasattr(node, 'perturbation'):
                        if node.perturbation is not None:
                            node.lower, node.upper = node.perturbation.init_interval(node.forward_value)
                            n.nonlinear = True

        final = self.node_dict[self.final_name]

        for i in range(len(self.nodes)):
            if hasattr(self.nodes[i], 'nonlinear') and self.nodes[i].nonlinear:
                for l_name in self.nodes[i].input_name:
                    node = self.node_dict[l_name]
                    if not hasattr(node, 'lower'):
                        if forward:
                            l, u = self._forward_general(
                                node=node, root=root, dim_in=dim_in, concretize=True)
                        else:
                            # assign concretized bound for ReLU layer to save computational cost
                            if isinstance(node, BoundReLU) and hasattr(self.node_dict[node.input_name[0]], 'lower'):
                                node.lower = node.forward(self.node_dict[node.input_name[0]].lower)
                                node.upper = node.forward(self.node_dict[node.input_name[0]].upper)
                            else:
                                dim = node.forward_value[0].reshape(-1).shape[0]
                                if isinstance(node, BoundLinear):
                                    eyeC = namedtuple('eyeC', 'shape device')
                                    newC = eyeC([node.forward_value.shape[0], dim, *node.forward_value.shape[1:]], node.forward_value.device)
                                else:
                                    newC = torch.eye(dim, device=node.forward_value.device)\
                                        .unsqueeze(0).repeat(node.forward_value.shape[0], 1, 1)\
                                        .view(node.forward_value.shape[0], dim, *node.forward_value.shape[1:])
                                l, u = self._backward_general(C=newC, node=node, root=root)

        if method == 'backward':
            return self._backward_general(C=C, node=final, root=root, bound_lower=bound_lower, bound_upper=bound_upper)
        elif method == 'forward':
            # TODO: `C` is not supported yet (identity matrix by default)
            l, u = self._forward_general(node=final, root=root, dim_in=dim_in, concretize=True)
            mask = torch.gt(C, 0.).to(torch.float)
            _l = (torch.bmm(mask * C, l.unsqueeze(-1)) + \
                  torch.bmm((1 - mask) * C, u.unsqueeze(-1))).squeeze(-1)
            _u = (torch.bmm(mask * C, u.unsqueeze(-1)) + \
                  torch.bmm((1 - mask) * C, l.unsqueeze(-1))).squeeze(-1)
            return _l, _u
        else:
            raise NotImplementedError

    def compute_worst_logits(self, y=None, aux=None, IBP=False, forward=False, method='backward', reuse_ibp=False):
        logits_l, logits_u = self.compute_bounds(aux=aux, IBP=IBP, forward=forward, method=method, reuse_ibp=reuse_ibp)
        num_classes = self.node_dict[self.final_name].forward_value.shape[-1]
        one_hot = F.one_hot(y, num_classes=num_classes).to(torch.float32).to(self.device)
        return logits_l * one_hot + logits_u * (1. - one_hot)

    def _IBP_general(self, node=None, C=None):
        for l_pre in node.input_name:
            l = self.node_dict[l_pre]
            if not hasattr(l, 'interval'):
                self._IBP_general(l)

        inp = [self.node_dict[l_pre].interval for l_pre in node.input_name]

        if C is not None:
            if isinstance(node, BoundLinear) and torch.isclose(inp[1][1], inp[1][0], 1e-8).all():
                # merge the output node with the specification, available when weights of this layer are not perturbed
                node.interval = node.interval_propagate(*inp, C=C)
            else:
                interval_before_C = [node.interval_propagate(*inp)]
                node.interval = BoundLinear.interval_propagate(None, *interval_before_C, C=C)
        else:
            node.interval = node.interval_propagate(*inp)
        node.lower, node.upper = node.interval

        return node.interval

    def _backward_general(self, C=None, node=None, root=None, bound_lower=True, bound_upper=True):
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
        node.lA = C if bound_lower else None
        node.uA = C if bound_upper else None
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

            if l.lA is not None or l.uA is not None:
                def add_bound(node, lA, uA):
                    if lA is not None:
                        node.lA = lA if node.lA is None else (node.lA + lA)
                    if uA is not None:
                        node.uA = uA if node.uA is None else (node.uA + uA)

                input_nodes = [self.node_dict[l_name] for l_name in l.input_name]
                A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *input_nodes)
                lb = lb + lower_b
                ub = ub + upper_b

                for i, l_pre in enumerate(l.input_name):
                    _l = self.node_dict[l_pre]
                    add_bound(_l, lA=A[i][0], uA=A[i][1])

        batch_size = C.shape[0]
        output_shape = node.forward_value.shape[1:]
        if node.forward_value.contiguous().view(batch_size, -1).shape[1] != C.shape[1]:
            output_shape = [-1]

        for i in range(len(root)): 
            if root[i].lA is None and root[i].uA is None: continue
            logger.debug('concretize node: {} shape: {}'.format(root[i],root[i].lA.shape))
            lA = root[i].lA.reshape(batch_size, root[i].lA.shape[1], -1) if bound_lower else None
            uA = root[i].uA.reshape(batch_size, root[i].uA.shape[1], -1) if bound_upper else None
            if root[i].perturbation is not None:
                if isinstance(root[i], BoundParams):
                    # add batch_size dim for weights node
                    lb = lb + root[i].perturbation.concretize(
                        root[i].center.unsqueeze(0).repeat(([batch_size] + [1] * len(root[i].center.shape))), lA,
                        sign=-1,aux=root[i].aux) if bound_lower else None
                    ub = ub + root[i].perturbation.concretize(
                        root[i].center.unsqueeze(0).repeat(([batch_size] + [1] * len(root[i].center.shape))), uA,
                        sign=+1,aux=root[i].aux) if bound_upper else None
                else:
                    lb = lb + root[i].perturbation.concretize(root[i].center, lA, sign=-1, aux=root[i].aux) if bound_lower else None
                    ub = ub + root[i].perturbation.concretize(root[i].center, uA, sign=+1, aux=root[i].aux) if bound_upper else None
            elif i < self.num_global_inputs:
                lb = lb + root[i].lA.reshape(batch_size, root[i].lA.shape[1], -1).bmm(
                    root[i].forward_value.view(batch_size, -1, 1)).squeeze(-1) if bound_lower else None
                ub = ub + root[i].uA.reshape(batch_size, root[i].uA.shape[1], -1).bmm(
                    root[i].forward_value.view(batch_size, -1, 1)).squeeze(-1) if bound_upper else None
            else:
                lb = lb + root[i].lA.reshape(batch_size, root[i].lA.shape[1], -1).matmul(
                    root[i].forward_value.view(-1, 1)).squeeze(-1) if bound_lower else None
                ub = ub + root[i].uA.reshape(batch_size, root[i].uA.shape[1], -1).matmul(
                    root[i].forward_value.view(-1, 1)).squeeze(-1) if bound_upper else None

        node.lower = lb.view(batch_size, *output_shape) if bound_lower else None
        node.upper = ub.view(batch_size, *output_shape) if bound_upper else None
        return node.lower, node.upper

    def _forward_general(self, node=None, root=None, dim_in=None, concretize=False):
        if hasattr(node, 'lower'):
            return node.lower, node.upper

        if not node.from_input:
            shape = node.forward_value.shape
            w = None
            b = node.forward_value
            node.linear = LinearBound(w, b, w, b, b, b)
            node.lower = node.upper = b
            node.interval = (node.lower, node.upper) 
            return node.interval

        if not hasattr(node, 'linear'):
            for l_pre in node.input_name:
                l = self.node_dict[l_pre]
                if not hasattr(l, 'linear'):
                    self._forward_general(node=l, root=root, dim_in=dim_in)

            inp = [self.node_dict[l_pre].linear for l_pre in node.input_name]
            node.linear = node.bound_forward(dim_in, *inp)

        if concretize:
            if node.linear.lw is not None:
                prev_dim_in = 0
                lower, upper = node.linear.lb, node.linear.ub
                lw, uw = node.linear.lw, node.linear.uw
                batch_size = lw.shape[0]
                assert (len(lw.shape) > 1)
                lA = lw.reshape(batch_size, dim_in, -1).transpose(1, 2)
                uA = uw.reshape(batch_size, dim_in, -1).transpose(1, 2)
                for i in range(len(root)):
                    if root[i].perturbation is not None:
                        _lA = lA[:, :, prev_dim_in : (prev_dim_in + root[i].dim)]
                        _uA = uA[:, :, prev_dim_in : (prev_dim_in + root[i].dim)]
                        lower = lower + root[i].perturbation.concretize(
                            root[i].center, _lA, sign=-1, aux=root[i].aux).view(node.forward_value.shape)
                        upper = upper + root[i].perturbation.concretize(
                            root[i].center, _uA, sign=+1, aux=root[i].aux).view(node.forward_value.shape)
                        prev_dim_in += root[i].dim
                node.linear = node.linear._replace(lower=lower, upper=upper)
                node.lower, node.upper = lower, upper
                return node.lower, node.upper
            else:
                node.lower, node.upper = node.linear.lb, node.linear.ub
                return node.lower, node.upper

    def _init_forward(self, root, dim_in):
        if dim_in == 0:
            raise ValueError("At least one node should have a specified perturbation")
        prev_dim_in = 0
        batch_size = root[0].forward_value.shape[0]
        for i in range(len(root)):
            if root[i].perturbation is not None:
                shape = root[i].linear.lw.shape
                device = root[i].linear.lw.device
                root[i].linear = root[i].linear._replace(
                    lw=torch.cat([
                        torch.zeros(shape[0], prev_dim_in, *shape[2:], device=device),
                        root[i].linear.lw,
                        torch.zeros(shape[0], dim_in - shape[1], *shape[2:], device=device)
                    ], dim=1),
                    uw=torch.cat([
                        torch.zeros(shape[0], prev_dim_in, *shape[2:], device=device),
                        root[i].linear.uw,
                        torch.zeros(shape[0], dim_in - shape[1] - prev_dim_in, *shape[2:], device=device)
                    ], dim=1)
                )
                if i >= self.num_global_inputs:
                    root[i].forward_value = root[i].forward_value.unsqueeze(0).repeat(
                        *([batch_size] + [1] * len(self.forward_value.shape)))
                prev_dim_in += shape[1]
            else:
                fv = root[i].forward_value
                shape = fv.shape
                if root[i].from_input:
                    w = torch.zeros(shape[0], dim_in, *shape[1:], device=self.device)
                else:
                    w = None
                b = fv
                root[i].linear = LinearBound(w, b, w, b, b, b)
                root[i].lower = root[i].upper = b
                root[i].interval = (root[i].lower, root[i].upper)

