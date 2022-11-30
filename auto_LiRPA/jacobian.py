"""Handle Jacobian bounds."""

import torch
import numpy as np
from auto_LiRPA.bound_ops import BoundInput, BoundParams, BoundAdd
from auto_LiRPA.bound_ops import GradNorm, JVP
from auto_LiRPA.utils import get_spec_matrix, Flatten
from collections import deque


def augment_gradient_graph(self, dummy_input, norm=None, vector=None):
    """Augment the computational graph with gradient computation."""

    device = dummy_input.device
    final_node = self.final_node()
    if final_node.forward_value is None:
        self(dummy_input)
    output = final_node.forward_value
    if output.ndim != 2:
        raise NotImplementedError(
            'The model should have a 2-D output shape of '
            '(batch_size, output_dim)')
    output_dim = output.size(1)

    # Gradient values in `grad` may not be accurate. We do not consider gradient
    # accumulation from multiple succeeding nodes. We only want the shapes but
    # not the accurate values.
    grad = {}
    # Dummy values in grad_start
    grad_start = torch.ones(output.size(0), output_dim, device=device)
    grad[final_node.name] = grad_start
    input_node_found = False

    # First BFS pass: traverse the graph, count degrees, and build gradient
    # layers.
    # Degrees of nodes.
    degree = {}
    # Original layer for gradient computation.
    layer_grad = {}
    # Input nodes in gradient computation in back propagation.
    input_nodes = {}
    # Dummy input values for gradient computation received.
    grad_input = {}
    # Extra nodes as arguments used for gradient computation.
    # They must match the order in grad_input.
    grad_extra_nodes = {}

    degree[final_node.name] = 0
    queue = deque([final_node])
    while len(queue) > 0:
        node = queue.popleft()
        grad_extra_nodes[node.name] = []
        input_nodes[node.name] = node.inputs

        if isinstance(node, BoundInput):
            if input_node_found:
                raise NotImplementedError(
                    'There must be exactly one BoundInput node, '
                    'but found more than 1.')
            if vector is not None:
                # Compute JVP bounds
                layer_grad[node.name] = JVP(vector=vector)
            else:
                if norm is None:
                    layer_grad[node.name] = Flatten()
                else:
                    if norm != np.inf:
                        raise NotImplementedError(
                            'Only inf norm is supported for now.')
                    dual_norm = 1. / (1. - 1. / norm) if norm != 1 else np.inf
                    layer_grad[node.name] = GradNorm(norm=dual_norm)
            grad_input[node.name] = (grad[node.name],)
            input_node_found = True
        else:
            ret = node.build_gradient_node(grad[node.name])
            node_grad, grad_input_, grad_extra_nodes_ = ret
            layer_grad[node.name] = node_grad
            grad_input[node.name] = grad_input_
            grad_extra_nodes[node.name] = grad_extra_nodes_

        # Propagate gradients to the input nodes and update degrees.
        grad_next = layer_grad[node.name](*grad_input[node.name])
        if isinstance(grad_next, torch.Tensor):
            grad_next = [grad_next]
        if not isinstance(node, BoundInput):
            for i in range(len(grad_next)):
                grad[input_nodes[node.name][i].name] = grad_next[i]
                if not input_nodes[node.name][i].name in degree:
                    degree[input_nodes[node.name][i].name] = 0
                    queue.append(input_nodes[node.name][i])
                degree[input_nodes[node.name][i].name] += 1

    if not input_node_found:
        raise NotImplementedError(
            'There must be exactly one BoundInput node, but found none.')

    # Second BFS pass: build the backward computational graph
    grad_node = {}
    grad_node[final_node.name] = BoundInput(
        f'/grad{final_node.name}', grad_start)
    grad_node[final_node.name].name = f'/grad{final_node.name}'
    self.add_input_node(grad_node[final_node.name], index='auto')
    queue = deque([final_node])
    while len(queue) > 0:
        node = queue.popleft()
        nodes_op, nodes_in, nodes_out, _ = self._convert_nodes(
            layer_grad[node.name], grad_input[node.name])
        rename_dict = {}
        assert isinstance(nodes_in[0], BoundInput)
        rename_dict[nodes_in[0].name] = grad_node[node.name].name
        for i in range(1, len(nodes_in)):
            # Assume it's a parameter here
            new_name = f'/grad{node.name}/params{nodes_in[i].name}'
            rename_dict[nodes_in[i].name] = new_name
        for i in range(len(nodes_op)):
            # intermediate nodes
            if not nodes_op[i].name in rename_dict:
                new_name = f'/grad{node.name}/tmp{nodes_op[i].name}'
                rename_dict[nodes_op[i].name] = new_name
        if isinstance(node, BoundInput):
            assert len(nodes_out) == 1
            rename_dict[nodes_out[0].name] = '/grad_norm'
        else:
            for i in range(len(nodes_out)):
                assert not isinstance(node.inputs[i], BoundParams)
                rename_dict[nodes_out[i].name] = f'/grad{node.inputs[i].name}'

        self.rename_nodes(nodes_op, nodes_in, rename_dict)
        # Replace input nodes
        # grad_extra_nodes[node.name]: ReLU's input
        input_nodes_replace = (
            [self._modules[nodes_in[0].name]] + grad_extra_nodes[node.name])
        for i in range(len(input_nodes_replace)):
            for n in nodes_op:
                for j in range(len(n.inputs)):
                    if n.inputs[j].name == nodes_in[i].name:
                        n.inputs[j] = input_nodes_replace[i]
        self.add_nodes(nodes_op + nodes_in[len(input_nodes_replace):])

        if not isinstance(node, BoundInput):
            for i in range(len(nodes_out)):
                if input_nodes[node.name][i].name in grad_node:
                    node_cur = grad_node[input_nodes[node.name][0].name]
                    node_add = BoundAdd(
                        attr=None, inputs=[node_cur, nodes_out[i]],
                        output_index=0, options={})
                    node_add.name = f'{nodes_out[i].name}/add'
                    grad_node[input_nodes[node.name][0].name] = node_add
                else:
                    grad_node[input_nodes[node.name][0].name] = nodes_out[i]
                degree[input_nodes[node.name][i].name] -= 1
                if degree[input_nodes[node.name][i].name] == 0:
                    queue.append(input_nodes[node.name][i])

    self(dummy_input, grad_start, final_node_name='/grad_norm')

    self.bound_opts['jacobian'] = {
        'norm': norm,
        'vector': vector is not None,
    }
    # Disable unstable options
    self.bound_opts.update({
        'sparse_intermediate_bounds': False,
        'sparse_conv_intermediate_bounds': False,
        'sparse_intermediate_bounds_with_ibp': False,
        'sparse_features_alpha': False,
        'sparse_spec_alpha': False,
    })

    self.forward_final_name = self.final_name
    self.final_name = '/grad_norm'
    self.output_name = ['/grad_norm']

    return self


def compute_jacobian_bounds(
        self, x, optimize=True, reduce=True, c_opt=None, labels=None):
    """Compute jacobian bounds on the pre-augmented graph.

    Args:
        x: Input to the mode.
        optimize: Optimize relaxation bounds.
        reduce: Reduce the Jacobian bounds by taking the max
        (for L-inf local Lipschitz constants).
        c_opt (optional): Specification for optimizing the bounds on the forward
        computational graph.
        labels (optional): Ground-truth labels for constructing a default c_opt
        if it c_opt not explicitly provided.

    Returns:
        ret: If reduce=True, return the maximum Jacobian bounds over all the
        output dimensions, otherwise return the Jacobian bounds for all the
        output dimensions in a tensor.
    """

    assert 'jacobian' in self.bound_opts, (
        'Call augment_gradient_graph to augment the computational graph '
        'with the backward graph first')
    norm = self.bound_opts.get('jacobian', {}).get('norm', None)
    assert norm is None or norm == np.inf, (
        'Only Linf norm of Jacobian is supported for now.')

    num_classes = self[self.forward_final_name].output_shape[-1]

    if optimize and c_opt is None:
        if labels is None:
            c_opt = None
        else:
            # Specification for optimizing the forward graph.
            c_opt = get_spec_matrix(x, labels, num_classes)

    ret, lower, upper = [], [], []
    grad_start = torch.zeros(x.size(0), num_classes).to(x)
    for j in range(num_classes):
        grad_start.zero_()
        grad_start[:, j] = 1
        x_extra = (grad_start,)
        intermediate_bounds = {}
        if optimize:
            self.compute_bounds(
                method='CROWN-Optimized',
                C=c_opt, x=(x,) + x_extra, bound_upper=False,
                final_node_name=self.forward_final_name)
            for node in self._modules.values():
                if hasattr(node, 'lower') and node.lower is not None:
                    intermediate_bounds[node.name] = (node.lower, node.upper)
        lb, ub = self.compute_bounds(
                method='CROWN', x=(x,) + x_extra, bound_lower=norm is None,
                intermediate_layer_bounds=intermediate_bounds)
        if norm is not None:
            ret.append(ub.view(-1))
        else:
            lower.append(lb.view(1, -1))
            upper.append(ub.view(1, -1))
    if norm is not None:
        ret = torch.concat(ret)
        if reduce:
            return ret.max()
        else:
            return ret
    else:
        lower = torch.concat(lower, dim=0)
        upper = torch.concat(upper, dim=0)
        return lower, upper
