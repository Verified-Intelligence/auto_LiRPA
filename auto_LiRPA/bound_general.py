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

import copy
from typing import List
import numpy as np
import warnings
from collections import OrderedDict, deque

import torch
from torch.nn import Parameter

from .bound_op_map import bound_op_map
from .bound_ops import *
from .bounded_tensor import BoundedTensor, BoundedParameter
from .parse_graph import parse_module
from .perturbations import *
from .utils import *
from .patches import Patches
from .optimized_bounds import default_optimize_bound_args


warnings.simplefilter('once')


class BoundedModule(nn.Module):
    """Bounded module with support for automatically computing bounds.

    Args:
        model (nn.Module): The original model to be wrapped by BoundedModule.

        global_input (tuple): A dummy input to the original model. The shape of
        the dummy input should be consistent with the actual input to the model
        except for the batch dimension.

        bound_opts (dict): Options for bounds. See
        `Bound Options <bound_opts.html>`_.

        device (str or torch.device): Device of the bounded module.
        If 'auto', the device will be automatically inferred from the device of
        parameters in the original model or the dummy input.

        custom_ops (dict): A dictionary of custom operators.
        The dictionary maps operator names to their corresponding bound classes
        (subclasses of `Bound`).

    """
    def __init__(self, model, global_input, bound_opts=None,
                device='auto', verbose=False, custom_ops=None):
        super().__init__()
        if isinstance(model, BoundedModule):
            for key in model.__dict__.keys():
                setattr(self, key, getattr(model, key))
            return

        self.global_input = global_input
        self.ori_training = model.training
        self.check_incompatible_nodes(model)

        if bound_opts is None:
            bound_opts = {}
        # Default options.
        default_bound_opts = {
            'conv_mode': 'patches',
            'sparse_intermediate_bounds': True,
            'sparse_conv_intermediate_bounds': True,
            'sparse_intermediate_bounds_with_ibp': True,
            'sparse_features_alpha': True,
            'sparse_spec_alpha': True,
            'minimum_sparsity': 0.9,
            'enable_opt_interm_bounds': False,
            'crown_batch_size': np.inf,
            'forward_refinement': False,
            'forward_max_dim': int(1e9),
            # Do not share alpha for conv layers.
            'use_full_conv_alpha': True,
            'disabled_optimization': [],
            # Threshold for number of unstable neurons for each layer to disable
            #  use_full_conv_alpha.
            'use_full_conv_alpha_thresh': 512,
            'verbosity': 1 if verbose else 0,
            'optimize_graph': {'optimizer': None},
            'compare_crown_with_ibp': False,
        }
        default_bound_opts.update(bound_opts)
        self.bound_opts = default_bound_opts
        optimize_bound_args = copy.deepcopy(default_optimize_bound_args)
        optimize_bound_args.update(
            self.bound_opts.get('optimize_bound_args', {}))
        self.bound_opts.update({'optimize_bound_args': optimize_bound_args})

        self.verbose = verbose
        self.custom_ops = custom_ops if custom_ops is not None else {}
        if device == 'auto':
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                # Model has no parameters. We use the device of input tensor.
                if isinstance(global_input, torch.Tensor):
                    self.device = global_input.device
                elif isinstance(global_input, tuple):
                    self.device = global_input[0].device
                else:
                    raise NotImplementedError( # pylint: disable=raise-missing-from
                        'Unable to decide the device. Consider providing a '
                        '`device` argument to `BoundedModule` explicitly.')
        else:
            self.device = device
        self.conv_mode = self.bound_opts.get('conv_mode', 'patches')
        # Cached IBP results which may be reused
        self.ibp_lower, self.ibp_upper = None, None

        self.optimizable_activations = []
        self.relus = []  # save relu layers for convenience
        self.layers_with_constraint = []

        state_dict_copy = copy.deepcopy(model.state_dict())
        object.__setattr__(self, 'ori_state_dict', state_dict_copy)
        model.to(self.device)
        inputs_unpacked = unpack_inputs(global_input, device=self.device)
        output = model(*inputs_unpacked)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                'Output of the model is expected to be a single torch.Tensor. '
                f'Actual type: {type(output)}')
        self.final_shape = output.shape
        self.bound_opts.update({'final_shape': self.final_shape})
        self._convert(model, global_input)
        self._optimize_graph()
        self._mark_perturbed_nodes(inputs_unpacked)
        self._expand_jacobian()
        self._check_patches_mode()

        self.next_split_hint = []  # Split hints, used in beta optimization.
        # Beta values for all intermediate bounds.
        # Set to None (not used) by default.
        self.best_intermediate_betas = None
        # Initialization value for intermediate betas.
        self.init_intermediate_betas = None
        # whether using cut
        self.cut_used = False
        # a placeholder for cut timestamp, which would be a non-positive int
        self.cut_timestamp = -1
        # a placeholder to save the latest samplewise mask for
        # pruning-in-iteration optimization
        self.last_update_preserve_mask = None
        # If output constraints are used, it is possible that none of the possible
        # inputs satisfy them. In this case, the lower bounds will be set to +inf,
        # and the upper bounds to -inf.
        self.infeasible_bounds = None
        self.solver_model = None
        # Needed for output constraints - the output layer should not use them
        self.final_node().is_final_node = True
        # Customized input domain
        self.input_domain = None
        self.dynamic = False

    def nodes(self) -> List[Bound]:
        return self._modules.values()

    def get_enabled_opt_act(self):
        # Optimizable activations that are actually used and perturbed
        return [
            n for n in self.optimizable_activations
            if n.used and n.perturbed and not getattr(n, 'is_linear_op', False)
        ]

    def get_optimizable_activations(self):
        for node in self.nodes():
            if (isinstance(node, BoundOptimizableActivation)
                    and node.optimizable
                    and len(getattr(node, 'requires_input_bounds', [])) > 0
                    and node not in self.optimizable_activations):
                disabled = False
                for item in self.bound_opts.get('disable_optimization', []):
                    if item.lower() in str(type(node)).lower():
                        disabled = True
                if disabled:
                    logging.info('Disabled optimization for %s', node)
                    continue
                if node not in self.optimizable_activations:
                    self.optimizable_activations.append(node)
            if isinstance(node, BoundRelu) and node not in self.relus:
                self.relus.append(node)

    def get_perturbed_optimizable_activations(self):
        return [n for n in self.optimizable_activations if n.perturbed]

    def get_splittable_activations(self):
        """Activation functions that can be split during branch and bound."""
        return [n for n in self.nodes() if n.perturbed and n.splittable]

    def get_layers_requiring_bounds(self):
        """Layer names whose intermediate layer bounds are required."""
        intermediate_layers = []
        tighten_input_bounds = (
            self.bound_opts['optimize_bound_args']['tighten_input_bounds']
        )
        directly_optimize_layer_names = (
            self.bound_opts['optimize_bound_args']['directly_optimize']
        )
        for node in self.nodes():
            if node.name in directly_optimize_layer_names:
                intermediate_layers.append(node)
            if not node.used or not node.perturbed:
                continue
            for i in getattr(node, 'requires_input_bounds', []):
                input_node = node.inputs[i]
                if (input_node not in intermediate_layers
                        and input_node.perturbed):
                    # If not perturbed, it may not have the batch dimension.
                    # So we do not include it, and it is unnecessary.
                    intermediate_layers.append(input_node)
            if (
                node.name in self.layers_with_constraint
                or (isinstance(node, BoundInput) and tighten_input_bounds)
            ):
                if node not in intermediate_layers:
                    intermediate_layers.append(node)
        return intermediate_layers

    def check_incompatible_nodes(self, model):
        """Check whether the model has incompatible nodes that the conversion
        may be inaccurate"""
        node_types = [type(m) for m in list(model.modules())]

        if (torch.nn.Dropout in node_types
                and torch.nn.BatchNorm1d in node_types
                and self.global_input.shape[0] == 1):
            print('We cannot support torch.nn.Dropout and torch.nn.BatchNorm1d '
                  'at the same time!')
            print('Suggest to use another dummy input which has batch size '
                  'larger than 1 and set model to train() mode.')
            return

        if not self.ori_training and torch.nn.Dropout in node_types:
            print('Dropout operation CANNOT be parsed during conversion when '
                  'the model is in eval() mode!')
            print('Set model to train() mode!')
            self.ori_training = True

        if self.ori_training and torch.nn.BatchNorm1d in node_types:
            print('BatchNorm1d may raise error during conversion when the model'
                  ' is in train() mode!')
            print('Set model to eval() mode!')
            self.ori_training = False

    def non_deter_wrapper(self, op, *args, **kwargs):
        """Some operations are non-deterministic and deterministic mode will
        fail. So we temporary disable it."""
        if self.bound_opts.get('deterministic', False):
            torch.use_deterministic_algorithms(False)
        ret = op(*args, **kwargs)
        if self.bound_opts.get('deterministic', False):
            torch.use_deterministic_algorithms(True)
        return ret

    def non_deter_scatter_add(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.scatter_add, *args, **kwargs)

    def non_deter_index_select(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.index_select, *args, **kwargs)

    def set_bound_opts(self, new_opts):
        for k, v in new_opts.items():
            # assert v is not dict, 'only support change optimize_bound_args'
            if type(v) == dict:
                self.bound_opts[k].update(v)
            else:
                self.bound_opts[k] = v

    def set_gcp_relu_indicators(self, relu_layer_name, relu_indicators):
        """
        Sets the GCP (Generalized Cutting Plane) relu indicators for
        the specified ReLU layer by name.
        Args:
            relu_layer_name (str):
                The name of the ReLU layer to update.
            relu_indicators (torch.Tensor):
                A tensor containing unstable relu indices or masks.
        """
        # Search for the layer by name
        for m in self.relus:
            if m.name == relu_layer_name:
                # Set the indicators for the found ReLU layer
                m.gcp_unstable_relu_indicators = relu_indicators
                return
        # If not found, raise an error
        raise ValueError(f'No ReLU layer found with name {relu_layer_name}')

    @staticmethod
    def _get_A_norm(A):
        if not isinstance(A, (list, tuple)):
            A = (A, )
        norms = []
        for aa in A:
            if aa is not None:
                if isinstance(aa, Patches):
                    aa = aa.patches
                norms.append(aa.abs().sum().item())
            else:
                norms.append(None)
        return norms

    def __call__(self, *input, **kwargs):
        if 'method_opt' in kwargs:
            opt = kwargs['method_opt']
            kwargs.pop('method_opt')
        else:
            opt = 'forward'
        for kwarg in [
            'disable_multi_gpu', 'no_replicas', 'get_property',
            'node_class', 'att_name']:
            if kwarg in kwargs:
                kwargs.pop(kwarg)
        if opt == 'compute_bounds':
            return self.compute_bounds(**kwargs)
        else:
            return self.forward(*input, **kwargs)

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                'cannot assign parameter before Module.__init__() call')
        elif not isinstance(name, str):
            raise TypeError('parameter name should be a string. '
                            f'Got {torch.typename(name)}')
        elif name == '':
            raise KeyError('parameter name can\'t be empty string')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f'attribute "{name}" already exists')

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f'cannot assign "{torch.typename(param)}" object to '
                f'parameter "{name}" '
                '(torch.nn.Parameter or None required)')
        elif param.grad_fn:
            raise ValueError(
                f'Cannot assign non-leaf Tensor to parameter "{name}". Model '
                'parameters must be created explicitly. To express "{name}" '
                'as a function of another Tensor, compute the value in '
                'the forward() method.')
        else:
            self._parameters[name] = param

    def _named_members(self,
                       get_members_fn,
                       prefix='',
                       recurse=True,
                       remove_duplicate: bool = True,
                       **kwargs):  # pylint: disable=unused-argument
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [
                                     (prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                # translate name to ori_name
                if name in self.node_name_map:
                    name = self.node_name_map[name]
                yield name, v

    def train(self, mode=True):
        super().train(mode)
        for node in self.nodes():
            node.train(mode=mode)

    def eval(self):
        super().eval()
        for node in self.nodes():
            node.eval()

    def to(self, *args, **kwargs):
        # Moves and/or casts some attributes except pytorch will do by default.
        for node in self.nodes():
            for attr in ['lower', 'upper', 'forward_value', 'd', 'lA',]:
                if hasattr(node, attr):
                    this_attr = getattr(node, attr)
                    if isinstance(this_attr, torch.Tensor):
                        this_attr = this_attr.to(*args, **kwargs)
                        setattr(node, attr, this_attr)

            if hasattr(node, 'interval'):
                # construct new interval
                this_attr = getattr(node, 'interval')
                setattr(node, 'interval', (this_attr[0].to(
                    *args, **kwargs), this_attr[1].to(*args, **kwargs)))

        return super().to(*args, **kwargs)

    def __getitem__(self, name):
        module = self._modules[name]
        # We never create modules that are None, the assert fixes type hints
        assert module is not None
        return module

    def roots(self):
        return [self[name] for name in self.root_names]

    def final_node(self):
        return self[self.final_name]

    @staticmethod
    def _is_shape_compatible(shape1, shape2):
        """
        Check whether two tensor shapes shape1 and shape2 are compatible:
        1. they need to have the same number of dimensions.
        2. for each dimension, the number of elements need to be the same,
           or one of the shape has only 1 element.
        """
        if len(shape1) != len(shape2):
            return False
        for (s1, s2) in zip(shape1, shape2):
            if not (s1 == s2 or s1 == 1 or s2 == 1):
                return False
        return True


    def get_forward_value(self, node):
        """ Recursively get `forward_value` for `node` and its parent nodes"""
        if getattr(node, 'forward_value', None) is not None:
            return node.forward_value
        inputs = [self.get_forward_value(inp) for inp in node.inputs]
        for inp in node.inputs:
            node.from_input = node.from_input or inp.from_input
        node.input_shape = inputs[0].shape if len(inputs) > 0 else None
        fv = node.forward(*inputs)
        if isinstance(fv, (torch.Size, tuple)):
            fv = torch.tensor(fv, device=self.device)
        node.forward_value = fv
        node.output_shape = fv.shape
        # In most cases, the batch dimension is just the first dimension
        # if the node depends on input. Otherwise if the node doesn't
        # depend on input, there is no batch dimension (default is -1).
        # FIXME: This heuristic is not always correct. Some BoundConstant and
        # BoundBuffer do not depend on inputs may also have batch dimension.
        node.batch_dim = 0 if node.from_input else node.batch_dim
        # If one input has the same shape as this layer, and this node has
        # a batch dimension, then this input must also has a batch dimension.
        for inp in node.inputs:
            if (node.batch_dim != -1 and inp.batch_dim == -1 and
                    len(node.output_shape) != 1 and self._is_shape_compatible(
                        node.output_shape, inp.output_shape)):
                # For now, enable this for constants and buffers only, because
                # these are the problems we found so far. Need further testing
                # on general cases.
                infer_batch_dim = isinstance(
                    inp, (BoundConstant, BoundBuffers))
                message = (f'Node {inp} with shape {inp.output_shape}'
                           f' {"used" if infer_batch_dim else "ignored"}'
                           f' a inferred batch dimension {inp.batch_dim}.'
                           f' The node {node} following it has a compatible'
                           f' shape {node.output_shape}')
                if infer_batch_dim:
                    inp.batch_dim = node.batch_dim
                logger.debug(message)
        # Unperturbed node but it is not a root node.
        # Save forward_value to value. (Can be used in forward bounds.)
        if not node.from_input and len(node.inputs) > 0:
            node.value = node.forward_value
        return fv

    def forward(self, *x, final_node_name=None, clear_forward_only=False,
            reset_perturbed_nodes=True):
        r"""Standard forward computation for the network.

        Args:
            x (tuple or None): Input to the model.

            final_node_name (str, optional): The name of the final node in the
            model. The value on the corresponding node will be returned.

            clear_forward_only (bool, default `False`): Whether only standard
            forward values stored on the nodes should be cleared. If `True`,
            only standard forward values stored on the nodes will be cleared.
            Otherwise, bound information on the nodes will also be cleared.

            reset_perturbed_nodes (bool, default `True`): Mark all perturbed
            nodes with input perturbations. When set to `True`, it may
            accidentally clear all .perturbed properties for intermediate
            nodes.

        Returns:
            output: The output of the model, or if `final_node_name` is not
            `None`, return the value on the corresponding node instead.
        """
        self.set_input(*x, clear_forward_only=clear_forward_only,
                reset_perturbed_nodes=reset_perturbed_nodes)
        if final_node_name is None:
            final_node_name = self.output_name[0]
        return self.get_forward_value(self[final_node_name])

    def _mark_perturbed_nodes(self, input):
        """Mark the graph nodes and determine which nodes need perturbation."""
        # Set some of the input as perturbed if they are bounded objects
        any_perturbed = False
        for name, index in zip(self.input_name, self.input_index):
            if index is None:
                continue
            if isinstance(input[index], (BoundedTensor, BoundedParameter)):
                self[name].perturbed = True
                any_perturbed = True
        # If none of the inputs is a bounded object, set all of them as perturbed
        if not any_perturbed:
            for name, index in zip(self.input_name, self.input_index):
                if index is not None:
                    self[name].perturbed = True

        degree_in = {}
        queue = deque()
        # Initially the queue contains all "root" nodes.
        for key in self._modules.keys():
            l = self[key]
            degree_in[l.name] = len(l.inputs)
            if degree_in[l.name] == 0:
                queue.append(l)  # in_degree ==0 -> root node

        while len(queue) > 0:
            node = queue.popleft()
            # Obtain all output node, and add the output nodes to the queue if
            # all its input nodes have been visited.
            # The initial "perturbed" property is set in BoundInput or
            # BoundParams object, depending on ptb.
            for name_next in node.output_name:
                node_next = self[name_next]
                if not node_next.never_perturbed:
                    # The next node is perturbed if it is already perturbed,
                    # or this node is perturbed.
                    node_next.perturbed = node_next.perturbed or node.perturbed
                degree_in[name_next] -= 1
                # all inputs of this node have been visited,
                # now put it in queue.
                if degree_in[name_next] == 0:
                    queue.append(node_next)
            node.update_requires_input_bounds()

        self.get_optimizable_activations()
        self.splittable_activations = self.get_splittable_activations()
        self.perturbed_optimizable_activations = (
            self.get_perturbed_optimizable_activations())
        return

    def _check_patches_mode(self):
        """Disable patches mode if there is no Conv node.

        This is a workaround (before a more general patches mode is implemented)
        to avoid issues relevant to the patches node,
        for complicated models without any Conv.
        """
        has_conv = False
        for node in self.nodes():
            if isinstance(node, (BoundConv, BoundConvTranspose, BoundConv2dGrad)):
                has_conv = True
        if not has_conv and self.conv_mode == 'patches':
            self.conv_mode = 'matrix'
            for node in self.nodes():
                if getattr(node, 'mode', None) == 'patches':
                    node.mode = 'matrix'

    def _clear_and_set_new(
        self,
        interm_bounds,
        clear_forward_only=False,
        reset_perturbed_nodes=True,
        cache_bounds=False,
    ):
        for l in self.nodes():
            if hasattr(l, 'linear'):
                if isinstance(l.linear, tuple):
                    for item in l.linear:
                        del item
                delattr(l, 'linear')

            if hasattr(l, 'patch_size'):
                l.patch_size = {}

            if clear_forward_only:
                if hasattr(l, 'forward_value'):
                    delattr(l, 'forward_value')
            else:
                for attr in ['interval', 'forward_value', 'd',
                             'lA', 'lower_d', 'upper_k']:
                    if hasattr(l, attr):
                        delattr(l, attr)
                if cache_bounds:
                    l.move_lower_and_upper_bounds_to_cache()
                else:
                    l.delete_lower_and_upper_bounds()

            for attr in ['zero_backward_coeffs_l', 'zero_backward_coeffs_u',
                         'zero_lA_mtx', 'zero_uA_mtx']:
                setattr(l, attr, False)
            # Given an interval here to make IBP/CROWN start from this node
            if interm_bounds is not None and l.name in interm_bounds.keys():
                l.interval = tuple(interm_bounds[l.name][:2])
                l.lower = interm_bounds[l.name][0]
                l.upper = interm_bounds[l.name][1]
                if l.lower is not None:
                    l.lower = l.lower.detach().requires_grad_(False)
                if l.upper is not None:
                    l.upper = l.upper.detach().requires_grad_(False)
            # Mark all nodes as non-perturbed except for weights.
            if reset_perturbed_nodes:
                if not hasattr(l, 'perturbation') or l.perturbation is None:
                    l.perturbed = False

            # Clear operator-specific attributes
            l.clear()

    def set_input(
        self,
        *x,
        interm_bounds=None,
        clear_forward_only=False,
        reset_perturbed_nodes=True,
        cache_bounds=False,
    ):
        self._clear_and_set_new(
            interm_bounds=interm_bounds,
            clear_forward_only=clear_forward_only,
            reset_perturbed_nodes=reset_perturbed_nodes,
            cache_bounds=cache_bounds,
        )
        inputs_unpacked = unpack_inputs(x)
        for name, index in zip(self.input_name, self.input_index):
            if index is None:
                continue
            node = self[name]
            node.value = inputs_unpacked[index]
            if isinstance(node.value, (BoundedTensor, BoundedParameter)):
                node.perturbation = node.value.ptb
            else:
                node.perturbation = None
        # Mark all perturbed nodes.
        if reset_perturbed_nodes:
            self._mark_perturbed_nodes(inputs_unpacked)

    def _get_node_input(self, nodesOP, nodesIn, node):
        ret = []
        for i in range(len(node.inputs)):
            for op in nodesOP:
                if op.name == node.inputs[i]:
                    ret.append(op.bound_node)
                    break
            if len(ret) == i + 1:
                continue
            for io in nodesIn:
                if io.name == node.inputs[i]:
                    ret.append(io.bound_node)
                    break
            if len(ret) <= i:
                raise ValueError(f'cannot find inputs of node: {node.name}')
        return ret

    def _to(self, obj, dest, inplace=False):
        """ Move all tensors in the object to a specified dest
        (device or dtype). The inplace=True option is available for dict."""
        if obj is None:
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.to(dest)
        elif isinstance(obj, Patches):
            return obj.patches.to(dest)
        elif isinstance(obj, tuple):
            return tuple([self._to(item, dest) for item in obj])
        elif isinstance(obj, list):
            return list([self._to(item, dest) for item in obj])
        elif isinstance(obj, dict):
            if inplace:
                for k, v in obj.items():
                    obj[k] = self._to(v, dest, inplace=True)
                return obj
            else:
                return {k: self._to(v, dest) for k, v in obj.items()}
        else:
            raise NotImplementedError(type(obj))

    def _convert_nodes(self, model, global_input):
        r"""
        Returns:
            nodesOP (list): List of operator nodes
            nodesIn (list): List of input nodes
            nodesOut (list): List of output nodes
            template (object): Template to specify the output format
        """
        global_input_cpu = self._to(global_input, 'cpu')
        if self.ori_training:
            model.train()
        else:
            model.eval()
        model.to('cpu')
        nodesOP, nodesIn, nodesOut, template = parse_module(
            model, global_input_cpu)
        model.to(self.device)
        for i in range(0, len(nodesIn)):
            if nodesIn[i].param is not None:
                nodesIn[i] = nodesIn[i]._replace(
                    param=nodesIn[i].param.to(self.device))
        global_input_unpacked = unpack_inputs(global_input)

        # Convert input nodes and parameters.
        attr = {'device': self.device}
        for i, n in enumerate(nodesIn):
            if n.input_index is not None:
                nodesIn[i] = nodesIn[i]._replace(bound_node=BoundInput(
                    ori_name=nodesIn[i].ori_name,
                    value=global_input_unpacked[nodesIn[i].input_index],
                    perturbation=nodesIn[i].perturbation,
                    input_index=n.input_index, options=self.bound_opts,
                    attr=attr))
            else:
                bound_class = BoundParams if isinstance(
                    nodesIn[i].param, nn.Parameter) else BoundBuffers
                nodesIn[i] = nodesIn[i]._replace(bound_node=bound_class(
                    ori_name=nodesIn[i].ori_name, value=nodesIn[i].param,
                    perturbation=nodesIn[i].perturbation, options=self.bound_opts,
                    attr=attr))

        unsupported_ops = []

        # Convert other operation nodes.
        for n in range(len(nodesOP)):
            attr = nodesOP[n].attr
            inputs = self._get_node_input(nodesOP, nodesIn, nodesOP[n])
            try:
                if nodesOP[n].op in self.custom_ops:
                    op = self.custom_ops[nodesOP[n].op]
                elif nodesOP[n].op in bound_op_map:
                    op = bound_op_map[nodesOP[n].op]
                elif nodesOP[n].op.startswith('aten::ATen'):
                    op = globals()[f'BoundATen{attr["operator"].capitalize()}']
                elif nodesOP[n].op.startswith('onnx::'):
                    op = globals()[f'Bound{nodesOP[n].op[6:]}']
                else:
                    raise KeyError
            except (NameError, KeyError):
                unsupported_ops.append(nodesOP[n])
                logger.error('The node has an unsupported operation: %s',
                             nodesOP[n])
                continue
            attr['device'] = self.device

            # FIXME generalize
            if (nodesOP[n].op == 'onnx::BatchNormalization'
                    or getattr(op, 'TRAINING_FLAG', False)):
                # BatchNormalization node needs model.training flag to set
                # running mean and vars set training=False to avoid wrongly
                # updating running mean/vars during bound wrapper
                nodesOP[n] = nodesOP[n]._replace(bound_node=op(
                    attr, inputs, nodesOP[n].output_index, self.bound_opts,
                    False))
            else:
                nodesOP[n] = nodesOP[n]._replace(bound_node=op(
                    attr, inputs, nodesOP[n].output_index, self.bound_opts))

        if unsupported_ops:
            logger.error('Unsupported operations:')
            for n in unsupported_ops:
                logger.error(f'Name: {n.op}, Attr: {n.attr}')
            raise NotImplementedError('There are unsupported operations')

        for node in nodesIn + nodesOP:
            node.bound_node.name = node.name

        nodes_dict = {}
        for node in nodesOP + nodesIn:
            nodes_dict[node.name] = node.bound_node
        nodesOP = [n.bound_node for n in nodesOP]
        nodesIn = [n.bound_node for n in nodesIn]
        nodesOut = [nodes_dict[n] for n in nodesOut]

        return nodesOP, nodesIn, nodesOut, template

    def _build_graph(self, nodesOP, nodesIn, nodesOut, template):
        # We were assuming that the original model had only one output node.
        assert len(nodesOut) == 1
        self.final_name = nodesOut[0].name
        self.input_name, self.input_index, self.root_names = [], [], []
        self.output_name = [n.name for n in nodesOut]
        self.output_template = template
        self._modules.clear()
        for node in nodesIn:
            self.add_input_node(node, index=node.input_index)
        self.add_nodes(nodesOP)
        if self.conv_mode == 'patches':
            self.root_names: List[str] = [node.name for node in nodesIn]

    def rename_nodes(self, nodesOP, nodesIn, rename_dict):
        def rename(node):
            node.name = rename_dict[node.name]
            return node
        for i in range(len(nodesOP)):
            nodesOP[i] = rename(nodesOP[i])
        for i in range(len(nodesIn)):
            nodesIn[i] = rename(nodesIn[i])

    def _split_complex(self, nodesOP, nodesIn):
        finished = True
        for n in range(len(nodesOP)):
            if hasattr(nodesOP[n], 'complex') and nodesOP[n].complex:
                complex_node = nodesOP[n]

                finished = False
                _nodesOP, _nodesIn, _nodesOut, _ = self._convert_nodes(
                    nodesOP[n].model, nodesOP[n].input)
                # assuming each supported complex operation only has one output
                assert len(_nodesOut) == 1

                name_base = nodesOP[n].name + '/split'
                rename_dict = {}
                for node in _nodesOP + _nodesIn:
                    rename_dict[node.name] = name_base + node.name
                num_inputs = len(nodesOP[n].inputs)
                for i in range(num_inputs):
                    rename_dict[_nodesIn[i].name] = nodesOP[n].input_name[i]
                rename_dict[_nodesOP[-1].name] = nodesOP[n].name

                self.rename_nodes(_nodesOP, _nodesIn, rename_dict)

                output_name = _nodesOP[-1].name
                # Any input node of some node within the complex node should be
                # replaced with the complex node's corresponding input node.
                for node in _nodesOP:
                    for i in range(len(node.inputs)):
                        if node.input_name[i] in nodesOP[n].input_name:
                            index = nodesOP[n].input_name.index(
                                node.input_name[i])
                            node.inputs[i] = nodesOP[n].inputs[index]
                # For any output node of this complex node,
                # modify its input node.
                for node in nodesOP:
                    if output_name in node.input_name:
                        index = node.input_name.index(output_name)
                        node.inputs[index] = _nodesOP[-1]
                # Mark where the nodes come from
                for node in _nodesOP:
                    node.from_complex_node = type(complex_node).__name__

                nodesOP = nodesOP[:n] + _nodesOP + nodesOP[(n + 1):]
                nodesIn = nodesIn + _nodesIn[num_inputs:]

                break

        return nodesOP, nodesIn, finished

    def _get_node_name_map(self):
        """Build a dict with {ori_name: name, name: ori_name}"""
        self.node_name_map = {}
        for node in self.nodes():
            if isinstance(node, (BoundInput, BoundParams)):
                for p in list(node.named_parameters()):
                    if node.ori_name not in self.node_name_map:
                        name = f'{node.name}.{p[0]}'
                        self.node_name_map[node.ori_name] = name
                        self.node_name_map[name] = node.ori_name
                for p in list(node.named_buffers()):
                    if node.ori_name not in self.node_name_map:
                        name = f'{node.name}.{p[0]}'
                        self.node_name_map[node.ori_name] = name
                        self.node_name_map[name] = node.ori_name

    # convert a Pytorch model to a model with bounds
    def _convert(self, model, global_input):
        if self.verbose:
            logger.info('Converting the model...')

        if not isinstance(global_input, tuple):
            global_input = (global_input,)
        self.num_global_inputs = len(global_input)

        nodesOP, nodesIn, nodesOut, template = self._convert_nodes(
            model, global_input)
        global_input = self._to(global_input, self.device)

        while True:
            self._build_graph(nodesOP, nodesIn, nodesOut, template)
            self.forward(*global_input)  # running means/vars changed
            nodesOP, nodesIn, finished = self._split_complex(nodesOP, nodesIn)
            if finished:
                break

        self._get_node_name_map()

        ori_state_dict_mapped = OrderedDict()
        for k, v in self.ori_state_dict.items():
            if k in self.node_name_map:
                ori_state_dict_mapped[self.node_name_map[k]] = v
        self.load_state_dict(ori_state_dict_mapped)
        if self.ori_training:
            model.load_state_dict(self.ori_state_dict)
        delattr(self, 'ori_state_dict')

        # The name of the final node used in the last call to `compute_bounds`
        self.last_final_node_name = None
        self.used_nodes = []

        if self.verbose:
            logger.info('Model converted to support bounds')

    def check_prior_bounds(self, node, C=None):
        if node.prior_checked or not (node.used and node.perturbed):
            return
        if C is not None and isinstance(node, BoundConcat):
            offset = 0
            assert isinstance(C, torch.Tensor) and C.ndim == 3
            C = C.abs().sum(dim=[0, 1])
            for node_input in node.inputs:
                size = prod(node_input.output_shape[1:])
                C_s = C[offset:offset+size].sum()
                if (C_s != 0).any():
                    self.check_prior_bounds(node_input)
                offset += size
        else:
            for n in node.inputs:
                self.check_prior_bounds(n)
        tighten_input_bounds = (
            self.bound_opts['optimize_bound_args']['tighten_input_bounds']
        )
        directly_optimize_layer_names = (
            self.bound_opts['optimize_bound_args']['directly_optimize']
        )
        for i in range(len(node.inputs)):
            if (
                i in node.requires_input_bounds
                or not node.inputs[i].perturbed
                or node.inputs[i].name in self.layers_with_constraint
                # allows to tighten input bounds
                or (isinstance(node.inputs[i], BoundInput) and tighten_input_bounds)
                # layers whos optimization is forced
                # (for consecutive layers introduced as part of invprop)
                or node.inputs[i].name in directly_optimize_layer_names
            ):
                self.compute_intermediate_bounds(
                    node.inputs[i], prior_checked=True)
        node.prior_checked = True

    def compute_intermediate_bounds(self, node: Bound, prior_checked=False):
        tighten_input_bounds = (
            self.bound_opts['optimize_bound_args']['tighten_input_bounds']
        )
        directly_optimize_layer_names = (
            self.bound_opts['optimize_bound_args']['directly_optimize']
        )
        best_of_oc_and_no_oc = (
            self.bound_opts['optimize_bound_args']['best_of_oc_and_no_oc']
        )
        if (
            node.is_lower_bound_current()
            and not (
                isinstance(node, BoundInput) and tighten_input_bounds
                or node.name in directly_optimize_layer_names
            )
        ):
            if node.name in self.layers_with_constraint:
                node.clamp_interim_bounds()
            return

        logger.debug(f'Getting the bounds of {node}')

        if not prior_checked:
            self.check_prior_bounds(node)

        if not node.perturbed:
            fv = self.get_forward_value(node)
            node.interval = node.lower, node.upper = fv, fv
            return

        # FIXME check that weight perturbation is not affected
        #      (from_input=True should be set for weights)
        if not node.from_input and hasattr(node, 'forward_value'):
            node.lower = node.upper = self.get_forward_value(node)
            return

        reference_bounds = self.reference_bounds

        if self.use_forward:
            node.lower, node.upper = self.forward_general(
                node=node, concretize=True)
            return

        if self.check_IBP_intermediate(node):
            # Intermediate bounds for some operators are directly
            # computed from their input nodes by IBP
            # (such as BoundRelu, BoundNeg)
            logger.debug('IBP propagation for intermediate bounds on %s', node)
        # For the first linear layer, IBP can give the same tightness as CROWN.
        elif not self.check_IBP_first_linear(node):
            ref_intermediate = self.get_ref_intermediate_bounds(node)
            sparse_C = self.get_sparse_C(node, ref_intermediate)
            newC, reduced_dim, unstable_idx, unstable_size = sparse_C

            # Special case for BoundRelu when sparse intermediate bounds are disabled
            # Currently sparse intermediate bounds are restricted to ReLU models only
            skip = False
            if unstable_idx is None:
                if (len(node.output_name) == 1
                        and isinstance(self[node.output_name[0]], BoundTwoPieceLinear)
                        and node.name in self.reference_bounds):
                    lower, upper = self.reference_bounds[node.name]
                    fully_stable = torch.logical_or(lower>=0, upper<=0).all()
                    if fully_stable:
                        node.lower, node.upper = lower, upper
                        skip = True
            elif unstable_size == 0:
                skip = True

            if not skip:
                apply_output_constraints_to = self.bound_opts[
                    'optimize_bound_args']['apply_output_constraints_to']
                if self.return_A:
                    node.lower, node.upper, _ = self.backward_general(
                        node, newC, unstable_idx=unstable_idx,
                        apply_output_constraints_to=apply_output_constraints_to)
                else:
                    # Compute backward bounds only when there are unstable
                    # neurons, or when we don't know which neurons are unstable.
                    node.lower, node.upper = self.backward_general(
                        node, newC, unstable_idx=unstable_idx,
                        apply_output_constraints_to=apply_output_constraints_to)
                if torch.any((node.upper - node.lower).abs() > 1e10):
                    if len(apply_output_constraints_to) > 0 and not best_of_oc_and_no_oc:
                        warnings.warn('Very weak bounds detected. This can potentially be '
                            'fixed by setting best_of_oc_and_no_oc=True.')

            if reduced_dim:
                self.restore_sparse_bounds(
                    node, unstable_idx, unstable_size, ref_intermediate)

            if self.bound_opts['compare_crown_with_ibp']:
                node.lower, node.upper = self.compare_with_IBP(node, node.lower, node.upper)

        # node.lower and node.upper (intermediate bounds) are computed in
        # the above function. If we have bound references, we set them here
        # to always obtain a better set of bounds.
        if node.name in reference_bounds:
            ref_bounds = reference_bounds[node.name]
            # Initially, the reference bound and the computed bound can be
            # exactly the same when intermediate layer beta is 0. This will
            # prevent gradients flow. So we need a small guard here.
            # Set the intermediate layer bounds using reference bounds,
            # always choosing the tighter one.
            node.lower = (torch.max(ref_bounds[0], node.lower).detach()
                          - node.lower.detach() + node.lower)
            node.upper = (node.upper - (node.upper.detach()
                          - torch.min(ref_bounds[1], node.upper).detach()))
            # Otherwise, we only use reference bounds to check which neurons
            # are unstable.

        # prior constraint bounds
        if node.name in self.layers_with_constraint:
            node.clamp_interim_bounds()
        # FIXME (12/28): we should be consistent, and only use
        # node.interval, do not use node.lower or node.upper!
        node.interval = (node.lower, node.upper)

    def get_ref_intermediate_bounds(self, node):
        sparse_intermediate_bounds_with_ibp = self.bound_opts.get(
            'sparse_intermediate_bounds_with_ibp', True)
        # Sparse intermediate bounds can be enabled
        # if aux_reference_bounds are given.
        # (this is enabled for ReLU only, and not for other activations.)
        sparse_intermediate_bounds = (self.bound_opts.get(
            'sparse_intermediate_bounds', False)
            and isinstance(self[node.output_name[0]], BoundRelu))

        ref_intermediate_lb, ref_intermediate_ub = None, None
        if sparse_intermediate_bounds:
            if node.name not in self.aux_reference_bounds:
                # If aux_reference_bounds are not available,
                # we can use IBP to compute these bounds.
                if sparse_intermediate_bounds_with_ibp:
                    with torch.no_grad():
                        # Get IBP bounds for this layer;
                        # we set delete_bounds_after_use=True which does
                        # not save extra intermediate bound tensors.
                        ret_ibp = self.IBP_general(
                            node=node, delete_bounds_after_use=True)
                        ref_intermediate_lb = ret_ibp[0]
                        ref_intermediate_ub = ret_ibp[1]
                else:
                    sparse_intermediate_bounds = False
            else:
                aux_bounds = self.aux_reference_bounds[node.name]
                ref_intermediate_lb, ref_intermediate_ub = aux_bounds

        return sparse_intermediate_bounds, ref_intermediate_lb, ref_intermediate_ub

    def merge_A_dict(self, lA_dict, uA_dict):
        merged_A = {}
        for output_node_name in lA_dict:
            merged_A[output_node_name] = {}
            lA_dict_ = lA_dict[output_node_name]
            uA_dict_ = uA_dict[output_node_name]
            for input_node_name in lA_dict_:
                merged_A[output_node_name][input_node_name] = {
                    'lA': lA_dict_[input_node_name]['lA'],
                    'uA': uA_dict_[input_node_name]['uA'],
                    'lbias': lA_dict_[input_node_name]['lbias'],
                    'ubias': uA_dict_[input_node_name]['ubias'],
                }
        return merged_A

    def compute_bounds(
            self, x=None, aux=None, C=None, method='backward', IBP=False,
            forward=False, bound_lower=True, bound_upper=True, reuse_ibp=False,
            reuse_alpha=False, return_A=False, needed_A_dict=None,
            final_node_name=None, average_A=False,
            interm_bounds=None, reference_bounds=None,
            intermediate_constr=None, alpha_idx=None,
            aux_reference_bounds=None, need_A_only=False,
            cutter=None, decision_thresh=None,
            update_mask=None, ibp_nodes=None, cache_bounds=False):
        r"""Main function for computing bounds.

        Args:
            x (tuple or None): Input to the model. If it is None, the input
            from the last `forward` or `compute_bounds` call is reused.
            Otherwise: the number of elements in the tuple should be
            equal to the number of input nodes in the model, and each element in
            the tuple corresponds to the value for each input node respectively.
            It should look similar as the `global_input` argument when used for
            creating a `BoundedModule`.

            aux (object, optional): Auxliary information that can be passed to
            `Perturbation` classes for initializing and concretizing bounds,
            e.g., additional information for supporting synonym word subsitution
            perturbaiton.

            C (Tensor): The specification matrix that can map the output of the
            model with an additional linear layer. This is usually used for
            maping the logits output of the model to classification margins.

            method (str): The main method for bound computation. Choices:
                * `IBP`: purely use Interval Bound Propagation (IBP) bounds.
                * `CROWN-IBP`: use IBP to compute intermediate bounds,
                but use CROWN (backward mode LiRPA) to compute the bounds of the
                final node.
                * `CROWN`: purely use CROWN to compute bounds for intermediate
                nodes and the final node.
                * `Forward`: purely use forward mode LiRPA.
                * `Forward+Backward`: use forward mode LiRPA for intermediate
                nodes, but further use CROWN for the final node.
                * `CROWN-Optimized` or `alpha-CROWN`: use CROWN, and also
                optimize the linear relaxation parameters for activations.
                * `forward-optimized`: use forward bounds with optimized linear
                relaxation.
                * `dynamic-forward`: use dynamic forward bound propagation where
                new input variables may be dynamically introduced for
                nonlinearities.
                * `dynamic-forward+backward`: use dynamic forward mode for
                intermediate nodes, but use CROWN for the final node.

            IBP (bool, optional): If `True`, use IBP to compute the bounds of
            intermediate nodes. It can be automatically set according to
            `method`.

            forward (bool, optional): If `True`, use the forward mode bound
            propagation to compute the bounds of intermediate nodes. It can be
            automatically set according to `method`.

            bound_lower (bool, default `True`): If `True`, the lower bounds of
            the output needs to be computed.

            bound_upper (bool, default `True`): If `True`, the upper bounds of
            the output needs to be computed.

            reuse_ibp (bool, optional): If `True` and `method` is None, reuse
            the previously saved IBP bounds.

            final_node_name (str, optional): Set the final node in the
            computational graph for bound computation. By default, the final
            node of the originally built computational graph is used.

            return_A (bool, optional): If `True`, return linear coefficients
            in bound propagation (`A` tensors) with `needed_A_dict` set.

            needed_A_dict (dict, optional): A dictionary specifying linear
            coefficients (`A` tensors) that are needed and should be returned.
            Each key in the dictionary is the name of a starting node in
            backward bound propagation, with a list as the value for the key,
            which specifies the names of the ending nodes in backward bound
            propagation, and the linear coefficients of the starting node w.r.t.
            the specified ending nodes are returned. By default, it is empty.

            reuse_alpha (bool, optional): If `True`, reuse previously saved
            alpha values when they are not being optimized.

            decision_thresh (float, optional): In CROWN-optimized mode, we will
            use this decision_thresh to dynamically optimize those domains that
            <= the threshold.

            interm_bounds: A dictionary of 2-element tuple/list
            containing lower and upper bounds for intermediate layers.
            The dictionary keys should include the names of the layers whose
            bounds should be set without recomputation. The layer names can be
            viewed by setting environment variable AUTOLIRPA_DEBUG=1.
            The values of each dictionary elements are (lower_bounds,
            upper_bounds) where "lower_bounds" and "upper_bounds" are two
            tensors with the same shape as the output shape of this layer. If
            you only need to set intermediate layer bounds for certain layers,
            then just include these layers' names in the dictionary.

            reference_bounds: Format is similar to "interm_bounds".
            However, these bounds are only used as a reference, and the bounds
            for intermediate layers will still be computed (e.g., using CROWN,
            IBP or other specified methods). The computed bounds will be
            compared to "reference_bounds" and the tighter one between the two
            will be used.

            aux_reference_bounds: Format is similar to intermediate layer
            bounds. However, these bounds are only used for determine which
            neurons are stable and which neurons are unstable for ReLU networks.
            Unstable neurons' intermediate layer bounds will be recomputed.

            cache_bounds: If `True`, the currently set lower and upper bounds will not
            be deleted, but cached for use by the INVPROP algorithm. This should not be
            set by the user, but only in `_get_optimized_bounds`.

        Returns:
            bound (tuple): When `return_A` is `False`, return a tuple of
            the computed lower bound and upper bound. When `return_A`
            is `True`, return a tuple of lower bound, upper bound, and
            `A` dictionary.
        """
        # This method only prepares everything by setting all required parameters.
        # The main logic is located in `_compute_bounds_main`. It may be called
        # repeatedly for CROWN optimizations.
        logger.debug(f'Compute bounds with {method}')

        if needed_A_dict is None: needed_A_dict = {}
        if not bound_lower and not bound_upper:
            raise ValueError(
                'At least one of bound_lower and bound_upper must be True')

        # Several shortcuts.
        compute_optimized = False
        method = method.lower() if method is not None else method
        if method == 'ibp':
            # Pure IBP bounds.
            method, IBP = None, True
        elif method in ['ibp+backward', 'ibp+crown', 'crown-ibp']:
            method, IBP = 'backward', True
        elif method == 'crown':
            method = 'backward'
        elif method == 'forward':
            forward = True
            self.dynamic = False
        elif method == 'dynamic-forward':
            forward = True
            self.dynamic = True
        elif method == 'forward+backward' or method == 'forward+crown':
            method, forward = 'backward', True
        elif method == 'dynamic-forward+backward' or method == 'dynamic-forward+crown':
            self.dynamic = True
            method, forward = 'backward', True
        elif method in ['crown-optimized', 'alpha-crown', 'forward-optimized']:
            # Lower and upper bounds need two separate rounds of optimization.
            if method == 'forward-optimized':
                method = 'forward'
            else:
                method = 'backward'
            compute_optimized = True

        if reference_bounds is None:
            reference_bounds = {}
        if aux_reference_bounds is None:
            aux_reference_bounds = {}

        # If y in self.backward_node_pairs[x], then node y is visited when
        # doing backward bound propagation starting from node x.
        self.backward_from = dict([(node, []) for node in self._modules])

        if not bound_lower and not bound_upper:
            raise ValueError(
                'At least one of bound_lower and bound_upper in compute_bounds '
                'should be True')
        A_dict = {} if return_A else None

        if x is not None:
            if isinstance(x, torch.Tensor):
                x = (x,)
            self.set_input(*x, interm_bounds=interm_bounds, cache_bounds=cache_bounds)

        roots = self.roots()
        batch_size = roots[0].value.shape[0]
        dim_in = 0

        for i in range(len(roots)):
            value = roots[i].forward()
            if getattr(roots[i], 'perturbation', None) is not None:
                ret_init = roots[i].perturbation.init(
                    value, aux=aux, forward=forward)
                roots[i].linear, roots[i].center, roots[i].aux = ret_init
                # This input/parameter has perturbation.
                # Create an interval object.
                roots[i].interval = Interval(
                    roots[i].linear.lower, roots[i].linear.upper,
                    ptb=roots[i].perturbation)
                if forward:
                    roots[i].dim = roots[i].linear.lw.shape[1]
                    dim_in += roots[i].dim
            else:
                # This input/parameter does not has perturbation.
                # Use plain tuple defaulting to Linf perturbation.
                roots[i].interval = (value, value)
                roots[i].forward_value = roots[i].value = value
                roots[i].center = roots[i].lower = roots[i].upper = value

            roots[i].lower, roots[i].upper = roots[i].interval

        if forward:
            self.init_forward(roots, dim_in)

        for n in self.nodes():
            if isinstance(n, BoundRelu):
                for node in n.inputs:
                    if isinstance(node, BoundConv):
                        # whether this Conv is followed by a ReLU
                        node.relu_followed = True

            # Inject update mask inside the activations
            # update_mask: None or bool tensor([batch_size])
            # If set to a tensor, only update the alpha and beta of selected
            # element (with element=1).
            n.alpha_beta_update_mask = update_mask

        final = (self.final_node() if final_node_name is None
                 else self[final_node_name])
        # BFS to find out whether each node is used given the current final node
        self._set_used_nodes(final)

        self.use_forward = forward
        self.batch_size = batch_size
        self.dim_in = dim_in
        self.return_A = return_A
        self.A_dict = A_dict
        self.needed_A_dict = needed_A_dict
        self.intermediate_constr = intermediate_constr
        self.reference_bounds = reference_bounds
        self.aux_reference_bounds = aux_reference_bounds
        self.final_node_name = final.name
        self.ibp_nodes = ibp_nodes

        if compute_optimized:
            kwargs = dict(x=x, C=C, method=method, interm_bounds=interm_bounds,
                reference_bounds=reference_bounds, return_A=return_A,
                aux_reference_bounds=aux_reference_bounds,
                needed_A_dict=needed_A_dict,
                final_node_name=final_node_name,
                cutter=cutter, decision_thresh=decision_thresh)
            if bound_upper:
                ret2 = self._get_optimized_bounds(bound_side='upper', **kwargs)
            else:
                ret2 = None
            if bound_lower:
                ret1 = self._get_optimized_bounds(bound_side='lower', **kwargs)
            else:
                ret1 = None
            if bound_lower and bound_upper:
                if return_A:
                    # Needs to merge the A dictionary.
                    return ret1[0], ret2[1], self.merge_A_dict(ret1[2], ret2[2])
                else:
                    return ret1[0], ret2[1]
            elif bound_lower:
                return ret1  # ret1[1] is None.
            elif bound_upper:
                return ret2  # ret2[0] is None.

        return self._compute_bounds_main(C=C,
                                         method=method,
                                         IBP=IBP,
                                         bound_lower=bound_lower,
                                         bound_upper=bound_upper,
                                         reuse_ibp=reuse_ibp,
                                         reuse_alpha=reuse_alpha,
                                         average_A=average_A,
                                         alpha_idx=alpha_idx,
                                         need_A_only=need_A_only,
                                         update_mask=update_mask)

    def save_intermediate(self, save_path=None):
        r"""A function for saving intermediate bounds.

        Please call this function after `compute_bounds`, or it will output
        IBP bounds by default.

        Args:
            save_path (str, default `None`): If `None`, the intermediate bounds
            will not be saved, or it will be saved at the designated path.

        Returns:
            save_dict (dict): Return a dictionary of lower and upper bounds, with
            the key being the name of the layer.
        """
        save_dict = OrderedDict()
        for node in self.nodes():
            if node.used and node.perturbed:
                if not hasattr(node, 'interval'):
                    ibp_lower, ibp_upper = self.IBP_general(node,
                        delete_bounds_after_use=True)
                    dim_output = int(prod(node.output_shape[1:]))
                    C = torch.eye(dim_output, device=self.device).expand(
                        self.batch_size, dim_output, dim_output)
                    crown_lower, crown_upper = self.backward_general(node, C=C)
                    save_dict[node.name] = (
                        torch.max(crown_lower, ibp_lower),
                        torch.min(crown_upper, ibp_upper))
                else:
                    save_dict[node.name] = (node.lower, node.upper)

        if save_path is not None:
            torch.save(save_dict, save_path)
        return save_dict

    def _compute_bounds_main(self, C=None, method='backward', IBP=False,
            bound_lower=True, bound_upper=True, reuse_ibp=False,
            reuse_alpha=False, average_A=False, alpha_idx=None,
            need_A_only=False, update_mask=None):
        """The core implementation of compute_bounds.

        Seperated because compute_bounds may call _get_optimized_bounds which
        repeatedly calls this method. Otherwise, the preprocessing done in
        compute_bounds would be executed for each iteration.
        """

        final = (self.final_node() if self.final_node_name is None
                 else self[self.final_node_name])
        logger.debug(f'Final node {final.__class__.__name__}({final.name})')

        if IBP and method is None and reuse_ibp:
            # directly return the previously saved ibp bounds
            return self.ibp_lower, self.ibp_upper

        if IBP:
            self.ibp_lower, self.ibp_upper = self.IBP_general(node=final, C=C)

        if method is None:
            return self.ibp_lower, self.ibp_upper

        # TODO: if compute_bounds is called with a method that causes alphas to be
        # optimized, C will be allocated in each iteration. We could allocate it once
        # in compute_bounds, but e.g. `IBP_general` and code in `_get_optimized_bounds`
        # relies on the fact that it can be None
        if C is None:
            # C is an identity matrix by default
            if final.output_shape is None:
                raise ValueError(
                    f'C is not missing while node {final} has no default shape')
            dim_output = int(prod(final.output_shape[1:]))
            # TODO: use an eyeC object here.
            C = torch.eye(dim_output, device=self.device).expand(
                self.batch_size, dim_output, dim_output)

        # Reuse previously saved alpha values,
        # even if they are not optimized now
        # This must be done here instead of `compute_bounds`, as other code might change
        # it (e.g. `_get_optimized_bounds`)
        if reuse_alpha:
            self.opt_reuse()
        else:
            self.opt_no_reuse()

        for node in self.nodes():
            # All nodes may need to be recomputed
            node.prior_checked = False

        self.check_prior_bounds(final, C=C)
        if method == 'backward':
            apply_output_constraints_to = (
                self.bound_opts['optimize_bound_args']['apply_output_constraints_to']
            )
            # This is for the final output bound.
            # No need to pass in intermediate layer beta constraints.
            ret = self.backward_general(
                final, C,
                bound_lower=bound_lower, bound_upper=bound_upper,
                average_A=average_A, need_A_only=need_A_only,
                unstable_idx=alpha_idx, update_mask=update_mask,
                apply_output_constraints_to=apply_output_constraints_to)

            if self.bound_opts['compare_crown_with_ibp']:
                new_lower, new_upper = self.compare_with_IBP(final, lower=ret[0], upper=ret[1], C=C)
                ret = (new_lower, new_upper) + ret[2:]

            # FIXME when C is specified, lower and upper should not be saved to
            # final.lower and final.upper, because they are not the bounds for
            # the node.
            final.lower, final.upper = ret[0], ret[1]

            return ret
        elif method == 'forward' or method == 'dynamic-forward':
            return self.forward_general(C=C, node=final, concretize=True)
        else:
            raise NotImplementedError

    def _set_used_nodes(self, final):
        if final.name != self.last_final_node_name:
            self.last_final_node_name = final.name
            self.used_nodes = []
            for i in self.nodes():
                i.used = False
            final.used = True
            queue = deque([final])
            while len(queue) > 0:
                n = queue.popleft()
                self.used_nodes.append(n)
                for n_pre in n.inputs:
                    if not n_pre.used:
                        n_pre.used = True
                        queue.append(n_pre)
        # Based on "used" and "perturbed" properties, find out which
        # layer requires intermediate layer bounds.
        self.layers_requiring_bounds = self.get_layers_requiring_bounds()

    from .interval_bound import (
        IBP_general, _IBP_loss_fusion, check_IBP_intermediate,
        check_IBP_first_linear, compare_with_IBP)
    from .forward_bound import (
        forward_general, forward_general_dynamic, forward_refinement, init_forward)
    from .backward_bound import (
        backward_general, get_sparse_C, concretize,
        check_optimized_variable_sparsity, restore_sparse_bounds,
        get_alpha_crown_start_nodes, get_unstable_locations, batched_backward,
        _preprocess_C)
    from .output_constraints import (
        backward_general_with_output_constraint, invprop_enabled,
        backward_general_invprop, invprop_init_infeasible_bounds,
        invprop_check_infeasible_bounds)
    from .optimized_bounds import (
        _get_optimized_bounds, init_alpha, update_best_beta,
        opt_reuse, opt_no_reuse, _to_float64, _to_default_dtype)
    from .beta_crown import (beta_crown_backward_bound, reset_beta, set_beta,
                             set_beta_cuts, get_split_nodes)
    from .jacobian import (compute_jacobian_bounds, _expand_jacobian)
    from .optimize_graph import _optimize_graph
    from .edit_graph import add_nodes, add_input_node, delete_node, replace_node
    from .tools import visualize


    from .solver_module import (
        build_solver_module, _build_solver_input, _build_solver_general,
        _reset_solver_vars, _reset_solver_model)
