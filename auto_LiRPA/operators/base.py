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
""" Base class and functions for implementing bound operators"""
from typing import Optional, List
import warnings
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from ..perturbations import *
from ..utils import *
from ..patches import *
from ..linear_bound import LinearBound

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

epsilon = 1e-12


def not_implemented_op(node, func):
    message = (
        f'Function `{func}` of `{node}` is not supported yet.'
        ' Please help to open an issue at https://github.com/Verified-Intelligence/auto_LiRPA'
        ' or implement this function in auto_LiRPA/bound_ops.py'
        ' or auto_LiRPA/operators by yourself.')
    raise NotImplementedError(message)


class Interval(tuple):
    """Interval object for interval bound propagation."""

    # Subclassing tuple object so that all previous code can be reused.
    def __new__(self, lb=None, ub=None, ptb=None):
        return tuple.__new__(Interval, (lb, ub))

    def __init__(self, lb, ub, ptb=None):
        if ptb is None:
            self.ptb = None
            # `self.ptb == None` means that this interval
            # is not perturbed and it shall be treated as a constant and lb = ub.
            # To avoid mistakes, in this case the caller must make sure lb and ub are the same object.
            assert lb is ub
        else:
            if not isinstance(ptb, Perturbation):
                raise ValueError("ptb must be a Perturbation object or None. Got type {}".format(type(ptb)))
            else:
                self.ptb = ptb

    def __str__(self):
        return "({}, {}) with ptb={}".format(self[0], self[1], self.ptb)

    def __repr__(self):
        return "Interval(lb={}, ub={}, ptb={})".format(self[0], self[1], self.ptb)

    @staticmethod
    def make_interval(lb, ub, other=None):
        """Checking if the other interval is tuple, keep the perturbation."""
        if isinstance(other, Interval):
            return Interval(lb, ub, ptb=other.ptb)
        else:
            return (lb, ub)

    @staticmethod
    def get_perturbation(interval):
        """Given a tuple or Interval object, returns the norm and eps."""
        if isinstance(interval, Interval) and interval.ptb is not None:
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            elif isinstance(interval.ptb, PerturbationSynonym):
                return torch.inf, 1.0
            elif isinstance(interval.ptb, PerturbationL0Norm):
                return 0, interval.ptb.eps, interval.ptb.ratio
            else:
                raise RuntimeError("get_perturbation() does not know how to handle {}".format(type(interval.ptb)))
        else:
            # Tuple object. Assuming L infinity norm lower and upper bounds.
            return torch.inf, np.nan


    @staticmethod
    def is_perturbed(interval):
        """Checking if a Interval or tuple object has perturbation enabled."""
        if isinstance(interval, Interval) and interval.ptb is None:
            return False
        else:
            return True


class Bound(nn.Module):
    r"""
    Base class for supporting the bound computation of an operator. Please see examples
    at `auto_LiRPA/operators`.

    Args:
        attr (dict): Attributes of the operator.

        inputs (list): A list of input nodes.

        output_index (int): The index in the output if the operator has multiple outputs. Usually output_index=0.

        options (dict): Bound options.

    Be sure to run `super().__init__(attr, inputs, output_index, options, device)`
    first in the `__init__` function.
    """

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__()
        attr = {} if attr is None else attr
        inputs = [] if inputs is None else inputs
        options = {} if options is None else options
        self.name: Optional[str] = None
        self.output_name = []
        self.device = attr.get('device')
        self.attr = attr
        self.inputs: List['Bound'] = inputs
        self.output_index = output_index
        self.options = options
        self.forward_value = None
        self.output_shape = None
        self.from_input = False
        self.bounded = False
        self.IBP_rets = None
        self.requires_input_bounds = []
        # If True, when building the Jacobian graph, this node should be treated
        # as a constant and there is no need to further propagate Jacobian.
        self.no_jacobian = False
        # If True, when we are computing intermediate bounds for these ops,
        # we simply use IBP to propagate bounds from its input nodes
        # instead of CROWN. Currently only operators with a single input can be
        # supported.
        self.ibp_intermediate = False
        self.splittable = False
        # Determine if this node has a perturbed output or not. The function BoundedModule._mark_perturbed_nodes() will set this property.
        self.perturbed = False
        self.never_perturbed = False
        if options is not None and 'loss_fusion' in options:
            self.loss_fusion = options['loss_fusion']
        else:
            self.loss_fusion = False
        self.options = options
        # Use `default_interval_propagate`
        self.use_default_ibp = False
        # If set to true, the backward bound output of this node is 0.
        self.zero_backward_coeffs_l = False
        self.zero_backward_coeffs_u = False
        # If set to true, the A matrix accumulated on this node is 0.
        self.zero_lA_mtx = False
        self.zero_uA_mtx = False
        self.patches_start = False
        self.alpha_beta_update_mask = None
        self.is_final_node = False
        # By default, we assue this node has no batch dimension.
        # It will be updated in BoundedModule.get_forward_value().
        self.batch_dim = -1

        # The .lower and .upper properties are written to as part of the bound propagation.
        # Usually, in iterative refinement, each bound only depends on bounds previously
        # computed in the same iteration. However, this changes if INVPROP is used to incorporate
        # output constraints. Then, we also need bounds of layers *after* the currently bounded
        # layer. Therefore, we have to cache the older bounds.
        self._is_lower_bound_current = False
        self._lower = None
        self._is_upper_bound_current = False
        self._upper = None

    def __repr__(self, attrs=None):
        inputs = ', '.join([node.name for node in self.inputs])
        ret = (f'{self.__class__.__name__}(name={self.name}, '
                f'inputs=[{inputs}], perturbed={self.perturbed}')
        if attrs is not None:
            for k, v in attrs.items():
                ret += f', {k}={v}'
        ret += ')'
        return ret

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        if not (value is None or isinstance(value, torch.Tensor)):
            raise TypeError(f'lower must be a tensor or None, got {type(value)}')
        if value is None:
            self._is_lower_bound_current = False
        else:
            self._is_lower_bound_current = True
        self._lower = value

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, value):
        if not (value is None or isinstance(value, torch.Tensor)):
            raise TypeError(f'upper must be a tensor or None, got {type(value)}')
        if value is None:
            self._is_upper_bound_current = False
        else:
            self._is_upper_bound_current = True
        self._upper = value

    def move_lower_and_upper_bounds_to_cache(self):
        if self._lower is not None:
            self._lower = self._lower.detach().requires_grad_(False)
            self._is_lower_bound_current = False
        if self._upper is not None:
            self._upper = self._upper.detach().requires_grad_(False)
            self._is_upper_bound_current = False

    def delete_lower_and_upper_bounds(self):
        self._lower = None
        self._upper = None
        self._is_lower_bound_current = False
        self._is_upper_bound_current = False

    def is_lower_bound_current(self):
        return self._is_lower_bound_current

    def is_upper_bound_current(self):
        return self._is_upper_bound_current

    def are_output_constraints_activated_for_layer(
        self: 'Bound',
        apply_output_constraints_to: Optional[List[str]],
    ):
        if self.is_final_node:
            return False
        if apply_output_constraints_to is None:
            return False
        for layer_type_or_name in apply_output_constraints_to:
            if layer_type_or_name.startswith('/'):
                if self.name == layer_type_or_name:
                    return True
            else:
                assert layer_type_or_name.startswith('Bound'), (
                    'To apply output constraints to tighten layer bounds, pass either the layer name '
                    '(starting with "/", e.g. "/input.7") or the layer type (starting with "Bound", '
                    'e.g. "BoundLinear")'
                )
                if type(self).__name__ == layer_type_or_name:
                    return True
        return False

    def init_gammas(self, num_constraints):
        if not self.are_output_constraints_activated_for_layer(
            self.options.get('optimize_bound_args', {}).get('apply_output_constraints_to', [])
        ):
            return
        assert len(self.output_shape) > 0, self
        neurons_in_this_layer = 1
        for d in self.output_shape[1:]:
            neurons_in_this_layer *= d
        init_gamma_value = 0.0
        # We need a different number of gammas depending on whether or not they are shared
        # However, to the code outside of this class, this should be transparent.
        # We create the correct number of gammas in gammas_underlying_tensor and if necessary
        # expand it to simulate a larger tensor. This is just a view, no additional memory is created.
        # By the outside, only .gammas should be used. However, we must take care to update this view
        # whenever gammas_underlying_tensor was changed (see clip_gammas)
        # Note that _set_gammas in optimized_bounds.py needs to refer to the gammas_underlying_tensor,
        # because that's the leaf tensor for which we need to compute gradients.
        if self.options.get('optimize_bound_args', {}).get('share_gammas', False):
            self.gammas_underlying_tensor = torch.full((2, num_constraints, 1), init_gamma_value, requires_grad=True, device=self.device)
            self.gammas = self.gammas_underlying_tensor.expand(-1, -1, neurons_in_this_layer)
        else:
            self.gammas_underlying_tensor = torch.full((2, num_constraints, neurons_in_this_layer), init_gamma_value, requires_grad=True, device=self.device)
            self.gammas = self.gammas_underlying_tensor

    def clip_gammas(self):
        if not hasattr(self, "gammas"):
            return
        self.gammas_underlying_tensor.data = torch.clamp(self.gammas_underlying_tensor.data, min=0.0)

        # If gammas are shared, self.gammas != self.gammas_underlying_tensor
        # We've changed self.gammas_underlying_tensor, those changes must be propagated to self.gammas
        neurons_in_this_layer = 1
        for d in self.output_shape[1:]:
            neurons_in_this_layer *= d
        if self.options.get('optimize_bound_args', {}).get('share_gammas', False):
            self.gammas = self.gammas_underlying_tensor.expand(-1, -1, neurons_in_this_layer)

    def is_input_perturbed(self, i=0):
        r"""Check if the i-th input is with perturbation or not."""
        return i < len(self.inputs) and self.inputs[i].perturbed

    def clear(self):
        """ Clear attributes when there is a new input to the network"""
        pass

    @property
    def input_name(self):
        return [node.name for node in self.inputs]

    def forward(self, *x):
        r"""
        Function for standard/clean forward.

        Args:
            x: A list of input values. The length of the list is equal to the number of input nodes.

        Returns:
            output (Tensor): The standard/clean output of this node.
        """
        return not_implemented_op(self, 'forward')

    def interval_propagate(self, *v):
        r"""
        Function for interval bound propagation (IBP) computation.

        There is a default function `self.default_interval_propagate(*v)` in the base class,
        which can be used if the operator is *monotonic*. To use it, set `self.use_default_ibp = True`
        in the `__init__` function, and the implementation of this function can be skipped.

        Args:
            v: A list of the interval bound of input nodes.
            Generally, for each element `v[i]`, `v[i][0]` is the lower interval bound,
            and `v[i][1]` is the upper interval bound.

        Returns:
            bound: The interval bound of this node, in a same format as v[i].
        """
        if self.use_default_ibp or self.never_perturbed:
            return self.default_interval_propagate(*v)
        else:
            return not_implemented_op(self, 'interval_propagate')

    def default_interval_propagate(self, *v):
        """Default IBP using the forward function.

        For unary monotonous functions or functions for altering shapes only
        but not values.
        """
        if len(v) == 0:
            return Interval.make_interval(self.forward(), self.forward())
        else:
            if len(v) > 1:
                for i in range(1, len(v)):
                    assert not self.is_input_perturbed(i)
            return Interval.make_interval(
                self.forward(v[0][0], *[vv[0] for vv in v[1:]]),
                self.forward(v[0][1], *[vv[0] for vv in v[1:]]), v[0])

    def bound_forward(self, dim_in, *x):
        r"""
        Function for forward mode bound propagation.

        Forward mode LiRPA computs a `LinearBound`
        instance representing the linear bound for each involved node.
        Major attributes of `LinearBound` include
        `lw`, `uw`, `lb`, `ub`, `lower`, and `upper`.

        `lw` and `uw` are coefficients of linear bounds w.r.t. model input.
        Their shape is `(batch_size, dim_in, *standard_shape)`,
        where `dim_in` is the total dimension of perturbed input nodes of the model,
        and `standard_shape` is the shape of the standard/clean output.
        `lb` and `ub` are bias terms of linear bounds, and their shape is equal
        to the shape of standard/clean output.
        `lower` and `upper` are concretized lower and upper bounds that will be
        computed later in BoundedModule.

        Args:
            dim_in (int): Total dimension of perturbed input nodes of the model.

            x: A list of the linear bound of input nodes. Each element in x is a `LinearBound` instance.

        Returns:
            bound (LinearBound): The linear bound of this node.
        """
        return not_implemented_op(self, 'bound_forward')

    def bound_dynamic_forward(self, *x, max_dim=None, offset=0):
        raise NotImplementedError(f'bound_dynamic_forward is not implemented for {self}.')

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        r"""
        Function for backward mode bound propagation.

        Args:
            last_lA (Tensor): `A` matrix for lower bound computation propagated to this node. It can be `None` if lower bound is not needed.

            last_uA (Tensor): `A` matrix for upper bound computation propagated to this node. It can be `None` if upper bound is not needed.

            x: A list of input nodes, with x[i].lower and x[i].upper that can be used as pre-activation bounds.

        Returns:
            A: A list of A matrices for the input nodes. Each element is a tuple (lA, uA).

            lbias (Tensor): The bias term for lower bound computation, introduced by the linear relaxation of this node. .

            ubias (Tensor): The bias term for upper bound computation, introduced by the linear relaxation of this node.
        """
        return not_implemented_op(self, 'bound_backward')

    def broadcast_backward(self, A, x):
        """
        Adjust shape of A, adding or removing broadcast dimensions, based on the other operand x.

        Typically, A has [spec, batch, ...].
        The other operand x may have shape [batch, ...], or no batch dimension.
        Here the "..." dimensions may be different.
        We need to make sure the two match, by adding or removing dimensions in A.
        """
        shape = x.output_shape

        if isinstance(A, Tensor):
            if x.batch_dim == -1:
                # The other operand has no batch dimension. (e.g., constants).
                # Add batch dimension to it.
                shape = torch.Size([A.shape[1]] + list(shape))
                dims = []
                cnt_sum = A.ndim - len(shape) - 1
                for i in range(2, A.ndim): # merge the output dimensions?
                    if cnt_sum > 0:
                        dims.append(i)
                        cnt_sum -= 1
                if dims:
                    A = torch.sum(A, dim=dims)
            else:
                dims = list(range(1, A.ndim - len(shape)))
                if dims:
                    A = torch.sum(A, dim=dims)
            dims = []
            for i in range(1, len(shape)):
                # Skip the batch dimension.
                # FIXME (05/11/2022): the following condition is not always correct.
                # We should not rely on checking dimension is "1" or not.
                if shape[i] == 1 and A.shape[i + 1] != 1:
                    dims.append(i + 1)
            if dims:
                A = torch.sum(A, dim=dims, keepdim=True)
            # Check the final shape - it should be compatible.
            assert A.shape[2:] == shape[1:]  # skip the spec and batch dimension.
        else:
            pass
        return A

    def build_gradient_node(self, grad_upstream):
        r"""
        Function for building the gradient node to bound the Jacobian.

        Args:
            grad_upstream: Upstream gradient in the gradient back-propagation.

        Returns:
            A list. Each item contains the following for computing the gradient
            of each input:
                module_grad (torch.nn.Module): Gradient node.

                grad_input (list): Inputs to the gradient node. Values do not
                matter. We only want the shapes.

                grad_extra_nodes (list): Extra nodes needed for the gradient.
        """
        return not_implemented_op(self, 'build_gradient_node')

    def get_bias(self, A, bias):
        if A is None:
            return 0
        if not Benchmarking:
            assert not isnan(A)
            assert not isnan(bias)
        if torch.isinf(bias).any():
            warnings.warn('There is an inf value in the bias of LiRPA bounds.')

        if isinstance(A, Tensor):
            if self.batch_dim != -1:
                bias_new = torch.einsum('sb...,b...->sb', A, bias)
            else:
                bias_new = torch.einsum('sb...,...->sb', A, bias)
            if isnan(bias_new):
                # NaN can be caused by 0 * inf, if 0 appears in `A` and inf appears in `bias`.
                # Force the whole bias to be 0, to avoid gradient issues.
                # FIXME maybe find a more robust solution.
                return 0
            else:
                # FIXME (09/17): handle the case for pieces.unstable_idx.
                return bias_new
        elif isinstance(A, eyeC):
            batch_size = A.shape[1]
            if self.batch_dim != -1:
                return bias.reshape(batch_size, -1).t()
            else:
                return bias.reshape(-1).unsqueeze(-1).repeat(1, batch_size)
        elif type(A) == Patches:
            # the shape of A.patches is [batch, L, out_c, in_c, K, K]
            if self.batch_dim != -1:
                # Input A patches has shape (spec, batch, out_h, out_w, in_c, H, W) or (unstable_size, batch, in_c, H, W).
                patches = A.patches
                # Here the size of bias is [batch_size, out_h, out_w, in_c, H, W]
                bias = inplace_unfold(bias, kernel_size=A.patches.shape[-2:], stride=A.stride, padding=A.padding, inserted_zeros=A.inserted_zeros, output_padding=A.output_padding)
                if A.unstable_idx is not None:
                    # Sparse bias has shape [unstable_size, batch_size, in_c, H, W]. No need to select over the out_c dimension.
                    bias = bias[:, A.unstable_idx[1], A.unstable_idx[2]]
                    # bias_new has shape (unstable_size, batch).
                    bias_new = torch.einsum('bschw,sbchw->sb', bias, patches)
                else:
                    # Sum over the in_c, H, W dimension. Use torch.einsum() to save memory, equal to:
                    # bias_new = (bias * patches).sum(-1,-2,-3).transpose(-2, -1)
                    # bias_new has shape (spec, batch, out_h, out_w).
                    bias_new = torch.einsum('bijchw,sbijchw->sbij', bias, patches)
            else:
                # Similar to BoundConstant. (BoundConstant does not have batch_dim).
                # FIXME (09/16): bias size is different for BoundConstant. We should use the same size!
                patches = A.patches
                bias_new = torch.sum(patches, dim=(-1, -2, -3)) * bias.to(self.device)
                # Return shape is (spec, batch, out_h, out_w) or (unstable_size, batch).
                return bias_new
            return bias_new
        else:
            return NotImplementedError()

    def make_axis_non_negative(self, axis, shape='input'):
        if isinstance(axis, (tuple, list)):
            return tuple([self.make_axis_non_negative(item, shape)
                          for item in axis])
        if shape == 'input':
            shape = self.input_shape
        elif shape == 'output':
            shape = self.output_shape
        else:
            assert isinstance(shape, torch.Size)
        if axis < 0:
            return axis + len(shape)
        else:
            return axis

    def update_requires_input_bounds(self):
        """Update requires_input_bounds.

        This function is called once we know if the input nodesare perturbed.
        """
        pass

    def clamp_interim_bounds(self):
        """Clamp intermediate bounds."""
        pass

    def check_constraint_available(self, node, flag=False):
        if hasattr(node, 'cstr_interval'):
            flag = True
        for n in node.inputs:
            if not n.from_input:
                flag = flag or self.check_constraint_available(n, flag)
        return flag

    def _ibp_constraint(self, node: 'Bound', delete_bounds_after_use=False):
        def _delete_unused_bounds(node_list):
            """Delete bounds from input layers after use to save memory. Used when
            sparse_intermediate_bounds_with_ibp is true."""
            if delete_bounds_after_use:
                for n in node_list:
                    del n.cstr_interval
                    del n.cstr_lower
                    del n.cstr_upper

        if not node.perturbed and hasattr(node, 'forward_value'):
            node.cstr_lower, node.cstr_upper = node.cstr_interval = (
                node.forward_value, node.forward_value)

        to_be_deleted_bounds = []
        if not hasattr(node, 'cstr_interval'):
            for n in node.inputs:
                if not hasattr(n, 'cstr_interval'):
                    # Node n does not have interval bounds; we must compute it.
                    self._ibp_constraint(
                        n, delete_bounds_after_use=delete_bounds_after_use)
                    to_be_deleted_bounds.append(n)
            inp = [n_pre.cstr_interval for n_pre in node.inputs]
            node.cstr_interval = node.interval_propagate(*inp)

            node.cstr_lower, node.cstr_upper = node.cstr_interval
            if isinstance(node.cstr_lower, torch.Size):
                node.cstr_lower = torch.tensor(node.cstr_lower)
                node.cstr_interval = (node.cstr_lower, node.cstr_upper)
            if isinstance(node.cstr_upper, torch.Size):
                node.cstr_upper = torch.tensor(node.cstr_upper)
                node.cstr_interval = (node.cstr_lower, node.cstr_upper)

        if node.is_lower_bound_current():
            node.lower = torch.where(node.lower >= node.cstr_lower, node.lower,
                            node.cstr_lower)
            node.upper = torch.where(node.upper <= node.cstr_upper, node.upper,
                            node.cstr_upper)
            node.interval = (node.lower, node.upper)

        _delete_unused_bounds(to_be_deleted_bounds)
        return node.cstr_interval

    def _check_weight_perturbation(self):
        weight_perturbation = False
        for n in self.inputs[1:]:
            if hasattr(n, 'perturbation'):
                if n.perturbation is not None:
                    weight_perturbation = True
        if weight_perturbation:
            self.requires_input_bounds = list(range(len(self.inputs)))
        else:
            self.requires_input_bounds = []
        return weight_perturbation

    def non_deter_wrapper(self, op, *args, **kwargs):
        """Some operations are non-deterministic and deterministic mode will fail.
        So we temporary disable it."""
        if self.options.get('deterministic', False):
            torch.use_deterministic_algorithms(False)
        ret = op(*args, **kwargs)
        if self.options.get('deterministic', False):
            torch.use_deterministic_algorithms(True)
        return ret

    def non_deter_scatter_add(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.scatter_add, *args, **kwargs)

    def non_deter_index_select(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.index_select, *args, **kwargs)
