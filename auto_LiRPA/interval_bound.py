#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
import copy

import torch
from .bound_ops import *
from .utils import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def _widen_nan_bounds(node, bounds):
    """Replace indeterminate IBP endpoints with conservative infinities."""
    if not isinstance(bounds, Interval):
        raise TypeError(
            f'{node}.interval_propagate() must return Interval, got '
            f'{type(bounds)}')
    lower, upper = bounds.lower, bounds.upper
    if isinstance(lower, torch.Size):
        lower = torch.tensor(lower)
    if isinstance(upper, torch.Size):
        upper = torch.tensor(upper)

    lower_nan = torch.isnan(lower)
    if lower_nan.any():
        print(
            f'[Interval Warning] NaN detected in lower bounds of node {node}. '
            f'Replacing with -inf.'
        )
        lower = torch.where(
            lower_nan, torch.full_like(lower, float('-inf')), lower)

    upper_nan = torch.isnan(upper)
    if upper_nan.any():
        print(
            f'[Interval Warning] NaN detected in upper bounds of node {node}. '
            f'Replacing with +inf.'
        )
        upper = torch.where(
            upper_nan, torch.full_like(upper, float('inf')), upper)

    return Interval(lower, upper)


def _interval_propagate(node, intervals, input_states, **kwargs):
    """Consume the first input's structured state, or run ordinary box IBP."""
    bound_from_state = getattr(
        node, '_bound_from_input_perturbation_state', None)
    if (input_states and input_states[0] is not None
            and bound_from_state is not None):
        return bound_from_state(
            *intervals, state=input_states[0], **kwargs)
    return node.interval_propagate(*intervals, **kwargs)


def _initialize_root_interval(self: 'BoundedModule', node, value):
    """Initialize a root's box interval and finite-p/L0 IBP state."""
    self._ibp_input_perturbation_states.pop(node, None)
    perturbation = getattr(node, 'perturbation', None)
    if perturbation is None:
        return Interval(value, value)

    lower, upper = node.linear.lower, node.linear.upper
    state = None
    if type(node) is BoundInput:
        # LiRPA keeps finite-p and L0 geometry in the perturbation object, but
        # ordinary box IBP still needs sound elementwise root bounds.
        if (isinstance(perturbation, PerturbationLpNorm)
                and perturbation.norm != torch.inf):
            eps = perturbation.eps
            if isinstance(eps, torch.Tensor) and eps.ndim < value.ndim:
                # Batched radii omit trailing input dimensions.
                eps = eps.reshape(
                    *eps.shape, *((1,) * (value.ndim - eps.ndim)))
            lower = value - eps
            upper = value + eps
            if perturbation.x_L is not None:
                lower = torch.maximum(lower, perturbation.x_L)
            if perturbation.x_U is not None:
                upper = torch.minimum(upper, perturbation.x_U)
            if perturbation.norm >= 1:
                state_perturbation = perturbation
                if (isinstance(perturbation.eps, torch.Tensor)
                        and perturbation.eps.ndim == 1
                        and value.ndim > 0
                        and perturbation.eps.shape[0] == value.shape[0]):
                    # Structured affine bounds use one radius per batch row.
                    state_perturbation = copy.copy(perturbation)
                    state_perturbation.eps = perturbation.eps.unsqueeze(-1)
                state = value, state_perturbation
        elif isinstance(perturbation, PerturbationL0Norm):
            lower = (torch.zeros_like(value) if perturbation.x_L is None
                     else perturbation.x_L)
            upper = (torch.ones_like(value) if perturbation.x_U is None
                     else perturbation.x_U)
            state = value, perturbation

    interval = Interval(lower, upper)
    if state is not None:
        self._ibp_input_perturbation_states[node] = state
    return interval


def IBP_general(self: 'BoundedModule', node=None, C=None,
                delete_bounds_after_use=False, recompute=False):

    input_perturbation_states = self._ibp_input_perturbation_states

    logger.debug('IBP for %s', node)

    def _delete_unused_bounds(node_list: List[Bound]):
        """Delete bounds from input layers after use to save memory. Used when
        sparse_intermediate_bounds_with_ibp is true."""
        if delete_bounds_after_use:
            for n in node_list:
                input_perturbation_states.pop(n, None)
                n.delete_lower_and_upper_bounds()

    if self.bound_opts.get('loss_fusion', False):
        res = self._IBP_loss_fusion(node, C)
        if res is not None:
            res = _widen_nan_bounds(node, res)
            input_perturbation_states.pop(node, None)
            node.lower = res.lower
            node.upper = res.upper
            return res

    if not node.perturbed:
        fv = self.get_forward_value(node)
        input_perturbation_states.pop(node, None)
        node.lower, node.upper = fv, fv
        node.check_bound_invariants(
            node.lower, node.upper, context='IBP output')

    to_be_deleted_bounds = []
    if (node.perturbed
            and (recompute
                 or not node.is_lower_bound_current()
                 or not node.is_upper_bound_current())):
        for n in node.inputs:
            if (not n.is_lower_bound_current()
                    or not n.is_upper_bound_current()):
                # Node n does not have interval bounds; we must compute it.
                self.IBP_general(
                    n, delete_bounds_after_use=delete_bounds_after_use)
                to_be_deleted_bounds.append(n)
        inp = [Interval(n_pre.lower, n_pre.upper) for n_pre in node.inputs]
        input_states = [
            input_perturbation_states.get(n_pre)
            for n_pre in node.inputs]
        if (C is not None and isinstance(node, BoundLinear)
                and not node.is_input_perturbed(1)):
            # merge the last BoundLinear node with the specification, available
            # when weights of this layer are not perturbed
            ret = _widen_nan_bounds(
                node, _interval_propagate(
                    node, inp, input_states, C=C))
            _delete_unused_bounds(to_be_deleted_bounds)
            return ret

        # Compute this node's box bounds first. Linear and Conv may consume a
        # preserved input perturbation state here instead of using box IBP.
        bounds = _interval_propagate(node, inp, input_states)

        # Independently determine whether that state remains valid after this
        # operator and should be made available to downstream nodes.
        propagate_state = getattr(
            node, '_propagate_input_perturbation_state', None)
        state = None
        if propagate_state is not None:
            state = propagate_state(inp, input_states)

        bounds = _widen_nan_bounds(node, bounds)
        if state is None:
            input_perturbation_states.pop(node, None)
        else:
            input_perturbation_states[node] = state
        node.lower = bounds.lower
        node.upper = bounds.upper

    if C is not None:
        _delete_unused_bounds(to_be_deleted_bounds)
        state = input_perturbation_states.get(node)
        if state is None:
            result = BoundLinear._bound_box(
                Interval(node.lower, node.upper), C)
        else:
            center, perturbation = state
            result = BoundLinear._bound_input_perturbation(
                center, perturbation, C)
        return _widen_nan_bounds(node, result)

    _delete_unused_bounds(to_be_deleted_bounds)
    return Interval(node.lower, node.upper)


def _IBP_loss_fusion(self: 'BoundedModule', node, C):
    """Merge BoundLinear, BoundGatherElements and BoundSub.

    Improvement when loss fusion is used in training.
    """
    # not using loss fusion
    if not self.bound_opts.get('loss_fusion', False):
        return None

    # Currently this function has issues in more complicated networks.
    if self.bound_opts.get('no_ibp_loss_fusion', False):
        return None

    if (C is None and isinstance(node, BoundSub)
            and isinstance(node.inputs[1], BoundGatherElements)
            and isinstance(node.inputs[0], BoundLinear)):
        node_gather = node.inputs[1]
        node_linear = node.inputs[0]
        node_start = node_linear.inputs[0]
        w = node_linear.inputs[1].param
        b = node_linear.inputs[2].param
        labels = node_gather.inputs[1]
        if (not node_start.is_lower_bound_current()
                or not node_start.is_upper_bound_current()):
            self.IBP_general(node_start)
        for n in node_gather.inputs:
            if (not n.is_lower_bound_current()
                    or not n.is_upper_bound_current()):
                self.IBP_general(n)
        if torch.isclose(labels.lower, labels.upper, 1e-8).all():
            labels = labels.lower
            batch_size = labels.shape[0]
            w = w.expand(batch_size, *w.shape)
            w = w - torch.gather(
                w, dim=1,
                index=labels.unsqueeze(-1).repeat(1, w.shape[1], w.shape[2]))
            b = b.expand(batch_size, *b.shape)
            b = b - torch.gather(b, dim=1,
                                    index=labels.repeat(1, b.shape[1]))
            lower = node_start.lower
            upper = node_start.upper
            lower, upper = lower.unsqueeze(1), upper.unsqueeze(1)
            bias = b.unsqueeze(1)
            intervals = [
                Interval(lower, upper), Interval(w, w),
                Interval(bias, bias)]
            output_interval = node_linear.interval_propagate(*intervals)
            node.lower = output_interval.lower.squeeze(1)
            node.upper = output_interval.upper.squeeze(1)
            return Interval(node.lower, node.upper)

    return None


def check_IBP_intermediate(self: 'BoundedModule', node):
    """ Check if we use IBP bounds to compute intermediate bounds on this node.

    Currently, assume all eligible operators have exactly one input.
    """
    tighten_input_bounds = (
        self.bound_opts['optimize_bound_args']['tighten_input_bounds']
    )
    directly_optimize_layer_names = (
        self.bound_opts['optimize_bound_args']['directly_optimize']
    )
    if isinstance(node, BoundInput) and tighten_input_bounds:
        return False
    if node.name in directly_optimize_layer_names:
        return False

    if self.ibp_nodes is not None and node.name in self.ibp_nodes:
        self.IBP_general(node)
        return True

    if (isinstance(node, BoundReshape)
            and node.inputs[0].is_lower_bound_current()
            and not node.inputs[1].perturbed):
        # Node for input value.
        val_input = node.inputs[0]
        # Node for input parameter (e.g., shape, permute)
        arg_input = node.inputs[1]
        argument = self.get_forward_value(arg_input)
        node.lower = node.forward(val_input.lower, argument)
        node.upper = node.forward(val_input.upper, argument)
        self._ibp_input_perturbation_states.pop(node, None)
        return True

    # Use IBP if node.ibp_intermediate == True (for nodes such as ReLU)
    nodes = []
    while (not node.is_lower_bound_current() or not node.is_upper_bound_current()):
        if not node.ibp_intermediate:
            return False
        nodes.append(node)
        node = node.inputs[0]
    nodes.reverse()
    for n in nodes:
        self.IBP_general(n)

    return True


def check_IBP_first_linear(self: 'BoundedModule', node):
    """Here we avoid creating a big C matrix in the first linear layer.
    Disable this optimization when we have beta for intermediate layer bounds.
    Disable this optimization when we need the A matrix of the first nonlinear
    layer, forcibly use CROWN to record A matrix.
    """
    tighten_input_bounds = (
        self.bound_opts['optimize_bound_args']['tighten_input_bounds']
    )
    directly_optimize_layer_names = (
        self.bound_opts['optimize_bound_args']['directly_optimize']
    )
    if isinstance(node, BoundInput) and tighten_input_bounds:
        return False
    if node.name in directly_optimize_layer_names:
        return False

    # This is the list of all intermediate layers where we need to refine.
    if self.intermediate_constr is not None:
        intermediate_beta_enabled_layers = [
            k for v in self.intermediate_constr.values() for k in v]
    else:
        intermediate_beta_enabled_layers = []

    if (node.name not in self.needed_A_dict.keys()
            and (type(node) == BoundLinear
                or type(node) == BoundConv
                and node.name not in intermediate_beta_enabled_layers)):
        if type(node.inputs[0]) == BoundInput:
            interval = self.IBP_general(node)
            node.lower, node.upper = interval.lower, interval.upper
            return True

    return False


def compare_with_IBP(self, node, lower, upper, C=None):
    """Re-compute the bounds by IBP given the existing intermediate bounds.
    Update the bounds if IBP gives tighter bounds."""

    interval = self.IBP_general(
        node, C=C, delete_bounds_after_use=True, recompute=True)
    lower_ibp, upper_ibp = interval.lower, interval.upper
    if lower is not None:
        lower = torch.max(lower, lower_ibp)
    if upper is not None:
        upper = torch.min(upper, upper_ibp)
    return lower, upper
