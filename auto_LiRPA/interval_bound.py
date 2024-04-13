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
from .bound_ops import *
from .utils import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def IBP_general(self: 'BoundedModule', node=None, C=None,
                delete_bounds_after_use=False):

    logger.debug('IBP for %s', node)

    def _delete_unused_bounds(node_list: List[Bound]):
        """Delete bounds from input layers after use to save memory. Used when
        sparse_intermediate_bounds_with_ibp is true."""
        if delete_bounds_after_use:
            for n in node_list:
                del n.interval
                n.delete_lower_and_upper_bounds()

    if self.bound_opts.get('loss_fusion', False):
        res = self._IBP_loss_fusion(node, C)
        if res is not None:
            return res

    if not node.perturbed:
        fv = self.get_forward_value(node)
        node.lower, node.upper = node.interval = (fv, fv)

    to_be_deleted_bounds = []
    if not hasattr(node, 'interval'):
        for n in node.inputs:
            if not hasattr(n, 'interval'):
                # Node n does not have interval bounds; we must compute it.
                self.IBP_general(
                    n, delete_bounds_after_use=delete_bounds_after_use)
                to_be_deleted_bounds.append(n)
        inp = [n_pre.interval for n_pre in node.inputs]
        if (C is not None and isinstance(node, BoundLinear)
                and not node.is_input_perturbed(1)):
            # merge the last BoundLinear node with the specification, available
            # when weights of this layer are not perturbed
            ret = node.interval_propagate(*inp, C=C)
            _delete_unused_bounds(to_be_deleted_bounds)
            return ret
        else:
            node.interval = node.interval_propagate(*inp)

        node.lower, node.upper = node.interval
        if isinstance(node.lower, torch.Size):
            node.lower = torch.tensor(node.lower)
            node.interval = (node.lower, node.upper)
        if isinstance(node.upper, torch.Size):
            node.upper = torch.tensor(node.upper)
            node.interval = (node.lower, node.upper)

    if C is not None:
        _delete_unused_bounds(to_be_deleted_bounds)
        return BoundLinear.interval_propagate(None, node.interval, C=C)
    else:
        _delete_unused_bounds(to_be_deleted_bounds)
        return node.interval


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
        if not hasattr(node_start, 'interval'):
            self.IBP_general(node_start)
        for n in node_gather.inputs:
            if not hasattr(n, 'interval'):
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
            lower, upper = node_start.interval
            lower, upper = lower.unsqueeze(1), upper.unsqueeze(1)
            node.lower, node.upper = node_linear.interval_propagate(
                (lower, upper), (w, w), (b.unsqueeze(1), b.unsqueeze(1)))
            node.interval = node.lower, node.upper = (
                node.lower.squeeze(1), node.upper.squeeze(1))
            return node.interval

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
            and hasattr(node.inputs[1], 'value')):
        # Node for input value.
        val_input = node.inputs[0]
        # Node for input parameter (e.g., shape, permute)
        arg_input = node.inputs[1]
        node.lower = node.forward(val_input.lower, arg_input.value)
        node.upper = node.forward(val_input.upper, arg_input.value)
        node.interval = (node.lower, node.upper)
        return True

    # Use IBP if node.ibp_intermediate == True (for nodes such as ReLU)
    nodes = []
    while (not node.is_lower_bound_current() or not node.is_upper_bound_current()):
        if not node.ibp_intermediate:
            return False
        assert len(node.inputs) == 1, (
            'Nodes with ibp_intermediate=True cannot have more than one input')
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
            node.lower, node.upper = self.IBP_general(node)
            return True

    return False
