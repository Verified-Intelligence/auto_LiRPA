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
import torch
import warnings
from .bound_ops import *
from .utils import *
from .linear_bound import LinearBound
from .perturbations import PerturbationL0Norm, PerturbationLpNorm
from .dynamic_forward_bound import (
    _require_box_perturbation, _substitute_input_box)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule

import sys
sys.setrecursionlimit(1000000)


def _save_forward_A(self: 'BoundedModule', node, lw, uw, lower, upper):
    """Save target-node forward affine bounds for perturbed model inputs.

    Forward coefficients use the concatenated perturbation coordinates of all
    roots. Each returned entry is mapped back to one model input. Its bias
    includes the concretized contributions of all other perturbed roots, as in
    the backward return_A contract. ``lw`` and ``uw`` have shape
    ``[batch, perturbation_dim, *target_shape]``; returned coefficients have
    shape ``[batch, specification, *input_shape]``.
    """
    roots = self.roots()
    lw_flat = lw.reshape(self.batch_size, self.dim_in, -1)
    uw_flat = uw.reshape(self.batch_size, self.dim_in, -1)
    lower = lower.reshape(self.batch_size, -1)
    upper = upper.reshape(self.batch_size, -1)
    entries = {}
    requested_inputs = set(self.needed_A_dict.get(node.name, ()))
    offset = 0
    for root in roots:
        if root.perturbation is None:
            continue
        next_offset = offset + root.dim
        if type(root) is not BoundInput or root.name not in requested_inputs:
            offset = next_offset
            continue
        # Lp and L0 initialize forward variables as input coordinates or a
        # selected subset, so multiplying by this map produces input-space A.
        if not isinstance(
                root.perturbation, (PerturbationL0Norm, PerturbationLpNorm)):
            raise NotImplementedError(
                'forward return_A supports Lp and L0 input perturbations, got '
                f'{type(root.perturbation).__name__}')
        if root.lb is None or root.ub is None:
            raise RuntimeError(
                f'Missing forward contribution for root {root.name}')

        input_lw = root.linear.lw[:, offset:next_offset].reshape(
            self.batch_size, root.dim, -1)
        input_uw = root.linear.uw[:, offset:next_offset].reshape(
            self.batch_size, root.dim, -1)
        coordinate_lA = lw_flat[:, offset:next_offset].transpose(1, 2)
        coordinate_uA = uw_flat[:, offset:next_offset].transpose(1, 2)
        # BMM requires one dtype; retain the precision of every coefficient
        # tensor participating in the coordinate transformation.
        calculation_dtype = torch.promote_types(
            torch.promote_types(coordinate_lA.dtype, coordinate_uA.dtype),
            torch.promote_types(input_lw.dtype, input_uw.dtype))
        # Expand identity/selector coordinates into model-input coordinates.
        lA = coordinate_lA.to(dtype=calculation_dtype).bmm(
            input_lw.to(dtype=calculation_dtype))
        uA = coordinate_uA.to(dtype=calculation_dtype).bmm(
            input_uw.to(dtype=calculation_dtype))
        input_shape = self.get_forward_value(root).shape[1:]
        # forward_concretize temporarily records this root's contribution.
        # Removing it leaves a bias that independently concretizes all other
        # roots and the affine constant term.
        lbias = lower - root.lb.reshape(self.batch_size, -1)
        ubias = upper - root.ub.reshape(self.batch_size, -1)
        entries[root.name] = {
            'lA': lA.reshape(self.batch_size, -1, *input_shape).detach(),
            'uA': uA.reshape(self.batch_size, -1, *input_shape).detach(),
            'lbias': lbias.detach(),
            'ubias': ubias.detach(),
            'unstable_idx': None,
        }
        offset = next_offset
    self.A_dict[node.name] = entries


def forward_general(
        self: 'BoundedModule', C=None, node:'Bound'=None, concretize=False,
        offset=0, from_node=False, return_A=False):
    """Propagate forward affine bounds and optionally save input-space A."""

    if self.dynamic:
        return self.forward_general_dynamic(C=C, node=node)
    if C is None:
        if (hasattr(node, 'linear') and
                node.linear.lower is not None
                and node.linear.upper is not None
                and not return_A):
            return node.linear.lower, node.linear.upper
        # Exactness follows the active perturbations, including parameters.
        if not node.perturbed:
            value = self.get_forward_value(node)
            node.lower = node.upper = value
            if node.batch_dim == -1:
                node.linear = LinearBound(
                    None, value, None, value, value, value)
                return value, value
            # Batch-shaped fixed branches propagate zero coefficients for
            # return_A, preserving the batch and variable axes when merged.
        if node.is_lower_bound_current() and not return_A:
            node.linear = LinearBound(None, node.lower, None, node.upper, node.lower, node.upper)
            return node.lower, node.upper

    for l_pre in node.inputs:
        if not hasattr(l_pre, 'linear'):
            self.forward_general(
                node=l_pre, offset=offset, from_node=from_node,
                return_A=return_A)
        elif (return_A and l_pre.perturbed
              and (getattr(l_pre.linear, 'lw', None) is None
                   or getattr(l_pre.linear, 'uw', None) is None)):
            self.forward_general(
                node=l_pre, offset=offset, from_node=from_node,
                return_A=True)
    inp = [l_pre.linear for l_pre in node.inputs]
    node._start = '_forward'
    if (C is not None and type(node) is BoundLinear and
            not node.is_input_perturbed(1)
            and not node.is_input_perturbed(2)):
        linear = node.bound_forward(self.dim_in, *inp, C=C)
        C_merged = True
    else:
        linear = node.linear = node.bound_forward(self.dim_in, *inp)
        C_merged = False

    lw, uw = linear.lw, linear.uw
    lower, upper = linear.lb, linear.ub

    # Combine linear bounds with C matrix
    if C is not None and not C_merged:
        # FIXME use bound_forward of BoundLinear
        C_pos, C_neg = C.clamp(min=0), C.clamp(max=0)
        # Flatten lw, uw for matrix multiplication
        lw = lw.reshape(self.batch_size, self.dim_in, -1)
        uw = uw.reshape(self.batch_size, self.dim_in, -1)
        _lw = torch.matmul(lw, C_pos.transpose(-1, -2)) + torch.matmul(uw, C_neg.transpose(-1, -2))
        _uw = torch.matmul(uw, C_pos.transpose(-1, -2)) + torch.matmul(lw, C_neg.transpose(-1, -2))
        lw, uw = _lw, _uw
        # Flatten lower, upper for matrix multiplication
        lower = lower.reshape(self.batch_size, -1)
        upper = upper.reshape(self.batch_size, -1)
        _lower = ( torch.matmul(lower.unsqueeze(1), C_pos.transpose(-1, -2)) 
                    + torch.matmul(upper.unsqueeze(1), C_neg.transpose(-1, -2)) )
        _upper = ( torch.matmul(upper.unsqueeze(1), C_pos.transpose(-1, -2))
                    + torch.matmul(lower.unsqueeze(1), C_neg.transpose(-1, -2)) )
        lower, upper = _lower.squeeze(1), _upper.squeeze(1)

    logger.debug(f'Forward bounds to {node}')

    if concretize:
        # return_A exposes temporary per-root concretization contributions;
        # clear them even if input-coordinate mapping raises an exception.
        try:
            if lw is not None or uw is not None:
                lower, upper = self.concretize_bounds(
                    node=node,
                    lower=lower,
                    upper=upper,
                    concretize_mode='forward',
                    lw=lw,
                    uw=uw,
                    return_A=return_A,
                    clip_neuron_selection_value=self.clip_neuron_selection_value,
                    clip_neuron_selection_type=self.clip_neuron_selection_type
                )

            linear.lower, linear.upper = lower, upper

            if C is None:
                node.linear = linear
                node.lower, node.upper = lower, upper

            if return_A:
                requested_inputs = self.needed_A_dict.get(node.name, ())
                if requested_inputs and (lw is None or uw is None):
                    raise RuntimeError(
                        f'Forward bounds for {node.name} do not contain both '
                        'lower and upper affine coefficients')
                if requested_inputs:
                    _save_forward_A(self, node, lw, uw, lower, upper)
        finally:
            if return_A:
                for root in self.roots():
                    root.lb = root.ub = None

        if self.bound_opts['forward_refinement']:
            need_refinement = False
            for out in node.output_name:
                out_node = self[out]
                for i in getattr(out_node, 'requires_input_bounds', []):
                    if out_node.inputs[i] == node:
                        need_refinement = True
                        break
            if need_refinement:
                self.forward_refinement(node)
        return lower, upper


def clean_memory(self: 'BoundedModule', node):
    """ Remove linear bounds that are no longer needed. """
    # TODO add an option to retain these bounds
    for inp in node.inputs:
        if hasattr(inp, 'linear') and inp.linear is not None:
            clean = True
            for out in inp.output_name:
                out_node = self[out]
                if not (hasattr(out_node, 'linear') and out_node.linear is not None):
                    clean = False
            if clean:
                if isinstance(inp.linear, tuple):
                    for item in inp.linear:
                        del item
                delattr(inp, 'linear')


def forward_refinement(self: 'BoundedModule', node):
    """ Refine forward bounds with backward bound propagation
    (only refine unstable positions). """
    unstable_size_before = torch.logical_and(node.lower < 0, node.upper > 0).sum()
    if unstable_size_before == 0:
        return
    unstable_idx, unstable_size = self.get_unstable_locations(
        node.lower, node.upper, conv=isinstance(node, BoundConv))
    logger.debug(f'Forward refinement for {node}')
    batch_size = node.lower.shape[0]
    ret = self.batched_backward(
        node, C=None, unstable_idx=unstable_idx, batch_size=batch_size)
    self.restore_sparse_bounds(
        node, unstable_idx, unstable_size, node.lower, node.upper,
        new_lower=ret[0], new_upper=ret[1])
    unstable_size_after = torch.logical_and(node.lower < 0, node.upper > 0).sum()
    logger.debug(f'  Unstable neurons: {unstable_size_before} -> {unstable_size_after}')
    # TODO also update linear bounds?


def init_forward(self: 'BoundedModule', roots, dim_in):
    if dim_in == 0:
        raise ValueError("At least one node should have a specified perturbation")
    if self.dynamic:
        # The variable layout below is rebuilt here, so linear bounds laid out
        # over the previous one have to go. compute_bounds(x=...) already
        # cleared them; compute_bounds(x=None) reuses the input and does not.
        root_names = {root.name for root in roots}
        for node in self.nodes():
            if node.name not in root_names and hasattr(node, 'linear'):
                delattr(node, 'linear')
    prev_dim_in = 0
    # Assumption: roots[0] is the input node which implies batch_size
    batch_size = self.get_forward_value(roots[0]).shape[0]
    for i in range(len(roots)):
        if hasattr(roots[i], 'perturbation') and roots[i].perturbation is not None:
            shape = roots[i].linear.lw.shape
            if self.dynamic:
                if shape[1] != dim_in:
                    raise NotImplementedError('Dynamic forward bound is not supported yet when there are multiple perturbed inputs.')
                ptb = roots[i].perturbation
                _require_box_perturbation(ptb)
                _substitute_input_box(roots[i].linear, ptb)
                # Which node owns which coefficient columns, seeded with the
                # input's own range so a node takes its range after it. One
                # layout serves one propagation.
                roots[i].linear.dynamic_variable_ranges = {
                    roots[i].name: (0, dim_in)}
            else:
                lw = torch.zeros(shape[0], dim_in, *shape[2:]).to(roots[i].linear.lw)
                lw[:, prev_dim_in:(prev_dim_in+shape[1])] = roots[i].linear.lw
                if roots[i].linear.lw.data_ptr() == roots[i].linear.uw.data_ptr():
                    uw = lw
                else:
                    uw = torch.zeros(shape[0], dim_in, *shape[2:]).to(roots[i].linear.uw)
                    uw[:, prev_dim_in:(prev_dim_in+shape[1])] = roots[i].linear.uw
                roots[i].linear.lw = lw
                roots[i].linear.uw = uw
            if i >= self.num_global_inputs:
                roots[i].forward_value = roots[i].forward_value.unsqueeze(0).repeat(
                    *([batch_size] + [1] * self.forward_value.ndim))
            prev_dim_in += shape[1]
        else:
            b = fv = roots[i].forward_value
            shape = fv.shape
            if roots[i].batch_dim != -1:
                w = torch.zeros(
                    shape[0], dim_in, *shape[1:], dtype=fv.dtype,
                    device=self.device)
                warnings.warn(f'Creating a LinearBound with zero weights with shape {w.shape}')
            else:
                w = None
            roots[i].linear = LinearBound(w, b, w, b, b, b)
            roots[i].lower = roots[i].upper = b
