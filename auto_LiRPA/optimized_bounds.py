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
import time
import os
from collections import OrderedDict
from contextlib import contextmanager, ExitStack

import torch
from torch import optim, Tensor
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
from .beta_crown import print_optimized_beta
from .utils import reduction_sum, multi_spec_keep_func_all, clone_sub_A_dict
from .opt_pruner import OptPruner
from .perturbations import PerturbationLpNorm
if TYPE_CHECKING:
    from .bound_general import BoundedModule


default_optimize_bound_args = {
    'enable_alpha_crown': True,  # Enable optimization of alpha.
    'enable_beta_crown': False,  # Enable beta split constraint.

    'apply_output_constraints_to': [],  # Enable optimization w.r.t. output constraints.
    'tighten_input_bounds': False,  # Don't tighten input bounds
    # If output constraints are activated, use only bounds computed with them.
    'best_of_oc_and_no_oc': False,
    'directly_optimize': [],  # No layer should be directly optimized
    'oc_lr': 0.1,  # learning rate for dualized output constraints
    'share_gammas': False,

    'iteration': 20,  # Number of alpha/beta optimization iterations.
    # Share some alpha variables to save memory at the cost of slightly
    # looser bounds.
    'use_shared_alpha': False,  # 'network': also share across start nodes and sides.
    # Optimizer used for alpha and beta optimization.
    'optimizer': 'adam',
    # Save best results of alpha/beta/bounds during optimization.
    'keep_best': True,
    # Only optimize bounds of last layer during alpha/beta CROWN.
    'fix_interm_bounds': True,
    # Learning rate for the optimizable parameter alpha in alpha-CROWN.
    'lr_alpha': 0.5,
    # Learning rate for the optimizable parameter beta in beta-CROWN.
    'lr_beta': 0.05,
    'lr_cut_beta': 5e-3,  # Learning rate for optimizing cut betas.
    # Initial alpha variables by calling CROWN once.
    'init_alpha': True,
    'lr_coeffs': 0.01,  # Learning rate for coeffs for refinement
    # Layers to be refined, separated by commas.
    # -1 means preactivation before last activation.
    'intermediate_refinement_layers': [-1],
    # When batch size is not 1, this reduction function is applied to
    # reduce the bounds into a scalar.
    'loss_reduction_func': reduction_sum,
    # Criteria function of early stop.
    'stop_criterion_func': lambda x: False,
    # Learning rate decay factor during bounds optimization.
    'lr_decay': 0.98,
    # Number of iterations that we will start considering early stop
    # if tracking no improvement.
    'early_stop_patience': 10,
    # Start to save optimized best bounds
    # when current_iteration > int(iteration*start_save_best)
    'start_save_best': 0.5,
    # Prune verified domain within iteration.
    'pruning_in_iteration': False,
    # Percentage of the minimum domains that can apply pruning.
    'pruning_in_iteration_threshold': 0.2,
    # For specification that will output multiple bounds for one
    # property, we use this function to prune them.
    'multi_spec_keep_func': multi_spec_keep_func_all,
    # Use the newly fixed loss function. By default, it is set to False
    # for compatibility with existing use cases.
    # Ensure that returned optimization parameters reproduce returned bounds.
    'reproduce_alpha_bounds': False,
    'max_time': 1e9,

    # Precision ('fp16', 'bf16', 'tf32', 'fp16+tf32', 'bf16+tf32', or None)
    # for the intermediate optimization iterations, while the first and last
    # iterations always run in full precision.
    'mixed_precision_dtype': None,
    # Prune domains during low-precision iterations too, as speculative
    # prunes that are rolled back and re-proved in full precision before
    # each full-precision iteration.
    'low_precision_prune_in_iter': True,
    # Only speculatively prune a domain if its low-precision bound clears
    # the decision threshold by this absolute margin (default suits TF32
    # bound error).
    'low_precision_prune_margin_abs': 0.01,
}


def opt_reuse(self: 'BoundedModule'):
    for node in self.get_enabled_opt_act():
        node.opt_reuse()


def opt_no_reuse(self: 'BoundedModule'):
    for node in self.get_enabled_opt_act():
        node.opt_no_reuse()


def _set_alpha(optimizable_activations, parameters, alphas, lr):
    """Set best_alphas, alphas and parameters list."""
    for node in optimizable_activations:
        alphas.extend(list(node.alpha.values()))
        node.opt_start()
    # Alpha has shape (2, output_shape, batch_dim, node_shape)
    parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
    # best_alpha is a dictionary of dictionary. Each key is the alpha variable
    # for one activation layer, and each value is a dictionary contains all
    # activation layers after that layer as keys.
    best_alphas = OrderedDict()
    for m in optimizable_activations:
        best_alphas[m.name] = {}
        for alpha_m in m.alpha:
            best_alphas[m.name][alpha_m] = m.alpha[alpha_m].detach().clone()
            # We will directly replace the dictionary for each activation layer after
            # optimization, so the saved alpha might not have require_grad=True.
            m.alpha[alpha_m].requires_grad_()

    return best_alphas


def _set_gammas(nodes, parameters):
    """
    Adds gammas to parameters list
    """
    gammas = []
    gamma_lr = 0.1
    for node in nodes:
        if hasattr(node, 'gammas'):
            gammas.append(node.gammas_underlying_tensor)
            # The learning rate is the same for all layers
            gamma_lr = node.options['optimize_bound_args']['oc_lr']
    parameters.append({'params': gammas, 'lr': gamma_lr})


def _save_ret_first_time(bounds, best_ret):
    """Save results at the first iteration to best_ret."""
    if bounds is not None:
        best_ret.append(bounds.detach().clone())
    else:
        best_ret.append(None)


def _get_idx_mask(idx: int, full_ret_bound: Tensor, best_ret_bound: Tensor, loss_reduction_func
                  ) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Get index for improved elements.
    :param idx:                 0 := updating the lower bound, 1 := updating the upper bound
    :param full_ret_bound:      Lower/upper bound results for this iteration
    :param best_ret_bound:      The best lower/upper bound results seen thus far
    :param loss_reduction_func: Loss reduction function that reduces the losses to just the batch
                                dimension.
    :return:
            idx_mask:           A mask on the batch dimension where the mask is true if a
                                sub-problem has seen loss improvement.
            improved_idx:       A Tensor of the indices in the batch dimension that have seen loss
                                improvement.
    """
    assert idx in (0, 1), 'idx must be 0 (lower bound) or 1 (upper bound)'
    reduced_full = loss_reduction_func(full_ret_bound)
    reduced_best = loss_reduction_func(best_ret_bound)
    idx_mask = (reduced_full > reduced_best) if idx == 0 else (reduced_full < reduced_best)
    idx_mask = idx_mask.view(-1)

    improved_idx = idx_mask.nonzero(as_tuple=True)[0] if idx_mask.any() else None
    return idx_mask, improved_idx

def _determine_improve(
    full_ret: Dict[str, Dict[str, Dict[str, Union[Tensor, 'Patches', Tuple]]]],
    best_ret: Dict[str, Dict[str, Dict[str, Union[Tensor, 'Patches', Tuple]]]],
    loss_reduction_func,
    idx: int,
):
    """
    Determine whether there is any improvement in the bounds by comparing with new results.
    :param full_ret:                The full return from the 'compute_bounds' method.
    :param best_ret:                The best return during optimization in the same format as
                                    'full_ret'
    :param loss_reduction_func:     Loss reduction function that reduces the losses to just the

    :return
            improved:          A boolean indicating whether there is any improvement in the bounds.
    """
    assert idx in (0, 1), 'idx must be 0 (lower bound) or 1 (upper bound)'

    reduced_full = loss_reduction_func(full_ret[idx])
    reduced_best = loss_reduction_func(best_ret[idx])
    return torch.any(
        (reduced_full > reduced_best) if idx == 0 else (reduced_full < reduced_best)
    ).item()


def _update_best_ret(
    full_ret: Dict[str, Dict[str, Dict[str, Union[Tensor, 'Patches', Tuple]]]],
    best_ret: Dict[str, Dict[str, Dict[str, Union[Tensor, 'Patches', Tuple]]]],
    loss_reduction_func,
    idx: int,
    reproduce_alpha_bounds: bool = False,
    best_out_in_A_dict: Optional[Dict[str, Union[Tensor, 'Patches', Tuple]]] = None,
    out_in_keys: Optional[Tuple[str, str]] = None,
):
    """
    Update best_ret_bound and best_ret by comparing with new results.
    :param full_ret:                The full return from the 'compute_bounds' method.
    :param best_ret:                The best return during optimization in the same format as
                                    'full_ret'
    :param loss_reduction_func:     Loss reduction function that reduces the losses to just the
                                    batch dimension.
    :param idx:                     0 := updating the lower bound, 1 := updating the upper bound
    :param reproduce_alpha_bounds:  If true, problems that have seen loss improvement will have
                                    their bounds directly saved as the new best bound. Otherwise,
                                    the current bounds will be compared to the current best bounds
                                    and the comparison result is saved as the new best bound. This
                                    keeps saved optimization parameters aligned with the bounds
                                    returned for the same iteration.
    :param best_out_in_A_dict:      If given, this is the A_dict entry corresponding to the output
    :param out_in_keys:             If given, this is a tuple whose first element is the first index
                                    into the A_dict and whose second element is the second index
                                    into the A_dict. In particular, the first element should be the
                                    name of the output layer of the network, and the second
                                    element should be the name of the input layer. If these indices
                                    are not given correctly, an indexing error will be thrown. If
                                    given, it is assumed that we should use these keys to update
                                    lA/uA/lbias/ubias depending on if the bounds have improved.
                                    Therefore, we must assert that 'full_ret' and 'best_ret' contain
                                    an A_dict.
    :return:
            best_ret:
            best_out_in_A_dict:     An updated A_dict entry corresponding to the output/input layer
            need_update:            Set to True in this method if at least one sub-problem has seen
                                    bound improvement.
            idx_mask:               A mask on the batch dimension where the mask is true if a
                                    sub-problem has seen loss improvement.
            improved_idx:           A Tensor of the indices in the batch dimension that have seen
                                    loss improvement.
    """
    assert idx in (0, 1), 'idx must be 0 (lower bound) or 1 (upper bound)'

    idx_mask, improved_idx = _get_idx_mask(idx, full_ret[idx], best_ret[idx], loss_reduction_func)
    if improved_idx is None:
        return best_ret, best_out_in_A_dict, False, idx_mask, None

    compare_fn = torch.max if idx == 0 else torch.min
    # Update detailed return tensors (if present)
    if full_ret[idx] is not None:
        if reproduce_alpha_bounds:
            best_ret[idx][improved_idx] = full_ret[idx][improved_idx]
            if out_in_keys is not None:
                _update_A_dict(
                    best_out_in_A_dict,
                    full_ret[2][out_in_keys[0]][out_in_keys[1]],
                    improved_idx
                )
        else:
            if out_in_keys is not None:
                # Since we must also update the A_dict, we don't want to use the original
                # 'compare' method as we need to know which specific problems have
                # seen improvement.
                cmp_op = (lambda x, y: (x > y)) if idx==0 else (lambda x, y: (x < y))
                c_mask = cmp_op(full_ret[idx][improved_idx], best_ret[idx][improved_idx])
                best_ret[idx][improved_idx] = torch.where(
                    c_mask, full_ret[idx][improved_idx], best_ret[idx][improved_idx])
                # Also update the lA/uA/lbias/ubias matrices/vectors from the output layer to
                # the input layer if the bounds have improved and if the output and input layer
                # keys were specified
                _update_A_dict(
                    best_out_in_A_dict,
                    full_ret[2][out_in_keys[0]][out_in_keys[1]],
                    improved_idx, c_mask
                )
            else:
                # Simple tensor-wise comparison (no A_dict)
                best_ret[idx][improved_idx] = compare_fn(
                    full_ret[idx][improved_idx],
                    best_ret[idx][improved_idx])

    return best_ret, best_out_in_A_dict, True, idx_mask, improved_idx


def _update_A_dict(best_A, full_A, improved_idx, c_mask: Optional[Tensor] = None):
    """
    Update best_A dict by full_A for entries at improved_idx.
    :param best_A:         The A_dict entry to be updated.
    :param full_A:         The A_dict entry containing the new values.
    :param improved_idx:   The indices in the batch dimension that have seen bound improvement.
    :param c_mask:         A mask on the batch dimension where the mask is true if a
                            sub-problem has seen bound improvement. If None, then the entire
                            slice at improved_idx will be replaced.
    """
    for key, val in full_A.items():
        if val is None:
            # An entry for lA/uA/lbias/ubias may be None depending on if we are
            # lower or upper bounding the network
            continue
        target = best_A[key][improved_idx]
        source = val[improved_idx]
        if c_mask is not None:
            c_mask_expanded = c_mask.view(
                *c_mask.shape,
                *([1] * (val.dim() - c_mask.dim()))
            ).expand_as(val[improved_idx])
            # Selectively update entries based on c_mask
            best_A[key][improved_idx] = torch.where(c_mask_expanded, source, target)
        else:
            # Replace the entire slice if no mask is provided
            best_A[key][improved_idx] = source


def _update_optimizable_activations(
        optimizable_activations, interm_bounds,
        fix_interm_bounds, best_intermediate_bounds,
        reference_idx, idx, alpha, best_alphas, reproduce_alpha_bounds,
        skip_interm_bounds_update):
    """
    Update bounds and alpha of optimizable_activations.
    """
    for node in optimizable_activations:
        # Update best intermediate layer bounds only when they are optimized.
        # If they are already fixed in interm_bounds, then do
        # nothing.
        if node.name not in best_intermediate_bounds:
            continue
        if (interm_bounds is None
                or node.inputs[0].name not in interm_bounds
                or not fix_interm_bounds) and not skip_interm_bounds_update:
            if reproduce_alpha_bounds:
                best_intermediate_bounds[node.name][0][idx] = node.inputs[0].lower[reference_idx]
                best_intermediate_bounds[node.name][1][idx] = node.inputs[0].upper[reference_idx]
            else:
                best_intermediate_bounds[node.name][0][idx] = torch.max(
                    best_intermediate_bounds[node.name][0][idx],
                    node.inputs[0].lower[reference_idx])
                best_intermediate_bounds[node.name][1][idx] = torch.min(
                    best_intermediate_bounds[node.name][1][idx],
                    node.inputs[0].upper[reference_idx])
        if alpha:
            # Each alpha has shape (2, output_shape, batch, *shape) for act.
            # For other activation function this can be different.
            for alpha_m in node.alpha:
                best_alphas[node.name][alpha_m][:, :,
                    idx] = node.alpha[alpha_m][:, :, idx]


def update_best_beta(self: 'BoundedModule', enable_opt_interm_bounds, betas,
                     best_betas, idx):
    """
    Update best beta by given idx.
    """
    if enable_opt_interm_bounds and betas:
        for node in self.get_splittable_activations():
            for node_input in node.inputs:
                if not node_input.perturbed:
                    continue
                for key in node_input.sparse_betas.keys():
                    best_betas[node_input.name][key] = (
                        node_input.sparse_betas[key].val.detach().clone())
        if self.cut_used:
            for gbidx, general_betas in enumerate(self.cut_beta_params):
                # FIXME need to check if 'cut' is a node name
                best_betas['cut'][gbidx] = general_betas.detach().clone()
    else:
        for node in self.nodes_with_beta:
            best_betas[node.name][idx] = node.sparse_betas[0].val[idx]
        if self.cut_used:
            regular_beta_length = len(betas) - len(self.cut_beta_params)
            for cut_beta_idx in range(len(self.cut_beta_params)):
                # general cut beta crown general_betas
                best_betas['cut'][cut_beta_idx][:, :, idx,
                    :] = betas[regular_beta_length + cut_beta_idx][:, :, idx, :]

@contextmanager
def tf32_ctx(enabled: bool):
    """Temporarily set TF32 policy for matmul and cuDNN."""
    old_mm  = torch.backends.cuda.matmul.allow_tf32
    old_dnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32       = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32       = old_dnn


def parse_mixed_precision_dtype(spec):
    """Parse a mixed precision spec ('fp16', 'bf16', 'fp32', 'tf32',
    'fp16+tf32', 'bf16+tf32', or None for disabled) into
    (autocast_dtype, enable_tf32). 'fp32' runs autocast with torch.float32
    directly (like fp16/bf16), rather than disabling mixed precision.
    """
    if spec is None:
        return None, False
    parts = set(str(spec).split('+'))
    enable_tf32 = 'tf32' in parts
    parts.discard('tf32')
    if not parts:
        autocast_dtype = None
    elif parts == {'fp16'}:
        autocast_dtype = torch.float16
    elif parts == {'bf16'}:
        autocast_dtype = torch.bfloat16
    elif parts == {'fp32'}:
        autocast_dtype = torch.float32
    else:
        raise ValueError(f'Unsupported mixed precision dtype: {spec!r}')
    return autocast_dtype, enable_tf32


@contextmanager
def mixed_precision_ctx(autocast_dtype, enable_tf32: bool, enabled: bool):
    """Run a low-precision iteration under CUDA autocast (autocast_dtype,
    skipped when None to avoid per-op dispatch overhead and implicit
    upcasts in TF32-only mode) and/or the TF32 matmul/cuDNN policy
    (enable_tf32), or as a no-op when not enabled.
    """
    if not enabled or (autocast_dtype is None and not enable_tf32):
        yield
        return
    with tf32_ctx(True) if enable_tf32 else ExitStack():
        if autocast_dtype is None:
            yield
        else:
            with torch.autocast(device_type='cuda', enabled=True,
                                dtype=autocast_dtype, cache_enabled=False):
                yield


def _validate_default_return_A_query(
        self, final_node_name, needed_A_dict):
    """Require a default query whose A entry follows optimized bounds.

    Optimization may select the best bound from different iterations for
    different batch and specification entries. The corresponding selection
    mask is applied only to the primary-output-to-primary-input A entry. A
    newly synthesized default query must therefore resolve to exactly that
    entry. Explicit queries may contain other entries, whose coefficients are
    taken from the final optimization iteration.
    """
    output_name, input_name = self.output_name[0], self.input_name[0]
    final_node_name = final_node_name or output_name
    requested_inputs = needed_A_dict.get(output_name, ())
    supported_query = (
        final_node_name == output_name
        and set(needed_A_dict) == {output_name}
        and isinstance(requested_inputs, (dict, list, set, tuple))
        and set(requested_inputs) == {input_name}
        and self[input_name].perturbation is not None)
    if not supported_query:
        raise NotImplementedError(
            'The default optimized return_A query supports only the primary '
            'model output with respect to its perturbed primary model input')


def _get_optimized_bounds(
        self: 'BoundedModule', x=None, aux=None, C=None, IBP=False,
        forward=False, method='backward', bound_side='lower',
        reuse_ibp=False, return_A=False, average_A=False, final_node_name=None,
        interm_bounds=None, reference_bounds=None,
        aux_reference_bounds=None, needed_A_dict=None,
        default_A_query=False, cutter=None,
        decision_thresh=None, epsilon_over_decision_thresh=1e-4):
    """
    Optimize CROWN lower/upper bounds by alpha and/or beta.
    """

    opts = self.bound_opts['optimize_bound_args']
    iteration = opts['iteration']
    if iteration < 1:
        raise ValueError('Optimized bounds require at least one iteration.')
    max_time = opts['max_time']
    beta = opts['enable_beta_crown']
    alpha = opts['enable_alpha_crown']
    apply_output_constraints_to = opts['apply_output_constraints_to']
    opt_choice = opts['optimizer']
    keep_best = opts['keep_best']
    fix_interm_bounds = opts['fix_interm_bounds']
    loss_reduction_func = opts['loss_reduction_func']
    stop_criterion_func = opts['stop_criterion_func']
    early_stop_patience = opts['early_stop_patience']
    start_save_best = opts['start_save_best']
    multi_spec_keep_func = opts['multi_spec_keep_func']
    mixed_precision_dtype = opts['mixed_precision_dtype']
    mp_autocast_dtype, mp_enable_tf32 = parse_mixed_precision_dtype(
        mixed_precision_dtype)
    low_precision_prune_in_iter = opts.get('low_precision_prune_in_iter', True)
    reproduce_alpha_bounds = opts['reproduce_alpha_bounds']
    enable_opt_interm_bounds = self.bound_opts.get(
        'enable_opt_interm_bounds', False)
    sparse_intermediate_bounds = self.bound_opts.get(
        'sparse_intermediate_bounds', False)
    verbosity = self.bound_opts['verbosity']

    if bound_side not in ['lower', 'upper']:
        raise ValueError(bound_side)
    bound_lower = bound_side == 'lower'
    bound_upper = bound_side == 'upper'

    if default_A_query:
        _validate_default_return_A_query(
            self, final_node_name, needed_A_dict)

    assert alpha or beta, (
        'nothing to optimize, use compute bound instead!')

    if C is not None:
        self.final_shape = C.size()[:2]
        self.bound_opts.update({'final_shape': self.final_shape})
    if opts['init_alpha']:
        # TODO: this should set up aux_reference_bounds.
        self.init_alpha(x, share_alphas=opts['use_shared_alpha'],
                        method=method, c=C, final_node_name=final_node_name,
                        interm_bounds=interm_bounds if fix_interm_bounds else None)

    optimizable_activations = self.get_enabled_opt_act()

    alphas, parameters = [], []
    dense_coeffs_mask = []
    if alpha:
        best_alphas = _set_alpha(
            optimizable_activations, parameters, alphas, opts['lr_alpha'])
    else:
        best_alphas = None
    if beta:
        ret_set_beta = self.set_beta(
            enable_opt_interm_bounds, parameters,
            opts['lr_beta'], opts['lr_cut_beta'], cutter, dense_coeffs_mask)
        betas, best_betas, coeffs, dense_coeffs_mask = ret_set_beta[:4]
    if apply_output_constraints_to is not None and len(apply_output_constraints_to) > 0:
        _set_gammas(self.nodes(), parameters)

    start = time.time()

    if isinstance(decision_thresh, torch.Tensor):
        if decision_thresh.dim() == 1:
            # add the spec dim to be aligned with compute_bounds return
            decision_thresh = decision_thresh.unsqueeze(-1)

    if opts['pruning_in_iteration']:
        if return_A:
            raise NotImplementedError(
                'Pruning in iteration optimization does not support '
                'return A yet. '
                'Please fix or discard this optimization by setting '
                '--disable_pruning_in_iteration '
                'or bab: pruning_in_iteration: false')
        if mixed_precision_dtype and verbosity > 1:
            print('Mixed precision + pruning-in-iteration: prunes from '
                  'low-precision iterations are speculative and rolled back '
                  'before each full-precision iteration.')
        pruner = OptPruner(
            x, threshold=opts['pruning_in_iteration_threshold'],
            multi_spec_keep_func=multi_spec_keep_func,
            loss_reduction_func=loss_reduction_func,
            decision_thresh=decision_thresh,
            epsilon_over_decision_thresh=epsilon_over_decision_thresh,
            fix_interm_bounds=fix_interm_bounds,
            low_precision_prune_margin_abs=opts.get(
                'low_precision_prune_margin_abs', 0.01))
    else:
        pruner = None

    if opt_choice == 'adam':
        opt = optim.Adam(parameters)
    elif opt_choice == 'sgd':
        opt = optim.SGD(parameters, momentum=0.9)
    else:
        raise NotImplementedError(opt_choice)

    # Create a weight vector to scale learning rate.
    loss_weight = torch.ones(
        size=(x[0].size(0),), dtype=x[0].dtype, device=x[0].device)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, opts['lr_decay'])

    # best_intermediate_bounds is linked to aux_reference_bounds!
    best_intermediate_bounds = {}
    if (sparse_intermediate_bounds and aux_reference_bounds is None
            and reference_bounds is not None):
        aux_reference_bounds = {}
        for name, (lb, ub) in reference_bounds.items():
            aux_reference_bounds[name] = [
                lb.detach().clone(), ub.detach().clone()]
    if aux_reference_bounds is None:
        aux_reference_bounds = {}

    if len(apply_output_constraints_to) > 0:
        # INVPROP requires that all layers have cached bounds. This may not be the case
        # unless we explicitly compute them.
        self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = []
        with torch.no_grad():
            self.compute_bounds(
                x=x, C=C, method='backward', bound_lower=bound_lower,
                bound_upper=bound_upper, final_node_name=final_node_name,
                interm_bounds=interm_bounds)
        self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = (
            apply_output_constraints_to
        )

    if (return_A and self.output_name[0] in needed_A_dict
            and self.input_name[0] in needed_A_dict[self.output_name[0]]):
        # _update_best_ret applies the best-bound selection mask only to the
        # primary-output-to-primary-input entry. Other requested entries come
        # from the final optimization iteration.
        out_in_keys = (self.output_name[0], self.input_name[0])
    else:
        out_in_keys = None

    need_grad = True
    patience = 0
    ret_0 = None
    best_ret = [None, None]
    best_out_in_A_dict = None

    need_full_precision_round = False
    last_round_precision: Literal['mixed', 'full', 'none'] = 'mixed'

    _prune_curve = []  # [prune-curve] per-iter (i, precision, n_domains_computed)

    for i in range(iteration):
        if i == 0:
            # If we are at the first iteration, we need to
            # set the constraints_optimized to None
            self.constraints_optimized = None

        if cutter:
            # cuts may be optimized by cutter
            self.cut_module = cutter.cut_module

        if self.constraints_optimized is not None:
            if self.bound_opts.get('clip_in_alpha_crown', False):
                assert self.bound_opts.get(
                    'clip_in_alpha_crown_disjuncts_separated', False), (
                    'clip_in_alpha_crown requires disjunctive specifications '
                    'to be batched separately before applying optimized '
                    'constraints. Set optimize_disjuncts_separately=True; if '
                    'this causes OOM, reduce build_batch_size to a smaller '
                    'value.'
                )
            for root in self.roots():
                if ( hasattr(root, 'perturbation')
                    and root.perturbation is not None
                    # Currently constraints solving is designed for LpNorm.
                    and isinstance(root.perturbation, PerturbationLpNorm) ):

                    # Reset the constraints for this root.
                    # TODO: Currently, the `reset` function simply overwrites,
                    #       should support more sophisticated reset logic.
                    root.perturbation.reset_constraints(
                        self.constraints_optimized, decision_thresh)


        intermediate_constr = None

        if not fix_interm_bounds:
            # If we still optimize all intermediate neurons, we can use
            # interm_bounds as reference bounds.
            if reference_bounds is None:
                reference_bounds = {}
            if interm_bounds is not None:
                reference_bounds.update(interm_bounds)
            interm_bounds = {}

        # The precision of this round is decided up here because a
        # full-precision round must roll back speculatively pruned domains
        # before any tensor is sliced or converted below.
        mixed_pre_condition = (
                i > 0
                and i < iteration - 1
                and (not need_full_precision_round)
                and need_grad
                and mixed_precision_dtype is not None
                and 'cuda' in str(self.device)
        )
        last_round_precision = 'mixed' if mixed_pre_condition else 'full'

        if pruner and not mixed_pre_condition:
            # Bring speculatively pruned domains back so this
            # full-precision round re-proves them before any prune commits.
            x, C = pruner.roll_back_speculative(
                x, C, interm_bounds, aux_reference_bounds, reference_bounds)
            # This full-precision round consumes any forced-full request (set
            # when a speculative prune emptied the batch); allow subsequent
            # rounds to use low precision again.
            need_full_precision_round = False

        if i == iteration - 1:
            # No grad update needed for the last iteration
            need_grad = False

        if pruner:
            # we will use last update preserve mask in caller functions to recover
            # lA, l, u, etc to full batch size
            self.last_update_preserve_mask = pruner.preserve_mask
            pruner.cache_full_sized_alpha(optimizable_activations)

        # If input bounds are tightened with output constraints, they depend on the
        # relaxations of all other layers. The current iteration will recompute them.
        # This involves concretizing them, so they will depend on themselves.
        # To avoid a loop of gradients, remove gradients here.
        tighten_input_bounds = (
                self.bound_opts['optimize_bound_args']['tighten_input_bounds']
        )
        if tighten_input_bounds:
            for root in self.roots():
                if hasattr(root, 'perturbation') and root.perturbation is not None:
                    root.perturbation.x_L = root.perturbation.x_L.detach()
                    root.perturbation.x_U = root.perturbation.x_U.detach()

        with torch.no_grad() if not need_grad else ExitStack():
            with mixed_precision_ctx(mp_autocast_dtype, mp_enable_tf32,
                                     enabled=mixed_pre_condition):
                # ret is lb, ub or lb, ub, A_dict (if return_A is set to true)
                ret = self.compute_bounds(
                    x, aux, C, method=method, IBP=IBP, forward=forward,
                    bound_lower=bound_lower, bound_upper=bound_upper,
                    reuse_ibp=reuse_ibp, return_A=return_A,
                    final_node_name=final_node_name, average_A=average_A,
                    # When intermediate bounds are recomputed, we must set it
                    # to None
                    interm_bounds=interm_bounds if fix_interm_bounds else None,
                    # This is the currently tightest interval, which will be used to
                    # pass split constraints when intermediate betas are used.
                    reference_bounds=reference_bounds,
                    # This is the interval used for checking for unstable neurons.
                    aux_reference_bounds=(
                        aux_reference_bounds if sparse_intermediate_bounds else None
                    ),
                    # These are intermediate layer beta variables and their
                    # corresponding A matrices and biases.
                    intermediate_constr=intermediate_constr,
                    needed_A_dict=needed_A_dict,
                    update_mask=pruner.get_update_mask() if pruner else None,
                    cache_bounds=len(apply_output_constraints_to) > 0,
                )
        # If output constraints are used, it's possible that no inputs satisfy them.
        # If one of the layer that uses output constraints realizes this, it sets
        # self.infeasible_bounds = True for this element in the batch.
        if self.infeasible_bounds is not None and torch.any(self.infeasible_bounds):
            if ret[0] is not None:
                ret = (
                    torch.where(
                        self.infeasible_bounds.unsqueeze(1),
                        torch.full_like(ret[0], float('inf')),
                        ret[0],
                    ),
                    ret[1],
                )
            if ret[1] is not None:
                ret = (
                    ret[0],
                    torch.where(
                        self.infeasible_bounds.unsqueeze(1),
                        torch.full_like(ret[1], float('-inf')),
                        ret[1],
                    ),
                )
        ret_l, ret_u = ret[0], ret[1]

        # [prune-curve] save how many domains were actually computed this iteration
        _prune_curve.append(
            (i, last_round_precision,
             int(ret_l.shape[0]) if ret_l is not None else
             (int(ret_u.shape[0]) if ret_u is not None else -1)))

        if pruner:
            pruner.recover_full_sized_alpha(optimizable_activations)

        if (self.cut_used and i % cutter.log_interval == 0
                and len(self.cut_beta_params) > 0):
            # betas[-1]: (2(0 lower, 1 upper), spec, batch, num_constrs)
            if ret_l is not None:
                print(i, 'lb beta sum:',
                    f'{self.cut_beta_params[-1][0].sum() / ret_l.size(0)},',
                    f'worst {ret_l.min()}')
            if ret_u is not None:
                print(i, 'lb beta sum:',
                    f'{self.cut_beta_params[-1][1].sum() / ret_u.size(0)},',
                    f'worst {ret_u.min()}')

        if i == 0:
            # save results at the first iteration
            best_ret = [ret.detach().clone() if ret is not None else None for ret in ret[:2]]
            ret_0 = ret[0].detach().clone() if bound_lower else ret[1].detach().clone()

            for node in optimizable_activations:
                if node.inputs[0].lower is None and node.inputs[0].upper is None:
                    continue
                new_intermediate = [node.inputs[0].lower.detach().clone(),
                                    node.inputs[0].upper.detach().clone()]
                best_intermediate_bounds[node.name] = new_intermediate
                if sparse_intermediate_bounds:
                    # Always using the best bounds so far as the reference
                    # bounds.
                    aux_reference_bounds[node.inputs[0].name] = new_intermediate

            if out_in_keys is not None:
                best_out_in_A_dict = clone_sub_A_dict(ret[2], out_in_keys)
            else:
                best_out_in_A_dict = None

        l = ret_l
        # Reduction over the spec dimension.
        if ret_l is not None and ret_l.shape[1] != 1:
            l = loss_reduction_func(ret_l)
        u = ret_u
        if ret_u is not None and ret_u.shape[1] != 1:
            u = loss_reduction_func(ret_u)

        # full_l, full_ret_l and full_u, full_ret_u is used for update the best
        full_ret_l, full_ret_u = ret_l, ret_u
        full_l = l
        full_ret = ret

        if pruner:
            (x, C, full_l, full_ret_l, full_ret_u,
            full_ret, stop_criterion) = pruner.prune(
                x, C, ret_l, ret_u, ret, full_l, full_ret_l, full_ret_u,
                full_ret, interm_bounds, aux_reference_bounds, reference_bounds,
                stop_criterion_func, bound_lower,
                # Prunes based on low-precision bounds are speculative and
                # will be rolled back before the next full-precision round.
                is_low_precision_bound=(last_round_precision == 'mixed'),
                # Optionally do not prune at all during low-precision rounds.
                allow_prune=(low_precision_prune_in_iter
                             or last_round_precision == 'full'))
            # If pruning removed every domain (e.g. an aggressive speculative
            # prune in a low-precision round), force a full-precision round so
            # roll_back_speculative restores the full batch and re-proves it.
            if x[0].shape[0] == 0 and last_round_precision == 'mixed':
                need_full_precision_round = True
        else:
            stop_criterion = (stop_criterion_func(full_ret_l) if bound_lower
                            else stop_criterion_func(-full_ret_u))

        loss_ = l if bound_lower else -u
        total_loss = -1 * loss_
        directly_optimize_layers = self.bound_opts['optimize_bound_args']['directly_optimize']
        for directly_optimize_layer_name in directly_optimize_layers:
            total_loss += (
                self[directly_optimize_layer_name].upper.sum()
                - self[directly_optimize_layer_name].lower.sum()
            )

        if type(stop_criterion) == bool:
            loss = total_loss.sum() * (not stop_criterion)
        else:
            assert total_loss.shape == stop_criterion.shape
            loss = (total_loss * stop_criterion.logical_not()).sum()

        stop_criterion_final = isinstance(
            stop_criterion, torch.Tensor) and stop_criterion.all().item()

        with torch.no_grad():
            # for lb and ub, we update them in every iteration since updating them is cheap
            need_update = False
            improved_idx = None
            idx_mask = None
            if last_round_precision == 'mixed':
            # only determine need_update
                need_update = (
                    _determine_improve(full_ret, best_ret, loss_reduction_func, idx=0)
                    if best_ret[0] is not None else False
                ) or (
                    _determine_improve(full_ret, best_ret, loss_reduction_func, idx=1)
                    if best_ret[1] is not None else False
                )
            elif keep_best:
                if best_ret[0] is not None:
                    (
                        best_ret, best_out_in_A_dict,
                        need_update, idx_mask, improved_idx,
                    ) = _update_best_ret(
                        full_ret, best_ret,
                        loss_reduction_func,
                        idx=0,
                        reproduce_alpha_bounds=reproduce_alpha_bounds,
                        best_out_in_A_dict=best_out_in_A_dict,
                        out_in_keys=out_in_keys,
                    )
                if best_ret[1] is not None:
                    (
                        best_ret, best_out_in_A_dict,
                        need_update, idx_mask, improved_idx,
                    ) = _update_best_ret(
                        full_ret, best_ret,
                        loss_reduction_func,
                        idx=1,
                        reproduce_alpha_bounds=reproduce_alpha_bounds,
                        best_out_in_A_dict=best_out_in_A_dict,
                        out_in_keys=out_in_keys,
                    )
            else:
                # Not saving the best, just keep the last iteration.
                if full_ret[0] is not None:
                    best_ret[0] = full_ret[0]
                if full_ret[1] is not None:
                    best_ret[1] = full_ret[1]

            if return_A:
                best_ret = [best_ret[0], best_ret[1], full_ret[2]]
                if out_in_keys is not None:
                    # Update the output/input A entry to match the best bounds.
                    best_ret[2][out_in_keys[0]][out_in_keys[1]] = best_out_in_A_dict

            patience = 0 if need_update else patience + 1
            time_spent = time.time() - start

            # Save variables if this is the best iteration.
            # To save computational cost, we only check keep_best at the first
            # (in case divergence) and second half iterations
            # or before early stop by either stop_criterion or
            # early_stop_patience reached.
            # Skip if during mixed precision rounds.
            if last_round_precision != 'mixed' and (
                i < 1
                or i > int(iteration * start_save_best)
                or reproduce_alpha_bounds
                or stop_criterion_final
                or patience == early_stop_patience
                or time_spent > max_time
            ):
                # compare with the first iteration results and get improved indexes
                if bound_lower:
                    if reproduce_alpha_bounds:
                        idx = improved_idx
                    else:
                        idx_mask, idx = _get_idx_mask(0, full_ret_l, ret_0, loss_reduction_func)
                else:
                    if reproduce_alpha_bounds:
                        idx = improved_idx
                    else:
                        idx_mask, idx = _get_idx_mask(1, full_ret_u, ret_0, loss_reduction_func)

                if idx is not None:
                    # for update propose, we condition the idx to update only
                    # on domains preserved
                    if pruner:
                        reference_idx, idx = pruner.prune_idx(idx_mask, idx, x)
                    else:
                        reference_idx = idx
                    # Update the improvement reference only at preserved
                    # cells: pruned cells hold filler values
                    # (threshold + eps), and a speculatively pruned domain
                    # that is later restored would otherwise compare against
                    # the filler and never register as improved again.
                    ret_0[idx] = (full_ret_l if bound_lower else full_ret_u)[idx]

                    _update_optimizable_activations(
                        optimizable_activations, interm_bounds,
                        fix_interm_bounds, best_intermediate_bounds,
                        reference_idx, idx, alpha, best_alphas,
                        reproduce_alpha_bounds,
                        skip_interm_bounds_update=(last_round_precision == 'mixed'),
                    )

                    if beta:
                        self.update_best_beta(enable_opt_interm_bounds, betas,
                                            best_betas, idx)

        if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
            learning_rate = opt.param_groups[0]['lr']
            print(f'****** iter [{i}]',
                f'loss: {loss.item()}, lr: {learning_rate}',
                (' pruning_in_iteration open status: '
                    f'{pruner.pruning_in_iteration}') if pruner else '')

        if stop_criterion_final:
            print(f'\nall verified at {i}th iter')
            if last_round_precision != 'mixed':
                break
            else:
                need_full_precision_round = True

        if patience > early_stop_patience:
            print(f'Early stop at {i}th iter due to {early_stop_patience}'
                ' iterations no improvement!')
            if last_round_precision != 'mixed':
                break
            else:
                need_full_precision_round = True

        if time_spent > max_time:
            print(f'Early stop at {i}th iter due to exceeding the time limit '
                f'for the optimization (time spent: {time_spent})')
            if last_round_precision != 'mixed':
                break
            else:
                need_full_precision_round = True

        if i != iteration - 1 and not loss.requires_grad and not need_full_precision_round:
            assert i == 0, (i, iteration)
            print('[WARNING] No optimizable parameters found. Will skip optimiziation. '
                'This happens e.g. if all optimizable layers are freezed or the '
                'network has no optimizable layers.')
            break

        opt.zero_grad(set_to_none=True)

        if verbosity > 2:
            current_lr = [param_group['lr'] for param_group in opt.param_groups]
            print(f'*** iter [{i}]\n', f'loss: {loss.item()}',
                total_loss.squeeze().detach().cpu().numpy(), 'lr: ',
                current_lr)
            if beta:
                print_optimized_beta(optimizable_activations)
            if beta and i == 0 and verbosity > 2:
                breakpoint()

        if i != iteration - 1:
            # we do not need to update parameters in the last step since the
            # best result already obtained
            # No autocast here: backward replays the dtypes recorded in the
            # forward pass; only the TF32 policy needs to match.
            with tf32_ctx(True) if (mixed_pre_condition and mp_enable_tf32) \
                    else ExitStack():
                loss.backward()
                # All intermediate variables are not needed at this point.
                self._clear_and_set_new(
                    None,
                    cache_bounds=len(apply_output_constraints_to) > 0,
                )
                if pruner and pruner.has_uncommitted_speculation():
                    # Zero the Adam momentum of zero-gradient entries while
                    # speculative prunes are outstanding, since pruned
                    # domains receive no gradient and stale momentum would
                    # otherwise drift their parameters until they fail the
                    # full-precision re-proof.
                    with torch.no_grad():
                        for group in opt.param_groups:
                            for p in group['params']:
                                state = opt.state.get(p, None)
                                if (p.grad is not None and state is not None
                                        and 'exp_avg' in state):
                                    state['exp_avg'][p.grad == 0] = 0
                opt.step()

        if beta:
            for b in betas:
                b.data = (b >= 0) * b.data
            for dmi in range(len(dense_coeffs_mask)):
                # apply dense mask to the dense split coeffs matrix
                coeffs[dmi].data = (
                    dense_coeffs_mask[dmi].to(dtype=coeffs[dmi].dtype)
                    * coeffs[dmi].data)


        if alpha:
            for m in optimizable_activations:
                m.clip_alpha()
        if apply_output_constraints_to is not None and len(apply_output_constraints_to) > 0:
            for m in self.nodes():
                m.clip_gammas()

        scheduler.step()

        if pruner:
            pruner.next_iter()

    if pruner and last_round_precision != 'mixed':
        best_ret = pruner.update_best(full_ret_l, full_ret_u, best_ret)

    if verbosity > 3:
        breakpoint()

    if keep_best:
        # Set all variables to their saved best values.
        with torch.no_grad():
            for idx, node in enumerate(optimizable_activations):
                if node.name not in best_intermediate_bounds:
                    continue
                if alpha:
                    # Assigns a new dictionary.
                    node.alpha = best_alphas[node.name]
                # Update best intermediate layer bounds only when they are
                # optimized. If they are already fixed in
                # interm_bounds, then do nothing.
                best_intermediate = best_intermediate_bounds[node.name]
                node.inputs[0].lower.data = best_intermediate[0].data
                node.inputs[0].upper.data = best_intermediate[1].data
            if beta:
                for node in self.nodes_with_beta:
                    assert getattr(node, 'sparse_betas', None) is not None
                    if enable_opt_interm_bounds:
                        for key in node.sparse_betas.keys():
                            node.sparse_betas[key].val.copy_(
                                best_betas[node.name][key])
                    else:
                        node.sparse_betas[0].val.copy_(best_betas[node.name])
            if self.cut_used:
                for ii in range(len(self.cut_beta_params)):
                    self.cut_beta_params[ii].data = best_betas['cut'][ii].data

    if interm_bounds is not None and not fix_interm_bounds:
        for l in self._modules.values():
            if (l.name in interm_bounds.keys()
                    and l.is_lower_bound_current()):
                l.lower = torch.max(l.lower, interm_bounds[l.name][0])
                l.upper = torch.min(l.upper, interm_bounds[l.name][1])
                infeasible_neurons = l.lower > l.upper
                if infeasible_neurons.any():
                    print(f'Infeasibility detected in layer {l.name}.',
                          infeasible_neurons.sum().item(),
                          infeasible_neurons.nonzero()[:, 0])

    if verbosity > 0:
        if best_ret[0] is not None:
            # FIXME: unify the handling of l and u.
            print('best_l after optimization:', best_ret[0].sum().item())
            if beta:
                print('beta sum per layer:', [p.sum().item() for p in betas])
        print('alpha/beta optimization time:', time.time() - start)

    for node in optimizable_activations:
        node.opt_end()

    if pruner:
        pruner.update_ratio(full_l, full_ret_l)
        pruner.clean_full_sized_alpha_cache()

    if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
        print()

    assert last_round_precision == 'full'

    if os.environ.get('AUTOLIRPA_PRUNE_CURVE'):
        _dt = mixed_precision_dtype if mixed_precision_dtype else 'none'
        _init = _prune_curve[0][2] if _prune_curve else -1
        _counts = ','.join(str(n) for _, _, n in _prune_curve)
        _prec = ''.join('m' if p == 'mixed' else 'f' for _, p, _ in _prune_curve)
        print(f'[prune-curve] dtype={_dt} init={_init} niter={len(_prune_curve)} '
              f'prec={_prec} counts={_counts}', flush=True)

    return best_ret


def init_alpha(self: 'BoundedModule', x, share_alphas=False, method='backward',
               c=None, bound_lower=True, bound_upper=True, final_node_name=None,
               interm_bounds=None, reference_alphas=None,
               skip_bound_compute=False):
    self(*x) # Do a forward pass to set perturbed nodes
    final = (self.final_node() if final_node_name is None
             else self[final_node_name])
    self._update_bound_ancestors(final)

    optimizable_activations = self.get_enabled_opt_act()
    for node in optimizable_activations:
        # TODO(7/6/2023) In the future, we may need to enable alpha sharing
        # automatically by consider the size of all the optimizable nodes in the
        # graph. For now, only an adhoc check in MatMul is added.
        node._all_optimizable_activations = optimizable_activations

        # initialize the parameters
        node.opt_init()

    apply_output_constraints_to = (
        self.bound_opts['optimize_bound_args']['apply_output_constraints_to']
    )
    if (not skip_bound_compute or interm_bounds is None or
            reference_alphas is None or not all(
                [act.name in reference_alphas
                 for act in optimizable_activations])):
        skipped = False
        # if new interval is None, then CROWN interval is not present
        # in this case, we still need to redo a CROWN pass to initialize
        # lower/upper
        with torch.no_grad():
            # We temporarilly deactivate output constraints
            self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = []
            l, u = self.compute_bounds(
                x=x, C=c, method=method, bound_lower=bound_lower,
                bound_upper=bound_upper, final_node_name=final_node_name,
                interm_bounds=interm_bounds)
            self.bound_opts['optimize_bound_args']['apply_output_constraints_to'] = (
                apply_output_constraints_to
            )
            if len(apply_output_constraints_to) > 0:
                # Some layers, such as the BoundTanh layer, do some of their initialization
                # in the forward pass. We need to call the forward pass again to ensure
                # that they are initialized for the output constraints, too.
                l, u = self.compute_bounds(
                    x=x, C=c, method=method, bound_lower=bound_lower,
                    bound_upper=bound_upper, final_node_name=final_node_name,
                    interm_bounds=interm_bounds, cache_bounds=True)
    else:
        # we skip, but we still would like to figure out the "used",
        # "perturbed", "backward_from" of each note in the graph
        skipped = True
        # this set the "perturbed" property
        self.set_input(*x, interm_bounds=interm_bounds)
        self.backward_from = {node: [final] for node in self._modules}
        l = u = None

    final_node_name = final_node_name or self.final_name

    init_intermediate_bounds = {}
    splittable_activations = set(self.get_splittable_activations())
    for node in optimizable_activations:
        start_nodes = []
        if method in ['forward', 'forward+backward']:
            start_nodes.append(('_forward', 1, None, False))
        if method in ['backward', 'forward+backward']:
            start_nodes += self.get_alpha_crown_start_nodes(
                node,
                c=c,
                share_alphas=share_alphas,
                final_node_name=final_node_name,
            )
        if not start_nodes:
            continue
        if skipped:
            node.restore_alpha(reference_alphas[node.name], device=x[0].device, dtype=x[0].dtype)

        else:
            node.init_opt_parameters(start_nodes)
        if node in splittable_activations:
            for i, input_node in enumerate(node.inputs):
                if (not input_node.perturbed
                        or node.inputs[i].lower is None
                        and node.inputs[i].upper is None):
                    continue
                init_intermediate_bounds[node.inputs[i].name] = (
                    [node.inputs[i].lower.detach(),
                    node.inputs[i].upper.detach()])
    if (
        apply_output_constraints_to is not None
        and len(apply_output_constraints_to) > 0
        and hasattr(self, 'constraints')
    ):
        # self.constraints.shape = (batch_size, num_constraints, num_output_neurons)
        # For abCROWN we know that:
        # If the output constraints are a conjunction, the shape is (1, num_constraints, *)
        # If the output constraints are a disjunction, the shape is (num_constraints, 1, *)
        # Checking which entry is 1 allows to discern both cases.
        # If auto_LiRPA is used directly, we could have batches of inputs with more than one
        # constraint. This is currently not supported.
        if self.constraints.size(0) == 1:
            num_gammas = self.constraints.size(1)
        elif self.constraints.size(1) == 1:
            num_gammas = self.constraints.size(0)
        else:
            raise NotImplementedError(
                'To use output constraints, either have a batch size of 1 or use only one '
                'output constraint'
            )
        for node in self.nodes():
            if not node.are_output_constraints_activated_for_layer(
                    apply_output_constraints_to):
                continue
            node.init_gammas(
                num_gammas,
                dtype=self.get_output_dtype(node),
                device=self.device,
            )

    if self.bound_opts['verbosity'] >= 1:
        print('Optimizable variables initialized.')
    if skip_bound_compute:
        return init_intermediate_bounds
    else:
        return l, u, init_intermediate_bounds
