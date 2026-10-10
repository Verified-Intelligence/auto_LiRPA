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
"""Pruning during the optimization."""

import os
import time

import torch

class OptPruner:

    def __init__(self, x, threshold, multi_spec_keep_func, loss_reduction_func,
                 decision_thresh, fix_interm_bounds,
                 epsilon_over_decision_thresh,
                 low_precision_prune_margin_abs):
        self.x = x
        self.threshold = threshold
        self.multi_spec_keep_func = multi_spec_keep_func
        self.loss_reduction_func = loss_reduction_func
        self.decision_thresh = decision_thresh
        self.fix_interm_bounds = fix_interm_bounds
        self.epsilon_over_decision_thresh = epsilon_over_decision_thresh
        # Only speculatively prune a domain if its low-precision bound clears
        # the decision threshold by this absolute margin, since marginal
        # domains often fail the full-precision re-proof.
        self.low_precision_prune_margin_abs = low_precision_prune_margin_abs

        # For computing the positive domain ratio
        self.original_size = x[0].shape[0]
        self.pruning_in_iteration = False
        self.preserve_mask = None
        self.preserve_mask_next = None
        self.time = 0

        # For holding full-sized alphas
        self.cached_alphas = {}

        # Snapshot of the pruning state decided by full-precision bounds,
        # to which speculative prunes are rolled back.
        self.committed = None
        # Domains restored by the last roll_back_speculative() call, tracked
        # only under SPEC_PRUNE_DEBUG for re-proof statistics.
        self._last_restored = None

    def prune(self, x, C, ret_l, ret_u, ret, full_l, full_ret_l, full_ret_u,
              full_ret, interm_bounds, aux_reference_bounds, reference_bounds,
              stop_criterion_func, bound_lower, is_low_precision_bound,
              allow_prune):
        # positive domains may already be filtered out, so we use all domains -
        # negative domains to compute
        # FIXME Only using ret_l but not ret_u.
        if self.decision_thresh is not None and ret_l is not None:
            if (isinstance(self.decision_thresh, torch.Tensor)
                    and self.decision_thresh.numel() > 1
                    and self.preserve_mask is not None):
                if self.decision_thresh.shape[-1] == 1:
                    # single spec with pruned domains
                    negative_domain = (
                        ret_l.view(-1)
                        <= self.decision_thresh[self.preserve_mask].view(-1)
                    ).sum()
                else:
                    # multiple spec with pruned domains
                    negative_domain = self.multi_spec_keep_func(
                        ret_l <= self.decision_thresh[self.preserve_mask]).sum()
            else:
                if ret_l.shape[-1] == 1:
                    # single spec
                    negative_domain = (
                        ret_l.view(-1) <= self.decision_thresh.view(-1)).sum()
                else:
                    # multiple spec
                    negative_domain = self.multi_spec_keep_func(
                        ret_l <= self.decision_thresh).sum()
            positive_domain_num = self.original_size - negative_domain
        else:
            positive_domain_num = -1
        positive_domain_ratio = float(
            positive_domain_num) / float(self.original_size)
        # threshold is 10% by default
        self.next_iter_pruning_in_iteration = (
            self.decision_thresh is not None
            and positive_domain_ratio > self.threshold)

        if self.pruning_in_iteration:
            stime = time.time()
            if self.committed is None:
                # First prune since pruning was activated: the current
                # (pre-slice) state is committed by construction.
                self._commit(x, C, interm_bounds, aux_reference_bounds,
                             reference_bounds, self.preserve_mask)
            if allow_prune:
                self.get_preserve_mask(
                    ret_l, is_low_precision_bound=is_low_precision_bound)
                nothing_pruned = (
                    self.now_preserve_mask.numel() == x[0].shape[0])
            else:
                # Pruning is disabled for this round (e.g. speculative
                # pruning turned off during low-precision iterations).
                nothing_pruned = True
            if nothing_pruned:
                # Nothing is pruned in this round: skip the slicing, which
                # would otherwise create a full copy of every tensor.
                self.preserve_mask_next = self.preserve_mask
                full_ret_l, full_l = self._recover_bounds_to_full_batch(ret_l)
                full_ret_u, full_u = self._recover_bounds_to_full_batch(ret_u)
                full_ret = (full_ret_l, full_ret_u) + ret[2:]
            else:
                # prune C
                if C is not None and C.shape[0] == x[0].shape[0]:
                    C = C[self.now_preserve_mask]  # means C is also batch specific
                # prune x
                x, pre_prune_size = self._prune_x(x)
                # prune bounds
                ret_prune = self._prune_bounds_by_mask(
                    ret_l, ret_u, ret,
                    interm_bounds, aux_reference_bounds, reference_bounds, pre_prune_size)
                full_l, full_ret_l, full_ret_u, full_ret = ret_prune
            if not is_low_precision_bound:
                self._debug_print_reproof_stats(ret_l)
                # Prune decided by full-precision bounds: commit it.
                self._commit(x, C, interm_bounds, aux_reference_bounds,
                             reference_bounds, self.preserve_mask_next)
            self.time += time.time() - stime

        stop_criterion = stop_criterion_func(
            full_ret_l) if bound_lower else stop_criterion_func(-full_ret_u)
        if (type(stop_criterion) != bool and stop_criterion.numel() > 1
                and self.pruning_in_iteration):
            stop_criterion = stop_criterion[self.preserve_mask]

        return (x, C, full_l, full_ret_l, full_ret_u,
                full_ret, stop_criterion)

    def _commit(self, x, C, interm_bounds, aux_reference_bounds,
                reference_bounds, mask):
        """Snapshot the pruning state decided by full-precision bounds,
        holding references (not copies) since slicing creates new tensors.
        """
        def _copy_dict(d):
            return ({k: list(v) for k, v in d.items()}
                    if d is not None else None)
        self.committed = {
            'x_data': x[0].data,
            'x_L': x[0].ptb.x_L if hasattr(x[0], 'ptb') else None,
            'x_U': x[0].ptb.x_U if hasattr(x[0], 'ptb') else None,
            'C': C,
            # Mirror the choice in _prune_bounds_by_mask: interm_bounds is
            # only sliced when intermediate bounds are fixed.
            'interm_bounds': _copy_dict(
                interm_bounds if self.fix_interm_bounds else None),
            'aux_reference_bounds': _copy_dict(aux_reference_bounds),
            'reference_bounds': _copy_dict(reference_bounds),
            'mask': mask,
        }

    @torch.no_grad()
    def roll_back_speculative(self, x, C, interm_bounds, aux_reference_bounds,
                            reference_bounds):
        """Roll back speculative prunes to the last committed state before
        a full-precision iteration, returning the restored (x, C) and
        restoring the bound dicts in place.
        """
        if (not self.pruning_in_iteration or self.committed is None
                # Masks only shrink, so equal sizes means no speculative
                # prune happened since the last commit.
                or self.preserve_mask.numel()
                    == self.committed['mask'].numel()):
            return x, C
        x = list(x)
        x[0].data = self.committed['x_data']
        if hasattr(x[0], 'ptb'):
            if self.committed['x_L'] is not None:
                x[0].ptb.x_L = self.committed['x_L']
            if self.committed['x_U'] is not None:
                x[0].ptb.x_U = self.committed['x_U']
        x = tuple(x)
        if self.committed['C'] is not None:
            C = self.committed['C']
        for target, saved in (
                (interm_bounds if self.fix_interm_bounds else None,
                 self.committed['interm_bounds']),
                (aux_reference_bounds,
                 self.committed['aux_reference_bounds']),
                (reference_bounds, self.committed['reference_bounds'])):
            if target is not None and saved is not None:
                for k, v in saved.items():
                    target[k] = list(v)
        if os.environ.get('SPEC_PRUNE_DEBUG'):
            self._last_restored = self.committed['mask'][
                ~torch.isin(self.committed['mask'], self.preserve_mask)]
        self.preserve_mask = self.committed['mask']
        return x, C

    @torch.no_grad()
    def _debug_print_reproof_stats(self, ret_l):
        """Under SPEC_PRUNE_DEBUG, after a full-precision prune, print how
        many restored domains failed the re-proof (still preserved) and how
        far their full-precision bounds land from the decision threshold.
        """
        if (not os.environ.get('SPEC_PRUNE_DEBUG')
                or self._last_restored is None
                or self._last_restored.numel() == 0):
            return
        kept = torch.isin(self.preserve_mask_next,
                          self._last_restored).sum().item()
        restored_pos = torch.isin(self.preserve_mask, self._last_restored)
        vals = ret_l[restored_pos].view(-1)
        if isinstance(self.decision_thresh, torch.Tensor) \
                and self.decision_thresh.numel() > 1:
            th = self.decision_thresh[self.preserve_mask][
                restored_pos].view(-1)
        else:
            th = self.decision_thresh
        gap = vals - th
        print(f'[spec-prune] restored {self._last_restored.numel()}, '
              f're-proof failed (kept) {kept}, '
              f'gap min/median/max = {gap.min().item():.4f}/'
              f'{gap.median().item():.4f}/{gap.max().item():.4f}')
        self._last_restored = None

    def get_update_mask(self):
        """Return preserve_mask for compute_bounds, or None when nothing is
        pruned, since a complete mask would still trigger per-layer
        alpha/beta masking in the backward pass for no effect.
        """
        if (not self.pruning_in_iteration
                or self.preserve_mask is None
                or self.preserve_mask.numel() == self.original_size):
            return None
        return self.preserve_mask

    def has_uncommitted_speculation(self):
        """True if some domains are currently speculatively pruned."""
        return (self.pruning_in_iteration and self.committed is not None
                and self.preserve_mask.numel()
                    < self.committed['mask'].numel())

    def prune_idx(self, idx_mask, idx, x):
        if self.pruning_in_iteration:
            # local sparse index of preserved samples where
            # idx == true
            local_idx = idx_mask[self.preserve_mask].nonzero().view(-1)
            # idx is global sparse index of preserved samples where
            # idx == true
            new_idx = torch.zeros_like(
                idx_mask, dtype=torch.bool, device=x[0].device)
            new_idx[self.preserve_mask] = idx_mask[self.preserve_mask]
            idx = new_idx.nonzero().view(-1)
            reference_idx = local_idx
        else:
            reference_idx = idx
        return reference_idx, idx

    def next_iter(self):
        if self.pruning_in_iteration:
            self.preserve_mask = self.preserve_mask_next
        if (not self.pruning_in_iteration
                and self.next_iter_pruning_in_iteration):
            # init preserve_mask etc
            self.preserve_mask = torch.arange(
                0, self.x[0].shape[0], device=self.x[0].device, dtype=torch.long)
            self.pruning_in_iteration = True

    def update_best(self, full_ret_l, full_ret_u, best_ret):
        if self.pruning_in_iteration:
            # overwrite pruned cells in best_ret by threshold + eps
            fin_l, fin_u = best_ret
            if fin_l is not None:
                new_fin_l = full_ret_l
                new_fin_l[self.preserve_mask] = fin_l[self.preserve_mask]
                fin_l = new_fin_l
            if fin_u is not None:
                new_fin_u = full_ret_u
                new_fin_u[self.preserve_mask] = fin_u[self.preserve_mask]
                fin_u = new_fin_u
            best_ret = (fin_l, fin_u)
        return best_ret

    def update_ratio(self, full_l, full_ret_l):
        if self.decision_thresh is not None and full_l.numel() > 0:
            stime = time.time()
            with torch.no_grad():
                if isinstance(self.decision_thresh, torch.Tensor):
                    if self.decision_thresh.shape[-1] == 1:
                        neg_domain_num = torch.sum(
                            full_ret_l.view(-1) <= self.decision_thresh.view(-1)
                        ).item()
                    else:
                        neg_domain_num = torch.sum(self.multi_spec_keep_func(
                            full_ret_l <= self.decision_thresh)).item()
                else:
                    if full_l.shape[-1] == 1:
                        neg_domain_num = torch.sum(
                            full_ret_l.view(-1) <= self.decision_thresh).item()
                    else:
                        neg_domain_num = torch.sum(self.multi_spec_keep_func(
                            full_ret_l <= self.decision_thresh)).item()
                now_pruning_ratio = (
                    1.0 - float(neg_domain_num) / float(full_l.shape[0]))
                print('pruning_in_iteration open status:',
                      self.pruning_in_iteration)
                print('ratio of positive domain =',
                    full_l.shape[0] - neg_domain_num,
                    '/', full_l.numel(), '=', now_pruning_ratio)
            self.time += time.time() - stime
            print('pruning-in-iteration extra time:', self.time)

    @torch.no_grad()
    def _prune_x(self, x):
        """
        Prune x by given now_preserve_mask.
        """
        x = list(x)
        pre_prune_size = x[0].shape[0]
        x[0].data = x[0][self.now_preserve_mask].data
        if hasattr(x[0], 'ptb'):
            if x[0].ptb.x_L is not None:
                x[0].ptb.x_L = x[0].ptb.x_L[self.now_preserve_mask]
            if x[0].ptb.x_U is not None:
                x[0].ptb.x_U = x[0].ptb.x_U[self.now_preserve_mask]
        x = tuple(x)

        return x, pre_prune_size

    def _prune_dict_of_lists(self, dict_of_lists, pre_prune_size):
        if dict_of_lists is not None:
            for k, v in dict_of_lists.items():
                v_l, v_r = v[0], v[1]
                if v_l.shape[0] == pre_prune_size:
                    # the first dim is batch size and matches the preserve mask
                    v_l = v_l[self.now_preserve_mask]
                if v_r.shape[0] == pre_prune_size:
                    # the first dim is batch size and matches the preserve mask
                    v_r = v_r[self.now_preserve_mask]
                dict_of_lists[k] = [v_l, v_r]

    @torch.no_grad()
    def _prune_bounds_by_mask(self, ret_l, ret_u, ret, interm_bounds,
                              aux_reference_bounds, reference_bounds, pre_prune_size):
        """
        Prune bounds by given now_preserve_mask.
        """
        full_ret_l, full_l = self._recover_bounds_to_full_batch(ret_l)
        full_ret_u, full_u = self._recover_bounds_to_full_batch(ret_u)

        full_ret = (full_ret_l, full_ret_u) + ret[2:]

        if self.fix_interm_bounds:
            interval_to_prune = interm_bounds
        else:
            interval_to_prune = None

        self._prune_dict_of_lists(interval_to_prune, pre_prune_size)
        self._prune_dict_of_lists(aux_reference_bounds, pre_prune_size)
        self._prune_dict_of_lists(reference_bounds, pre_prune_size)

        # update the global mask here for possible next iteration
        self.preserve_mask_next = self.preserve_mask[self.now_preserve_mask]

        return full_l, full_ret_l, full_ret_u, full_ret

    @torch.no_grad()
    def get_preserve_mask(self, ret_l, is_low_precision_bound):
        """
        Get preserve mask by decision_thresh to filter out the satisfied bounds.
        :param ret_l: the lower bound of the current batch, which is used to compare with the decision threshold for pruning.
        :param is_low_precision_bound: whether the ret_l is from low-precision bounds.
        """
        if is_low_precision_bound:
            # Require low-precision bounds to clear the threshold by a margin
            # before speculatively pruning.
            ret_l = ret_l - self.low_precision_prune_margin_abs
        if (isinstance(self.decision_thresh, torch.Tensor)
                and self.decision_thresh.numel() > 1):
            if self.decision_thresh.shape[-1] == 1:
                self.now_preserve_mask = (
                    ret_l <= self.decision_thresh[self.preserve_mask]
                ).view(-1).nonzero().view(-1)
            else:
                self.now_preserve_mask = self.multi_spec_keep_func(
                    ret_l <= self.decision_thresh[self.preserve_mask]
                ).nonzero().view(-1)
        else:
            if self.decision_thresh.shape[-1] == 1:
                self.now_preserve_mask = (
                    ret_l <= self.decision_thresh).view(-1).nonzero().view(-1)
            else:
                self.now_preserve_mask = self.multi_spec_keep_func(
                    ret_l <= self.decision_thresh).nonzero().view(-1)

    def _recover_bounds_to_full_batch(self, ret):
        """
        Recover lower and upper bounds to full batch size so that later we can
        directly update using the full batch size of l and u.
        """
        if ret is not None:
            if ret.shape[0] == self.original_size:
                # ret is already full batch size, no need to recover
                full_ret = ret
            elif (isinstance(self.decision_thresh, torch.Tensor)
                    and self.decision_thresh.numel() > 1):
                full_ret = (
                    self.decision_thresh.clone().to(ret.device).type(ret.dtype)
                    + self.epsilon_over_decision_thresh)
                full_ret[self.preserve_mask] = ret
            else:
                num_decision_thresh = self.decision_thresh
                if isinstance(num_decision_thresh, torch.Tensor):
                    num_decision_thresh = num_decision_thresh.item()
                full_ret = torch.full(
                    (self.original_size,) + tuple(ret.shape[1:]),
                    fill_value=(num_decision_thresh
                                + self.epsilon_over_decision_thresh),
                    device=ret.device, dtype=ret.dtype)
                full_ret[self.preserve_mask] = ret
            if full_ret.shape[1] > 1:
                full_reduced_ret = self.loss_reduction_func(full_ret)
            else:
                full_reduced_ret = full_ret
        else:
            full_ret = full_reduced_ret = None

        return full_ret, full_reduced_ret

    def cache_full_sized_alpha(self, optimizable_activations: list):
        """
        When preserve mask is in use, cache the full-sized alphas in self.cached_alphas,
        and rewrite the alphas in nodes according to the preserve mask.
        The full-sized alphas will be recovered back to nodes after compute_bounds,
        via the function named recover_full_sized_alphas()
        :param optimizable_activations: list of nodes that may have slope alphas as optimizable variables
        :return: None
        """
        if self.pruning_in_iteration:
            for act in optimizable_activations:
                if act.name in self.cached_alphas:
                    self.cached_alphas[act.name].clear()
                self.cached_alphas[act.name] = {}
                if act.alpha is not None:
                    for start_node in act.alpha:
                        # cached alphas and alphas stored in nodes should share the same memory space
                        self.cached_alphas[act.name][start_node] = act.alpha[start_node]
                        # skip slicing if self.preserve_mask is a full set. 
                        ok_to_skip_slicing = self.preserve_mask.numel() == act.alpha[start_node].shape[2]
                        if ok_to_skip_slicing:
                            continue
                        act.alpha[start_node] = act.alpha[start_node][:, :, self.preserve_mask]

    def recover_full_sized_alpha(self, optimizable_activations: list):
        """
        After bound computation, recover the full-sized alphas back to nodes.
        :param optimizable_activations: ist of nodes that may have slope alphas as optimizable variables
        :return: None
        """
        if self.pruning_in_iteration:
            for act in optimizable_activations:
                for start_node in self.cached_alphas[act.name]:
                    act.alpha[start_node] = self.cached_alphas[act.name][start_node]

    def clean_full_sized_alpha_cache(self):
        for act_node in self.cached_alphas:
            self.cached_alphas[act_node].clear()
        self.cached_alphas.clear()
