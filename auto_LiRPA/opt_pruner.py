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
"""Pruning during the optimization."""

import time

import torch


class OptPruner:

    def __init__(self, x, threshold, multi_spec_keep_func, loss_reduction_func,
                 decision_thresh, fix_interm_bounds,
                 epsilon_over_decision_thresh):
        self.x = x
        self.threshold = threshold
        self.multi_spec_keep_func = multi_spec_keep_func
        self.loss_reduction_func = loss_reduction_func
        self.decision_thresh = decision_thresh
        self.fix_interm_bounds = fix_interm_bounds
        self.epsilon_over_decision_thresh = epsilon_over_decision_thresh

        # For computing the positive domain ratio
        self.original_size = x[0].shape[0]
        self.pruning_in_iteration = False
        self.preserve_mask = None
        self.preserve_mask_next = None
        self.time = 0

        # For holding full-sized alphas
        self.cached_alphas = {}

    def prune(self, x, C, ret_l, ret_u, ret, full_l, full_ret_l, full_ret_u,
              full_ret, interm_bounds, aux_reference_bounds,
              stop_criterion_func, bound_lower):
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
            self.get_preserve_mask(ret_l)
            # prune C
            if C is not None and C.shape[0] == x[0].shape[0]:
                C = C[self.now_preserve_mask]  # means C is also batch specific
            # prune x
            x, pre_prune_size = self._prune_x(x)
            # prune bounds
            ret_prune = self._prune_bounds_by_mask(
                ret_l, ret_u, ret,
                interm_bounds, aux_reference_bounds, pre_prune_size)
            full_l, full_ret_l, full_ret_u, full_ret = ret_prune
            self.time += time.time() - stime

        stop_criterion = stop_criterion_func(
            full_ret_l) if bound_lower else stop_criterion_func(-full_ret_u)
        if (type(stop_criterion) != bool and stop_criterion.numel() > 1
                and self.pruning_in_iteration):
            stop_criterion = stop_criterion[self.preserve_mask]

        return (x, C, full_l, full_ret_l, full_ret_u,
                full_ret, stop_criterion)

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

    @torch.no_grad()
    def _prune_bounds_by_mask(self, ret_l, ret_u, ret, interm_bounds,
                              aux_reference_bounds, pre_prune_size):
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
        if interval_to_prune is not None:
            for k, v in interval_to_prune.items():
                interm_interval_l, interm_interval_r = v[0], v[1]
                if interm_interval_l.shape[0] == pre_prune_size:
                    # the first dim is batch size and matches preserve mask
                    interm_interval_l = interm_interval_l[
                        self.now_preserve_mask]
                if interm_interval_r.shape[0] == pre_prune_size:
                    # the first dim is batch size and matches preserve mask
                    interm_interval_r = interm_interval_r[
                        self.now_preserve_mask]
                interval_to_prune[k] = [interm_interval_l, interm_interval_r]

        if aux_reference_bounds is not None:
            for k in aux_reference_bounds:
                aux_ref_l, aux_ref_r = aux_reference_bounds[k]
                if aux_ref_l.shape[0] == pre_prune_size:
                    # the first dim is batch size and matches the preserve mask
                    aux_ref_l = aux_ref_l[self.now_preserve_mask]
                if aux_ref_r.shape[0] == pre_prune_size:
                    # the first dim is batch size and matches the preserve mask
                    aux_ref_r = aux_ref_r[self.now_preserve_mask]
                aux_reference_bounds[k] = [aux_ref_l, aux_ref_r]

        # update the global mask here for possible next iteration
        self.preserve_mask_next = self.preserve_mask[self.now_preserve_mask]

        return full_l, full_ret_l, full_ret_u, full_ret

    @torch.no_grad()
    def get_preserve_mask(self, ret_l):
        """
        Get preserve mask by decision_thresh to filter out the satisfied bounds.
        """
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
            if (isinstance(self.decision_thresh, torch.Tensor)
                    and self.decision_thresh.numel() > 1):
                full_ret = (
                    self.decision_thresh.clone().to(ret.device).type(ret.dtype)
                    + self.epsilon_over_decision_thresh)
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
