from collections import OrderedDict
import torch
from torch import Tensor
from .patches import Patches, inplace_unfold

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


class SparseBeta:
    def __init__(self, shape, bias=False, betas=None, device='cpu'):
        self.device = device
        self.val = torch.zeros(shape)
        self.loc = torch.zeros(shape, dtype=torch.long, device=device)
        self.sign = torch.zeros(shape, device=device)
        self.bias = torch.zeros(shape) if bias else None
        if betas:
            for bi in range(len(betas)):
                if betas[bi] is not None:
                    self.val[bi, :len(betas[bi])] = betas[bi]
        self.val = self.val.detach().to(
            device, non_blocking=True).requires_grad_()

    def apply_splits(self, history, key):
        for bi in range(len(history)):
            # Add history splits. (layer, neuron) is the current decision.
            split_locs, split_coeffs = history[bi][key][:2]
            split_len = len(split_locs)
            if split_len > 0:
                self.sign[bi, :split_len] = torch.as_tensor(
                    split_coeffs, device=self.device)
                self.loc[bi, :split_len] = torch.as_tensor(
                    split_locs, device=self.device)
                if self.bias is not None:
                    split_bias = history[bi][key][2]
                    self.bias[bi, :split_len] = torch.as_tensor(
                        split_bias, device=self.device)
        self.loc = self.loc.to(device=self.device, non_blocking=True)
        self.sign = self.sign.to(device=self.device, non_blocking=True)
        if self.bias is not None:
            self.bias = self.bias.to(device=self.device, non_blocking=True)


def get_split_nodes(self: 'BoundedModule', input_split=False):
    self.split_nodes = []
    self.split_activations = {}
    splittable_activations = self.get_splittable_activations()
    self._set_used_nodes(self[self.final_name])
    for layer in self.layers_requiring_bounds:
        split_activations_ = []
        for activation_name in layer.output_name:
            activation = self[activation_name]
            if activation in splittable_activations:
                split_activations_.append(
                    (activation, activation.inputs.index(layer)))
        if split_activations_:
            self.split_nodes.append(layer)
            self.split_activations[layer.name] = split_activations_
    if input_split:
        root = self[self.root_names[0]]
        if root not in self.split_nodes:
            self.split_nodes.append(root)
            self.split_activations[root.name] = []
    return self.split_nodes, self.split_activations


def set_beta(self: 'BoundedModule', enable_opt_interm_bounds, parameters,
             lr_beta, lr_cut_beta, cutter, dense_coeffs_mask):
    """
    Set betas, best_betas, coeffs, dense_coeffs_mask, best_coeffs, biases
    and best_biases.
    """
    coeffs = None
    betas = []
    best_betas = OrderedDict()

    # TODO compute only once
    self.nodes_with_beta = []
    for node in self.split_nodes:
        if not hasattr(node, 'sparse_betas'):
            continue
        self.nodes_with_beta.append(node)
        if enable_opt_interm_bounds:
            for sparse_beta in node.sparse_betas.values():
                if sparse_beta is not None:
                    betas.append(sparse_beta.val)
            best_betas[node.name] = {
                beta_m: sparse_beta.val.detach().clone()
                for beta_m, sparse_beta in node.sparse_betas.items()
            }
        else:
            betas.append(node.sparse_betas[0].val)
            best_betas[node.name] = node.sparse_betas[0].val.detach().clone()

    # Beta has shape (batch, max_splits_per_layer)
    parameters.append({'params': betas.copy(), 'lr': lr_beta, 'batch_dim': 0})

    if self.cut_used:
        self.set_beta_cuts(parameters, lr_cut_beta, betas, best_betas, cutter)

    return betas, best_betas, coeffs, dense_coeffs_mask


def set_beta_cuts(self: 'BoundedModule', parameters, lr_cut_beta, betas,
                  best_betas, cutter):
    # also need to optimize cut betas
    parameters.append({'params': self.cut_beta_params,
                        'lr': lr_cut_beta, 'batch_dim': 0})
    betas += self.cut_beta_params
    best_betas['cut'] = [beta.detach().clone() for beta in self.cut_beta_params]
    if getattr(cutter, 'opt', False):
        parameters.append(cutter.get_parameters())


def reset_beta(self: 'BoundedModule', node, shape, betas, bias=False,
               start_nodes=None):
    # Create only the non-zero beta. For each layer, it is padded to maximal length.
    # We create tensors on CPU first, and they will be transferred to GPU after initialized.
    if self.bound_opts.get('enable_opt_interm_bounds', False):
        node.sparse_betas = {
            key: SparseBeta(
                shape,
                betas=[(betas[j][i] if betas[j] is not None else None)
                        for j in range(len(betas))],
                device=self.device, bias=bias,
            ) for i, key in enumerate(start_nodes)
        }
    else:
        node.sparse_betas = [SparseBeta(
            shape, betas=betas, device=self.device, bias=bias)]


def beta_crown_backward_bound(self: 'BoundedModule', node, lA, uA, start_node=None):
    """Update A and bias with Beta-CROWN.

    Must be explicitly called at the end of "bound_backward".
    """
    # Regular Beta CROWN with single neuron split
    # Each split constraint only has single neuron (e.g., second ReLU neuron > 0).
    A = lA if lA is not None else uA
    lbias = ubias = 0

    def _bias_unsupported():
        raise NotImplementedError('Bias for beta not supported in this case.')

    if type(A) is Patches:
        if not self.bound_opts.get('enable_opt_interm_bounds', False):
            raise NotImplementedError('Sparse beta not supported in the patches mode')
        if node.sparse_betas[start_node.name].bias is not None:
            _bias_unsupported()
        # expand sparse_beta to full beta
        beta_values = (node.sparse_betas[start_node.name].val
                       * node.sparse_betas[start_node.name].sign)
        beta_indices = node.sparse_betas[start_node.name].loc
        node.masked_beta = torch.zeros(2, *node.shape).reshape(2, -1).to(A.patches.dtype)
        node.non_deter_scatter_add(
            node.masked_beta, dim=1, index=beta_indices,
            src=beta_values.to(node.masked_beta.dtype))
        node.masked_beta = node.masked_beta.reshape(2, *node.shape)
        # unfold the beta as patches, size (batch, out_h, out_w, in_c, H, W)
        A_patches = A.patches
        masked_beta_unfolded = inplace_unfold(
            node.masked_beta, kernel_size=A_patches.shape[-2:],
            padding=A.padding, stride=A.stride,
            inserted_zeros=A.inserted_zeros, output_padding=A.output_padding)
        if A.unstable_idx is not None:
            masked_beta_unfolded = masked_beta_unfolded.permute(1, 2, 0, 3, 4, 5)
            # After selection, the shape is (unstable_size, batch, in_c, H, W).
            masked_beta_unfolded = masked_beta_unfolded[A.unstable_idx[1], A.unstable_idx[2]]
        else:
            # Add the spec (out_c) dimension.
            masked_beta_unfolded = masked_beta_unfolded.unsqueeze(0)
        if node.alpha_beta_update_mask is not None:
            masked_beta_unfolded = masked_beta_unfolded[node.alpha_beta_update_mask]
        if uA is not None:
            uA = uA.create_similar(uA.patches + masked_beta_unfolded)
        if lA is not None:
            lA = lA.create_similar(lA.patches - masked_beta_unfolded)
    elif type(A) is Tensor:
        if self.bound_opts.get('enable_opt_interm_bounds', False):
            if node.sparse_betas[start_node.name].bias is not None:
                _bias_unsupported()
            # For matrix mode, beta is sparse.
            beta_values = (
                node.sparse_betas[start_node.name].val
                * node.sparse_betas[start_node.name].sign
            ).expand(A.size(0), -1, -1)
            # node.single_beta_loc has shape [batch, max_single_split].
            # Need to expand at the specs dimension.
            beta_indices = (node.sparse_betas[start_node.name].loc
                            .unsqueeze(0).expand(A.size(0), -1, -1))
            beta_bias = node.sparse_betas[start_node.name].bias
        else:
            # For matrix mode, beta is sparse.
            beta_values = (
                node.sparse_betas[0].val * node.sparse_betas[0].sign
            ).expand(A.size(0), -1, -1)
            # self.single_beta_loc has shape [batch, max_single_split].
            # Need to expand at the specs dimension.
            beta_indices = node.sparse_betas[0].loc.unsqueeze(0).expand(A.size(0), -1, -1)
            beta_bias = node.sparse_betas[0].bias
        # For conv layer, the last dimension is flattened in indices.
        beta_values = beta_values.to(A.dtype)
        if beta_bias is not None:
            beta_bias = beta_bias.expand(A.size(0), -1, -1)
        if node.alpha_beta_update_mask is not None:
            beta_indices = beta_indices[:, node.alpha_beta_update_mask]
            beta_values = beta_values[:, node.alpha_beta_update_mask]
            if beta_bias is not None:
                beta_bias = beta_bias[:, node.alpha_beta_update_mask]
        if uA is not None:
            uA = node.non_deter_scatter_add(
                uA.reshape(uA.size(0), uA.size(1), -1), dim=2,
                index=beta_indices, src=beta_values).view(uA.size())
        if lA is not None:
            lA = node.non_deter_scatter_add(
                lA.reshape(lA.size(0), lA.size(1), -1), dim=2,
                index=beta_indices, src=beta_values.neg()).view(lA.size())
        if beta_bias is not None:
            bias = (beta_values * beta_bias).sum(dim=-1)
            lbias = bias
            ubias = -bias
    else:
        raise RuntimeError(f"Unknown type {type(A)} for A")

    return lA, uA, lbias, ubias


def print_optimized_beta(acts):
    masked_betas = []
    for model in acts:
        masked_betas.append(model.masked_beta)
        if model.history_beta_used:
            print(f'{model.name} history beta', model.new_history_beta.squeeze())
        if model.split_beta_used:
            print(f'{model.name} split beta:', model.split_beta.view(-1))
            print(f'{model.name} bias:', model.split_bias)
