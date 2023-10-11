"""BoundRelu."""
from typing import Optional, Tuple
import torch
from torch import Tensor
from collections import OrderedDict
from .base import *
from .clampmult import multiply_by_A_signs
from .activation_base import BoundActivation, BoundOptimizableActivation
from .gradient_modules import ReLUGrad
from .solver_utils import grb
from ..utils import unravel_index, prod


class BoundTwoPieceLinear(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.options = options
        self.ibp_intermediate = True
        self.splittable = True
        self.use_sparse_spec_alpha = options.get('sparse_spec_alpha', False)
        self.use_sparse_features_alpha = options.get('sparse_features_alpha', False)
        self.alpha_lookup_idx = self.alpha_indices = None
        self.beta = self.masked_beta = self.sparse_betas = None
        self.split_beta_used = False
        self.history_beta_used = False
        self.flattened_nodes = None
        self.patch_size = {}
        self.cut_used = False
        self.cut_module = None

    def init_opt_parameters(self, start_nodes):
        ref = self.inputs[0].lower # a reference variable for getting the shape
        batch_size = ref.size(0)
        self.alpha = OrderedDict()
        self.alpha_lookup_idx = OrderedDict()  # For alpha with sparse spec dimention.
        self.alpha_indices = None  # indices of non-zero alphas.
        verbosity = self.options.get('verbosity', 0)

        # Alpha can be sparse in both spec dimension, and the C*H*W dimension.
        # We first deal with the sparse-feature alpha, which is sparse in the
        # C*H*W dimesnion of this layer.
        minimum_sparsity = self.options.get('minimum_sparsity', 0.9)
        if (self.use_sparse_features_alpha
                and hasattr(self.inputs[0], 'lower')
                and hasattr(self.inputs[0], 'upper')):
            # Pre-activation bounds available, we will store the alpha for unstable neurons only.
            # Since each element in a batch can have different unstable neurons,
            # for simplicity we find a super-set using any(dim=0).
            # This can be non-ideal if the x in a batch are very different.
            self.get_unstable_idx()
            total_neuron_size = self.inputs[0].lower.numel() // batch_size
            if self.alpha_indices[0].size(0) <= minimum_sparsity * total_neuron_size:
                # Shape is the number of unstable neurons in this layer.
                alpha_shape = [self.alpha_indices[0].size(0)]
                # Skip the batch, spec dimension, and find the lower slopes for all unstable neurons.
                if len(self.alpha_indices) == 1:
                    # This layer is after a linear layer.
                    alpha_init = self.init_d[:, :, self.alpha_indices[0]]
                elif len(self.alpha_indices) == 3:
                    # This layer is after a conv2d layer.
                    alpha_init = self.init_d[
                        :, :, self.alpha_indices[0], self.alpha_indices[1],
                        self.alpha_indices[2]]
                elif len(self.alpha_indices) == 2:
                    # This layer is after a conv1d layer.
                    alpha_init = self.init_d[
                                 :, :, self.alpha_indices[0], self.alpha_indices[1]]
                else:
                    raise ValueError
                if verbosity > 0:
                    print(f'layer {self.name} using sparse-features alpha with shape {alpha_shape}; unstable size '
                          f'{self.alpha_indices[0].size(0)}; total size {total_neuron_size} ({list(ref.shape)})')
            else:
                alpha_shape = self.shape  # Full alpha.
                alpha_init = self.init_d
                if verbosity > 0:
                    print(f'layer {self.name} using full alpha with shape {alpha_shape}; unstable size '
                          f'{self.alpha_indices[0].size(0)}; total size {total_neuron_size} ({list(ref.shape)})')
                self.alpha_indices = None  # Use full alpha.
        else:
            alpha_shape = self.shape  # Full alpha.
            alpha_init = self.init_d
        # Now we start to create alphas for all start nodes.
        # When sparse-spec feature is enabled, alpha is created for only
        # unstable neurons in start node.
        for start_node in start_nodes:
            ns, output_shape, unstable_idx = start_node[:3]
            if isinstance(output_shape, (list, tuple)):
                if len(output_shape) > 1:
                    size_s = prod(output_shape)  # Conv layers.
                else:
                    size_s = output_shape[0]
            else:
                size_s = output_shape
            # unstable_idx may be a tensor (dense layer or conv layer
            # with shared alpha), or tuple of 3-d tensors (conv layer with
            # non-sharing alpha).
            sparsity = float('inf') if unstable_idx is None else unstable_idx.size(0) if isinstance(unstable_idx, torch.Tensor) else unstable_idx[0].size(0)
            if sparsity <= minimum_sparsity * size_s and self.use_sparse_spec_alpha:
                # For fully connected layer, or conv layer with shared alpha per channel.
                # shape is (2, sparse_spec, batch, this_layer_shape)
                # We create sparse specification dimension, where the spec dimension of alpha only includes slopes for unstable neurons in start_node.
                self.alpha[ns] = torch.empty([self.alpha_size, sparsity + 1, batch_size, *alpha_shape],
                                             dtype=torch.float, device=ref.device, requires_grad=True)
                self.alpha[ns].data.copy_(alpha_init.data)  # This will broadcast to (2, sparse_spec) dimensions.
                if verbosity > 0:
                    print(f'layer {self.name} start_node {ns} using sparse-spec alpha {list(self.alpha[ns].size())}'
                          f' with unstable size {sparsity} total_size {size_s} output_shape {output_shape}')
                # unstable_idx is a list of used neurons (or channels for BoundConv) for the start_node.
                assert unstable_idx.ndim == 1 if isinstance(unstable_idx, torch.Tensor) else unstable_idx[0].ndim == 1
                # We only need to the alpha for the unstable neurons in start_node.
                indices = torch.arange(1, sparsity + 1, device=alpha_init.device, dtype=torch.long)
                if isinstance(output_shape, int) or len(output_shape) == 1:
                    # Fully connected layers, or conv layer in patches mode with partially shared alpha (pixels in the same channel use the same alpha).
                    self.alpha_lookup_idx[ns] = torch.zeros(size_s, dtype=torch.long, device=alpha_init.device)
                    # This lookup table maps the unstable_idx to the actual alpha location in self.alpha[ns].
                    # Note that self.alpha[ns][:,0] is reserved for any unstable neurons that are not found in the lookup table. This usually should not
                    # happen, unless reference bounds are not properly set.
                    self.alpha_lookup_idx[ns].data[unstable_idx] = indices
                else:
                    # conv layer in matrix mode, or in patches mode but with non-shared alpha. The lookup table is 3-d.
                    assert len(output_shape) == 3
                    self.alpha_lookup_idx[ns] = torch.zeros(output_shape, dtype=torch.long, device=alpha_init.device)
                    if isinstance(unstable_idx, torch.Tensor):
                        # Convert the unstable index from flattend 1-d to 3-d. (matrix mode).
                        unstable_idx_3d = unravel_index(unstable_idx, output_shape)
                    else:
                        # Patches mode with non-shared alpha, unstable_idx is already 3d.
                        unstable_idx_3d = unstable_idx
                    # Build look-up table.
                    self.alpha_lookup_idx[ns].data[unstable_idx_3d[0], unstable_idx_3d[1], unstable_idx_3d[2]] = indices
            else:
                # alpha shape is (2, spec, batch, this_layer_shape). "this_layer_shape" may still be sparse.
                self.alpha[ns] = torch.empty([self.alpha_size, size_s, batch_size, *alpha_shape],
                                             dtype=torch.float, device=ref.device, requires_grad=True)
                self.alpha[ns].data.copy_(alpha_init.data)  # This will broadcast to (2, spec) dimensions
                if verbosity > 0:
                    print(f'layer {self.name} start_node {ns} using full alpha {list(self.alpha[ns].size())} with unstable '
                          f'size {sparsity if unstable_idx is not None else None} total_size {size_s} output_shape {output_shape}')
                # alpha_lookup_idx can be used for checking if sparse alpha is used or not.
                self.alpha_lookup_idx[ns] = None

    def select_alpha_by_idx(self, last_lA, last_uA, unstable_idx, start_node, alpha_lookup_idx):
        # Each alpha has shape (2, output_shape, batch_size, *relu_node_shape].
        # If slope is shared, output_shape will be 1.
        # The *relu_node_shape might be sparse (sparse-feature alpha), where the non-zero values are indicated by self.alpha_indices.
        # The out_shape might be sparse (sparse-spec alpha), where the non-zero values are indexed by self.alpha_lookup_idx.
        if unstable_idx is not None:
            # print(f'relu layer {self.name}, start_node {start_node}, unstable_idx {type(unstable_idx)} alpha idx {self.alpha_lookup_idx[start_node.name].size()}')
            if self.alpha_lookup_idx is not None:
                alpha_lookup_idx = self.alpha_lookup_idx[start_node.name]
            else:
                alpha_lookup_idx = None
            if isinstance(unstable_idx, tuple):
                # Start node is a conv node.
                selected_alpha = self.alpha[start_node.name]
                if isinstance(last_lA, Tensor) or isinstance(last_uA, Tensor):
                    # Start node is a conv node but we received tensors as A matrices.
                    # Patches mode converted to matrix, or matrix mode used. Need to select accross the spec dimension.
                    # For this node, since it is in matrix mode, the spec dimension is out_c * out_h * out_w
                    # Shape is [2, spec, batch, *this_layer_shape]
                    if alpha_lookup_idx is None:
                        if self.options['optimize_bound_args'].get('use_shared_alpha', False):
                            # alpha is shared, and its spec dimension is always 1. In this case we do not need to select.
                            # selected_alpha will have shape [2, 1, batch, *this_layer_shape]
                            pass
                        else:
                            # alpha is not shared, so it has shape [2, spec, batch, *this_layer_shape]
                            # Reshape the spec dimension to c*h*w so we can select used alphas based on unstable index.
                            # Shape becomes [2, out_c, out_h, out_w, batch, *this_layer_shape]
                            selected_alpha = selected_alpha.view(selected_alpha.size(0), *start_node.output_shape[1:], *selected_alpha.shape[2:])
                            selected_alpha = selected_alpha[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]]
                    else:
                        assert alpha_lookup_idx.ndim == 3
                        # We only stored some alphas, and A is also sparse, so the unstable_idx must be first translated to real indices.
                        # alpha shape is (2, sparse_spec_shape, batch_size, *relu_node_shape) where relu_node_shape can also be sparse.
                        # We use sparse-spec alphas. Need to convert these unstable_idx[0], unstable_idx[1], unstable_idx[0] using lookup table.
                        _unstable_idx = alpha_lookup_idx[unstable_idx[0], unstable_idx[1], unstable_idx[2]]
                        selected_alpha = self.non_deter_index_select(selected_alpha, index=_unstable_idx, dim=1)
                else:
                    # Patches mode. Alpha must be selected after unfolding, so cannot be done here.
                    # Selection is deferred to maybe_unfold() using alpha_lookup_idx.
                    # For partially shared alpha, its shape is (2, out_c, batch_size, *relu_node_shape).
                    # For full alpha, its shape is (2, out_c*out_h*out_w, batch_size, *relu_node_shape).
                    # Both the spec dimension and relu_node_shape dimensions can be sparse.
                    pass
            elif unstable_idx.ndim == 1:
                # Start node is a FC node.
                # Only unstable neurons of the start_node neurons are used.
                assert alpha_lookup_idx is None or alpha_lookup_idx.ndim == 1
                if self.options['optimize_bound_args'].get('use_shared_alpha', False):
                    # Shared alpha is used, all output specs use the same alpha. No selection is needed.
                    # The spec dim is 1 and will be broadcast.
                    selected_alpha = self.alpha[start_node.name]
                else:
                    _unstable_idx = alpha_lookup_idx[unstable_idx] if alpha_lookup_idx is not None else unstable_idx
                    selected_alpha = self.non_deter_index_select(self.alpha[start_node.name], index=_unstable_idx, dim=1)
            elif unstable_idx.ndim == 2:
                assert alpha_lookup_idx is None, "sparse spec alpha has not been implemented yet."
                # Each element in the batch selects different neurons.
                selected_alpha = batched_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
            else:
                raise ValueError
        else:
            # Spec dimension is dense. Alpha must not be created sparsely.
            assert self.alpha_lookup_idx is None or self.alpha_lookup_idx[start_node.name] is None
            selected_alpha = self.alpha[start_node.name]
        return selected_alpha, alpha_lookup_idx

    def reconstruct_full_alpha(self, sparse_alpha, full_alpha_shape, alpha_indices):
        full_alpha = torch.zeros(full_alpha_shape, dtype=sparse_alpha.dtype, device=sparse_alpha.device)
        if len(alpha_indices) == 1:
            # Relu after a dense layer.
            full_alpha[:, :, alpha_indices[0]] = sparse_alpha
        elif len(alpha_indices) == 3:
            # Relu after a conv2d layer.
            full_alpha[:, :, alpha_indices[0], alpha_indices[1], alpha_indices[2]] = sparse_alpha
        elif len(alpha_indices) == 2:
            # Relu after a conv1d layer.
            full_alpha[:, :, alpha_indices[0], alpha_indices[1]] = sparse_alpha
        else:
            raise ValueError
        return full_alpha

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None,
                       unstable_idx=None, reduce_bias=True, **kwargs):
        """
        start_node: the name of the layer where the backward bound propagation starts.
                    Can be the output layer or an intermediate layer.
        unstable_idx: indices for the unstable neurons, whose bounds need to be computed.
                      Either be a tuple (for patches) or a 1-D tensor.
        """
        # Usage of output constraints requires access to bounds of the previous iteration
        # (see _clear_and_set_new)
        apply_output_constraints_to = self.options["optimize_bound_args"]["apply_output_constraints_to"]
        if hasattr(x, "lower"):
            lower = x.lower
        else:
            assert start_node.are_output_constraints_activated_for_layer(apply_output_constraints_to)
            lower = x.previous_iteration_lower
        if hasattr(x, "upper"):
            upper = x.upper
        else:
            assert start_node.are_output_constraints_activated_for_layer(apply_output_constraints_to)
            upper = x.previous_iteration_upper
        # Get element-wise CROWN linear relaxations.
        (upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d,
            lb_upper_d, ub_upper_d, alpha_lookup_idx) = \
            self._backward_relaxation(last_lA, last_uA, x, start_node, unstable_idx)
        # save for calculate babsr score
        self.d = upper_d
        self.lA = last_lA
        # Save for initialization bounds.
        self.init_d = lower_d

        # Choose upper or lower bounds based on the sign of last_A
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0
            # Obtain the new linear relaxation coefficients based on the signs in last_A.
            _A, _bias = multiply_by_A_signs(
                last_A, d_pos, d_neg, b_pos, b_neg, reduce_bias=reduce_bias)
            if isinstance(last_A, Patches):
                # Save the patch size, which will be used in init_alpha() to determine the number of optimizable parameters.
                A_prod = _A.patches
                if start_node is not None:
                    if last_A.unstable_idx is not None:
                        # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                        self.patch_size[start_node.name] = [
                            last_A.output_shape[1], A_prod.size(1),
                            last_A.output_shape[2], last_A.output_shape[3],
                            A_prod.size(-3), A_prod.size(-2), A_prod.size(-1)]
                    else:
                        # Regular patches.
                        self.patch_size[start_node.name] = A_prod.size()
            return _A, _bias

        ######## A problem with patches mode for cut constraint start ##########
        # There are cases that  the node that is in the constraint but not selected by the patches for the output node
        # trick: only count the small patches that have all the split node coeffs[ci].sum() equal to coeffs_unfolded[ci][out_h, out_w, -1].sum()
        # we should force these beta to be 0 to disable the effect of these constraints
        A = last_lA if last_lA is not None else last_uA
        current_layer_shape = lower.size()[1:]
        if self.cut_used and type(A) is Patches:
            self.cut_module.patch_trick(start_node, self.name, A, current_layer_shape)
        ######## A problem with patches mode for cut constraint end ##########

        if self.cut_used:
            if self.leaky_alpha > 0:
                raise NotImplementedError
            # propagate postrelu node in cut constraints
            last_lA, last_uA = self.cut_module.relu_cut(
                start_node, self.name, last_lA, last_uA, current_layer_shape,
                unstable_idx, batch_mask=self.inputs[0].alpha_beta_update_mask)

        # In patches mode we might need an unfold.
        # lower_d, upper_d, lower_b, upper_b: 1, batch, current_c, current_w, current_h or None
        upper_d = maybe_unfold_patches(upper_d, last_lA if last_lA is not None else last_uA)
        lower_d = maybe_unfold_patches(lower_d, last_lA if last_lA is not None else last_uA)
        upper_b = maybe_unfold_patches(upper_b, last_lA if last_lA is not None else last_uA)
        lower_b = maybe_unfold_patches(lower_b, last_lA if last_lA is not None else last_uA)  # for ReLU it is always None; keeping it here for completeness.
        # ub_lower_d and lb_lower_d might have sparse spec dimension, so they may need alpha_lookup_idx to convert to actual spec dim.
        ub_lower_d = maybe_unfold_patches(ub_lower_d, last_uA, alpha_lookup_idx=alpha_lookup_idx)
        ub_upper_d = maybe_unfold_patches(ub_upper_d, last_uA, alpha_lookup_idx=alpha_lookup_idx)
        # optimizable slope lb_lower_d: spec (only channels in spec layer), batch, current_c, current_w, current_h
        # patches mode lb_lower_d after unfold: unstable, batch, in_C, H, W
        lb_lower_d = maybe_unfold_patches(lb_lower_d, last_lA, alpha_lookup_idx=alpha_lookup_idx)
        lb_upper_d = maybe_unfold_patches(lb_upper_d, last_lA, alpha_lookup_idx=alpha_lookup_idx)

        if self.cut_used:
            assert reduce_bias
            I = (lower < 0) * (upper > 0)
            # propagate integer var of relu neuron (arelu) in cut constraints through relu layer
            lA, uA, lbias, ubias = self.cut_module.arelu_cut(
                start_node, self.name, last_lA, last_uA, lower_d, upper_d,
                lower_b, upper_b, lb_lower_d, ub_lower_d, I, x, self.patch_size,
                current_layer_shape, unstable_idx,
                batch_mask=self.inputs[0].alpha_beta_update_mask)
        else:
            uA, ubias = _bound_oneside(
                last_uA, ub_upper_d if upper_d is None else upper_d,
                ub_lower_d if lower_d is None else lower_d, upper_b, lower_b)
            lA, lbias = _bound_oneside(
                last_lA, lb_lower_d if lower_d is None else lower_d,
                lb_upper_d if upper_d is None else upper_d, lower_b, upper_b)

        if self.cut_used:
            # propagate prerelu node in cut constraints
            lA, uA = self.cut_module.pre_cut(
                start_node, self.name, lA, uA, current_layer_shape, unstable_idx,
                batch_mask=self.inputs[0].alpha_beta_update_mask)
        self.masked_beta_lower = self.masked_beta_upper = None

        return [(lA, uA)], lbias, ubias

    def dump_optimized_params(self):
        ret = {'alpha': self.alpha}
        if self.use_sparse_spec_alpha:
            ret['alpha_lookup_idx'] = self.alpha_lookup_idx
        if self.use_sparse_features_alpha:
            ret['alpha_indices'] = self.alpha_indices
        return ret

    def restore_optimized_params(self, alpha):
        self.alpha = alpha['alpha']
        if self.use_sparse_spec_alpha:
            self.alpha_lookup_idx = alpha['alpha_lookup_idx']
        if self.use_sparse_features_alpha:
            self.alpha_indices = alpha['alpha_indices']


class BoundRelu(BoundTwoPieceLinear):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.relu_options = options.get('relu', 'adaptive')  # FIXME: use better names.
        self.leaky_alpha = attr.get('alpha', 0)
        self.alpha_size = 2
        # Alpha dimension is (2, output_shape, batch, *shape) for ReLU.

    def get_unstable_idx(self):
        self.alpha_indices = torch.logical_and(
            self.inputs[0].lower < 0, self.inputs[0].upper > 0).any(dim=0).nonzero(as_tuple=True)

    def clip_alpha(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, self.leaky_alpha, 1.)

    def forward(self, x):
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        if self.leaky_alpha > 0:
            return F.leaky_relu(x, negative_slope=self.leaky_alpha)
        else:
            return F.relu(x)

    def _relu_lower_bound_init(self, upper_k):
        """Return the initial lower bound without relaxation."""
        if self.relu_options == "same-slope":
            # the same slope for upper and lower
            lower_k = upper_k
        elif self.relu_options == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_k = torch.zeros_like(upper_k)
            lower_k = (upper_k >= 1.0).to(upper_k)
            if self.leaky_alpha > 0:
                lower_k += (upper_k < 1.0).to(upper_k) * self.leaky_alpha
        elif self.relu_options == "one-lb":
            # Always use slope 1 as lower bound
            lower_k = ((upper_k > self.leaky_alpha).to(upper_k)
                       + (upper_k <= self.leaky_alpha).to(upper_k)
                          * self.leaky_alpha)
        else:
            # adaptive
            if self.leaky_alpha == 0:
                lower_k = (upper_k > 0.5).to(upper_k)
            else:
                # FIXME this may not be optimal for leaky relu
                lower_k = ((upper_k > 0.5).to(upper_k)
                           + (upper_k <= 0.5).to(upper_k) * self.leaky_alpha)
        return lower_k

    def _forward_relaxation(self, x):
        self._init_masks(x)
        self.mask_pos = self.mask_pos.to(x.lower)
        self.mask_both = self.mask_both.to(x.lower)

        upper_k, upper_b = self._relu_upper_bound(
            x.lower, x.upper, self.leaky_alpha)
        self.uw = self.mask_pos + self.mask_both * upper_k
        self.ub = self.mask_both * upper_b

        if self.opt_stage in ['opt', 'reuse']:
            # Each actual alpha in the forward mode has shape (batch_size, *relu_node_shape].
            # But self.alpha has shape (2, output_shape, batch_size, *relu_node_shape]
            # and we do not need its first two dimensions.
            lower_k = self.alpha['_forward'][0, 0]
        else:
            lower_k = self._relu_lower_bound_init(upper_k)

        # NOTE #FIXME Saved for initialization bounds for optimization.
        # In the backward mode, same-slope bounds are used.
        # But here it is using adaptive bounds which seem to be better
        # for nn4sys benchmark with loose input bounds. Need confirmation
        # for other cases.
        self.lower_d = lower_k.detach() # saved for initializing optimized bounds

        self.lw = self.mask_both * lower_k + self.mask_pos

    def bound_dynamic_forward(self, x, max_dim=None, offset=0):
        if self.leaky_alpha > 0:
            raise NotImplementedError

        self._init_masks(x)
        self.mask_pos = self.mask_pos.to(x.lower)
        self.mask_both = self.mask_both.to(x.lower)

        upper_k, upper_b = self._relu_upper_bound(
            x.lower, x.upper, self.leaky_alpha)
        w_new = (self.mask_pos.unsqueeze(1) * x.lw
            + self.mask_both.unsqueeze(1) * upper_k.unsqueeze(1) * x.lw)
        upper_b = self.mask_both * upper_b / 2
        b_new = (self.mask_pos * x.lb
            + self.mask_both * upper_k * x.lb + upper_b)

        # Create new variables for unstable ReLU
        batch_size = w_new.shape[0]
        device = w_new.device
        unstable = self.mask_both.view(batch_size, -1)
        tot_unstable = int(unstable.sum(dim=-1).max())
        tot_dim = x.tot_dim + tot_unstable

        if offset + w_new.shape[1] < x.tot_dim:
            return LinearBound(
                w_new, b_new, w_new, b_new, x_L=x.x_L, x_U=x.x_U, tot_dim=tot_dim)

        index = torch.cumsum(unstable, dim=-1).to(torch.int64)
        index = (index - (offset + w_new.shape[1] - x.tot_dim)).clamp(min=0)
        num_new_dim = int(index.max())
        num_new_dim_actual = min(num_new_dim, max_dim - w_new.shape[1])
        index = index.clamp(max=num_new_dim_actual+1)
        w_unstable = torch.zeros(batch_size, num_new_dim_actual + 2, unstable.size(-1), device=device)
        x_L_unstable = -torch.ones(batch_size, num_new_dim_actual, device=device)
        x_U_unstable = torch.ones(batch_size, num_new_dim_actual, device=device)
        w_unstable.scatter_(dim=1, index=index.unsqueeze(1), src=upper_b.view(batch_size, 1, -1), reduce='add')
        w_unstable = w_unstable[:, 1:-1].view(batch_size, num_new_dim_actual, *w_new.shape[2:])

        w_new = torch.cat([w_new, w_unstable], dim=1)
        x_L_new = torch.cat([x.x_L, x_L_unstable], dim=-1)
        x_U_new = torch.cat([x.x_U, x_U_unstable], dim=-1)

        return LinearBound(
            w_new, b_new, w_new, b_new, x_L=x_L_new, x_U=x_U_new, tot_dim=tot_dim)

    def bound_forward(self, dim_in, x):
        self._forward_relaxation(x)
        lb = self.lw * x.lb
        ub = self.uw * x.ub + self.ub
        lw = (self.lw.unsqueeze(1) * x.lw) if x.lw is not None else None
        uw = (self.uw.unsqueeze(1) * x.uw) if x.uw is not None else None
        if not lw.requires_grad:
            del self.mask_both, self.mask_pos
            del self.lw, self.uw, self.ub
        return LinearBound(lw, lb, uw, ub)

    @staticmethod
    @torch.jit.script
    def _relu_upper_bound(lb, ub, leaky_alpha: float):
        """Upper bound slope and intercept according to CROWN relaxation."""
        lb_r = lb.clamp(max=0)
        ub_r = ub.clamp(min=0)
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        if leaky_alpha > 0:
            upper_d = (ub_r - leaky_alpha * lb_r) / (ub_r - lb_r)
            upper_b = - lb_r * upper_d + leaky_alpha * lb_r
        else:
            upper_d = ub_r / (ub_r - lb_r)
            upper_b = - lb_r * upper_d
        return upper_d, upper_b

    @staticmethod
    def _relu_mask_alpha(lower, upper, lb_lower_d : Optional[Tensor],
                         ub_lower_d : Optional[Tensor], leaky_alpha : float = 0,
                        ) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        lower_mask = (lower >= 0).requires_grad_(False).to(lower.dtype)
        upper_mask = (upper <= 0).requires_grad_(False)
        if leaky_alpha > 0:
            zero_coeffs = False
        else:
            zero_coeffs = upper_mask.all()
        no_mask = (1. - lower_mask) * (1. - upper_mask.to(upper.dtype))
        if lb_lower_d is not None:
            lb_lower_d = (
                torch.clamp(lb_lower_d, min=leaky_alpha, max=1.) * no_mask
                + lower_mask)
            if leaky_alpha > 0:
                lb_lower_d += upper_mask * leaky_alpha
        if ub_lower_d is not None:
            ub_lower_d = (
                torch.clamp(ub_lower_d, min=leaky_alpha, max=1.) * no_mask
                + lower_mask)
            if leaky_alpha > 0:
                ub_lower_d += upper_mask * leaky_alpha
        return lb_lower_d, ub_lower_d, zero_coeffs

    def _backward_relaxation(self, last_lA, last_uA, x, start_node, unstable_idx):
        # Usage of output constraints requires access to bounds of the previous iteration
        # (see _clear_and_set_new)
        if x is not None:
            apply_output_constraints_to = self.options['optimize_bound_args']['apply_output_constraints_to']
            if hasattr(x, "lower"):
                lower = x.lower
            else:
                assert start_node.are_output_constraints_activated_for_layer(apply_output_constraints_to)
                lower = x.previous_iteration_lower
            if hasattr(x, "upper"):
                upper = x.upper
            else:
                assert start_node.are_output_constraints_activated_for_layer(apply_output_constraints_to)
                upper = x.previous_iteration_upper
        else:
            lower = self.lower
            upper = self.upper

        # Upper bound slope and intercept according to CROWN relaxation.
        upper_d, upper_b = self._relu_upper_bound(lower, upper, self.leaky_alpha)

        flag_expand = False
        ub_lower_d = lb_lower_d = None
        lower_b = None  # ReLU does not have lower bound intercept (=0).
        alpha_lookup_idx = None  # For sparse-spec alpha.
        if self.opt_stage in ['opt', 'reuse']:
            # Alpha-CROWN.
            lower_d = None
            selected_alpha, alpha_lookup_idx = self.select_alpha_by_idx(last_lA, last_uA,
                unstable_idx, start_node, alpha_lookup_idx)
            # The first dimension is lower/upper intermediate bound.
            if last_lA is not None:
                lb_lower_d = selected_alpha[0]
            if last_uA is not None:
                ub_lower_d = selected_alpha[1]

            if self.alpha_indices is not None:
                # Sparse alpha on the hwc dimension. We store slopes for unstable neurons in this layer only.
                # Recover to full alpha first.
                sparse_alpha_shape = lb_lower_d.shape if lb_lower_d is not None else ub_lower_d.shape
                full_alpha_shape = sparse_alpha_shape[:-1] + self.shape
                if lb_lower_d is not None:
                    lb_lower_d = self.reconstruct_full_alpha(
                        lb_lower_d, full_alpha_shape, self.alpha_indices)
                if ub_lower_d is not None:
                    ub_lower_d = self.reconstruct_full_alpha(
                        ub_lower_d, full_alpha_shape, self.alpha_indices)

            lb_lower_d, ub_lower_d, zero_coeffs = self._relu_mask_alpha(lower, upper, lb_lower_d, ub_lower_d)
            self.zero_backward_coeffs_l = self.zero_backward_coeffs_u = zero_coeffs
            flag_expand = True  # we already have the spec dimension.
        else:
            # FIXME: the shape can be incorrect if unstable_idx is not None.
            # This will cause problem if some ReLU layers are optimized, some are not.
            lower_d = self._relu_lower_bound_init(upper_d)

        # Upper bound always needs an extra specification dimension, since they only depend on lb and ub.
        upper_d = upper_d.unsqueeze(0)
        upper_b = upper_b.unsqueeze(0)
        if not flag_expand:
            if self.opt_stage in ['opt', 'reuse']:
                # We have different slopes for lower and upper bounds propagation.
                lb_lower_d = lb_lower_d.unsqueeze(0) if last_lA is not None else None
                ub_lower_d = ub_lower_d.unsqueeze(0) if last_uA is not None else None
            else:
                lower_d = lower_d.unsqueeze(0)
        return (upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d,
            None, None, alpha_lookup_idx)

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        return self.forward(h_L), self.forward(h_U)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if self.leaky_alpha > 0:
            raise NotImplementedError

        # e.g., last layer input gurobi vars (8,16,16)
        gvars_array = np.array(v[0])
        this_layer_shape = gvars_array.shape
        assert gvars_array.shape == self.output_shape[1:]

        pre_lbs = self.inputs[0].lower.cpu().detach().numpy().reshape(-1)
        pre_ubs = self.inputs[0].upper.cpu().detach().numpy().reshape(-1)

        new_layer_gurobi_vars = []
        relu_integer_vars = []
        new_relu_layer_constrs = []
        # predefined zero variable shared in the whole solver model
        zero_var = model.getVarByName("zero")

        for neuron_idx, pre_var in enumerate(gvars_array.reshape(-1)):
            pre_ub = pre_ubs[neuron_idx]
            pre_lb = pre_lbs[neuron_idx]

            if pre_lb >= 0:
                # ReLU is always passing
                var = pre_var
            elif pre_ub <= 0:
                var = zero_var
            else:
                ub = pre_ub

                var = model.addVar(ub=ub, lb=pre_lb,
                                   obj=0,
                                   vtype=grb.GRB.CONTINUOUS,
                                   name=f'ReLU{self.name}_{neuron_idx}')

                if model_type == "mip" or model_type == "lp_integer":
                    # binary indicator
                    if model_type == "mip":
                        a = model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{self.name}_{neuron_idx}')
                    elif model_type == "lp_integer":
                        a = model.addVar(ub=1, lb=0, vtype=grb.GRB.CONTINUOUS, name=f'aReLU{self.name}_{neuron_idx}')
                    relu_integer_vars.append(a)

                    new_relu_layer_constrs.append(
                        model.addConstr(pre_var - pre_lb * (1 - a) >= var,
                                        name=f'ReLU{self.name}_{neuron_idx}_a_0'))
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= pre_var, name=f'ReLU{self.name}_{neuron_idx}_a_1'))
                    new_relu_layer_constrs.append(
                        model.addConstr(pre_ub * a >= var, name=f'ReLU{self.name}_{neuron_idx}_a_2'))
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= 0, name=f'ReLU{self.name}_{neuron_idx}_a_3'))

                elif model_type == "lp":
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= 0, name=f'ReLU{self.name}_{neuron_idx}_a_0'))
                    new_relu_layer_constrs.append(
                        model.addConstr(var >= pre_var, name=f'ReLU{self.name}_{neuron_idx}_a_1'))
                    new_relu_layer_constrs.append(model.addConstr(
                        pre_ub * pre_var - (pre_ub - pre_lb) * var >= pre_ub * pre_lb,
                        name=f'ReLU{self.name}_{neuron_idx}_a_2'))

                else:
                    print(f"gurobi model type {model_type} not supported!")

            new_layer_gurobi_vars.append(var)

        new_layer_gurobi_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape).tolist()
        if model_type in ["mip", "lp_integer"]:
            self.integer_vars = relu_integer_vars
        self.solver_vars = new_layer_gurobi_vars
        self.solver_constrs = new_relu_layer_constrs
        model.update()

    def build_gradient_node(self, grad_upstream):
        if self.leaky_alpha > 0:
            raise NotImplementedError
        node_grad = ReLUGrad()
        grad_input = (grad_upstream, self.inputs[0].forward_value)
        # An extra node is needed to consider the state of ReLU activation
        grad_extra_nodes = [self.inputs[0]]
        return node_grad, grad_input, grad_extra_nodes

    def get_split_mask(self, lower, upper, input_index):
        assert input_index == 0
        return torch.logical_and(lower < 0, upper > 0)


class BoundLeakyRelu(BoundRelu):
    pass


class BoundSign(BoundActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True

    def forward(self, x):
        return torch.sign(x)

    def bound_relax(self, x, init=False):
        if init:
            self.init_linear_relaxation(x)
        mask_0 = torch.logical_and(x.lower == 0, x.upper == 0)
        mask_pos_0 = torch.logical_and(x.lower == 0, x.upper > 0)
        mask_neg_0 = torch.logical_and(x.lower < 0, x.upper == 0)
        mask_pos = x.lower > 0
        mask_neg = x.upper < 0
        mask_both = torch.logical_not(torch.logical_or(torch.logical_or(
            mask_0, torch.logical_or(mask_pos, mask_pos_0)),
            torch.logical_or(mask_neg, mask_neg_0)))
        self.add_linear_relaxation(mask=mask_0, type='lower',
            k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=0)
        self.add_linear_relaxation(mask=mask_0, type='upper',
            k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=0)

        self.add_linear_relaxation(mask=mask_pos_0, type='lower',
            k=1/x.upper.clamp(min=1e-8), x0=torch.zeros_like(x.upper), y0=0)
        self.add_linear_relaxation(mask=torch.logical_or(mask_pos_0, mask_pos), type='upper',
            k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=1)

        self.add_linear_relaxation(mask=torch.logical_or(mask_neg_0, mask_neg), type='lower',
            k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=-1)
        self.add_linear_relaxation(mask=mask_neg_0, type='upper',
            k=-1/x.lower.clamp(max=-1e-8), x0=torch.zeros_like(x.upper), y0=0)

        self.add_linear_relaxation(mask=mask_pos, type='lower', k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=1)
        self.add_linear_relaxation(mask=mask_neg, type='upper', k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=-1)
        self.add_linear_relaxation(mask=mask_both, type='lower', k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=-1)
        self.add_linear_relaxation(mask=mask_both, type='upper', k=0, x0=torch.zeros_like(x.upper, requires_grad=True), y0=1)


class SignMergeFunction_loose(torch.autograd.Function):
    # Modified SignMerge operator.
    # Change its backward function so that the "gradient" can be used for pgd attack
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(torch.sign(input) + 1e-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        eps = 5     # should be carefully chosen
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[abs(input) >= eps] = 0
        grad_input /= eps
        return grad_input

class SignMergeFunction_tight(torch.autograd.Function):
    # Modified SignMerge operator.
    # Change its backward function so that the "gradient" can be used for pgd attack
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(torch.sign(input) + 1e-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        eps = 0.1     # should be carefully chosen
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[abs(input) >= eps] = 0
        grad_input /= eps
        return grad_input


class BoundSignMerge(BoundTwoPieceLinear):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.alpha_size = 4
        self.loose_function = SignMergeFunction_loose
        self.tight_function = SignMergeFunction_tight
        self.signmergefunction = self.tight_function    # default

    def get_unstable_idx(self):
        self.alpha_indices = torch.logical_and(
            self.inputs[0].lower < 0, self.inputs[0].upper >= 0).any(dim=0).nonzero(as_tuple=True)

    def forward(self, x):
        self.shape = x.shape[1:]
        return self.signmergefunction.apply(x)

    def _mask_alpha(self, lower, upper, lb_lower_d, ub_lower_d, lb_upper_d, ub_upper_d):
        lower_mask = (lower >= 0.).requires_grad_(False).to(lower.dtype)
        upper_mask = (upper < 0.).requires_grad_(False).to(upper.dtype)
        no_mask = 1. - (lower_mask + upper_mask)
        if lb_lower_d is not None:
            lb_lower_d = torch.min(lb_lower_d, 2/upper.clamp(min=1e-8))
            lb_lower_d = torch.clamp(lb_lower_d, min=0) * no_mask
            lb_upper_d = torch.min(lb_upper_d, -2/lower.clamp(max=-1e-8))
            lb_upper_d = torch.clamp(lb_upper_d, min=0) * no_mask
        if ub_lower_d is not None:
            ub_lower_d = torch.min(ub_lower_d, 2/upper.clamp(min=1e-8))
            ub_lower_d = torch.clamp(ub_lower_d, min=0) * no_mask
            ub_upper_d = torch.min(ub_upper_d, -2/lower.clamp(max=-1e-8))
            ub_upper_d = torch.clamp(ub_upper_d, min=0) * no_mask
        return lb_lower_d, ub_lower_d, lb_upper_d, ub_upper_d

    def _backward_relaxation(self, last_lA, last_uA, x, start_node, unstable_idx):
        if x is not None:
            lower, upper = x.lower, x.upper
        else:
            lower, upper = self.lower, self.upper

        flag_expand = False
        ub_lower_d = lb_lower_d = lb_upper_d = ub_upper_d = None
        alpha_lookup_idx = None  # For sparse-spec alpha.
        if self.opt_stage in ['opt', 'reuse']:
            # Alpha-CROWN.
            upper_d = lower_d = None
            selected_alpha, alpha_lookup_idx = self.select_alpha_by_idx(last_lA, last_uA,
                unstable_idx, start_node, alpha_lookup_idx)
            # The first dimension is lower/upper intermediate bound.
            if last_lA is not None:
                lb_lower_d = selected_alpha[0]
                lb_upper_d = selected_alpha[2]
            if last_uA is not None:
                ub_lower_d = selected_alpha[1]
                ub_upper_d = selected_alpha[3]

            if self.alpha_indices is not None:
                # Sparse alpha on the hwc dimension. We store slopes for unstable neurons in this layer only.
                # Recover to full alpha first.
                sparse_alpha_shape = lb_lower_d.shape if lb_lower_d is not None else ub_lower_d.shape
                full_alpha_shape = sparse_alpha_shape[:-1] + self.shape
                if lb_lower_d is not None:
                    lb_lower_d = self.reconstruct_full_alpha(
                        lb_lower_d, full_alpha_shape, self.alpha_indices)
                    lb_upper_d = self.reconstruct_full_alpha(
                        lb_upper_d, full_alpha_shape, self.alpha_indices)
                if ub_lower_d is not None:
                    ub_lower_d = self.reconstruct_full_alpha(
                        ub_lower_d, full_alpha_shape, self.alpha_indices)
                    ub_upper_d = self.reconstruct_full_alpha(
                        ub_upper_d, full_alpha_shape, self.alpha_indices)

            # condition only on the masked part
            if self.inputs[0].alpha_beta_update_mask is not None:
                update_mask = self.inputs[0].alpha_beta_update_mask
                if lb_lower_d is not None:
                    lb_lower_d_new = lb_lower_d[:, update_mask]
                    lb_upper_d_new = lb_upper_d[:, update_mask]
                else:
                    lb_lower_d_new = lb_upper_d_new = None
                if ub_lower_d is not None:
                    ub_lower_d_new = ub_lower_d[:, update_mask]
                    ub_upper_d_new = ub_upper_d[:, update_mask]
                else:
                    ub_lower_d_new = ub_upper_d_new = None
                lb_lower_d, ub_lower_d, lb_upper_d, ub_upper_d = self._mask_alpha(lower, upper,
                    lb_lower_d_new, ub_lower_d_new, lb_upper_d_new, ub_upper_d_new)
            else:
                lb_lower_d, ub_lower_d, lb_upper_d, ub_upper_d = self._mask_alpha(lower, upper,
                    lb_lower_d, ub_lower_d, lb_upper_d, ub_upper_d)
            flag_expand = True  # we already have the spec dimension.
        else:
            lower_d = torch.zeros_like(upper, requires_grad=True)
            upper_d = torch.zeros_like(upper, requires_grad=True)

        mask_pos = (x.lower >= 0.).requires_grad_(False).to(x.lower.dtype)
        mask_neg = (x.upper < 0.).requires_grad_(False).to(x.upper.dtype)
        lower_b = (-1 * (1 - mask_pos) + mask_pos).unsqueeze(0)
        upper_b = (-1 * mask_neg + (1 - mask_neg)).unsqueeze(0)

        # Upper bound always needs an extra specification dimension, since they only depend on lb and ub.
        if not flag_expand:
            if self.opt_stage in ['opt', 'reuse']:
                # We have different slopes for lower and upper bounds propagation.
                lb_lower_d = lb_lower_d.unsqueeze(0) if last_lA is not None else None
                ub_lower_d = ub_lower_d.unsqueeze(0) if last_uA is not None else None
                lb_upper_d = lb_lower_d.unsqueeze(0) if last_lA is not None else None
                ub_upper_d = ub_lower_d.unsqueeze(0) if last_uA is not None else None
            else:
                lower_d = lower_d.unsqueeze(0)
                upper_d = upper_d.unsqueeze(0)
        return (upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d,
            lb_upper_d, ub_upper_d, alpha_lookup_idx)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):

        # e.g., last layer input gurobi vars (8,16,16)
        gvars_array = np.array(v[0])
        this_layer_shape = gvars_array.shape
        assert gvars_array.shape == self.output_shape[1:]

        pre_lbs = self.inputs[0].lower.cpu().detach().numpy().reshape(-1)
        pre_ubs = self.inputs[0].upper.cpu().detach().numpy().reshape(-1)

        new_layer_gurobi_vars = []
        integer_vars = []
        layer_constrs = []
        # predefined zero variable shared in the whole solver model
        one_var = model.getVarByName("one")
        neg_one_var = model.getVarByName("neg_one")

        for neuron_idx, pre_var in enumerate(gvars_array.reshape(-1)):
            pre_ub = pre_ubs[neuron_idx]
            pre_lb = pre_lbs[neuron_idx]

            if pre_lb >= 0:
                var = one_var
            elif pre_ub < 0:
                var = neg_one_var
            else:
                ub = pre_ub

                var = model.addVar(ub=ub, lb=pre_lb,
                                   obj=0,
                                   vtype=grb.GRB.CONTINUOUS,
                                   name=f'Sign{self.name}_{neuron_idx}')

                a = model.addVar(vtype=grb.GRB.BINARY, name=f'aSign{self.name}_{neuron_idx}')
                integer_vars.append(a)

                layer_constrs.append(
                    model.addConstr(pre_lb * a <= pre_var, name=f'Sign{self.name}_{neuron_idx}_a_0'))
                layer_constrs.append(
                    model.addConstr(pre_ub * (1 - a) >= pre_var, name=f'Sign{self.name}_{neuron_idx}_a_1'))
                layer_constrs.append(
                    model.addConstr(var == 1 - 2*a, name=f'Sign{self.name}_{neuron_idx}_a_2'))

            new_layer_gurobi_vars.append(var)

        new_layer_gurobi_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape).tolist()
        if model_type in ["mip", "lp_integer"]:
            self.integer_vars = integer_vars
        self.solver_vars = new_layer_gurobi_vars
        self.solver_constrs = layer_constrs
        model.update()