""" Bound classes for gradient operators """
import torch
import torch.nn.functional as F
from auto_LiRPA.patches import Patches, inplace_unfold
from .base import Bound, Interval
from .activation_base import BoundActivation
from .gradient_modules import relu_grad


# FIXME reuse the function from auto_LiRPA.patches
def _maybe_unfold(d_tensor, last_A):
    if d_tensor is None:
        return None

    #[batch, out_dim, in_c, in_H, in_W]
    d_shape = d_tensor.size()

    # Reshape to 4-D tensor to unfold.
    #[batch, out_dim*in_c, in_H, in_W]
    d_tensor = d_tensor.view(d_shape[0], -1, *d_shape[-2:])
    # unfold the slope matrix as patches.
    # Patch shape is [batch, out_h, out_w, out_dim*in_c, H, W).
    d_unfolded = inplace_unfold(
        d_tensor, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride,
        padding=last_A.padding)
    # Reshape to [batch, out_H, out_W, out_dim, in_C, H, W]
    d_unfolded_r = d_unfolded.view(
        *d_unfolded.shape[:3], d_shape[1], *d_unfolded.shape[-2:])
    if last_A.unstable_idx is not None:
        if len(last_A.unstable_idx) == 4:
            # [batch, out_H, out_W, out_dim, in_C, H, W]
            # to [out_H, out_W, batch, out_dim, in_C, H, W]
            d_unfolded_r = d_unfolded_r.permute(1, 2, 0, 3, 4, 5, 6)
            d_unfolded_r = d_unfolded_r[
                last_A.unstable_idx[2], last_A.unstable_idx[3]]
        else:
            raise NotImplementedError
    # For sparse patches, the shape after unfold is
    # (unstable_size, batch_size, in_c, H, W).
    # For regular patches, the shape after unfold is
    # (spec, batch, out_h, out_w, in_c, H, W).
    return d_unfolded_r


class BoundReluGrad(BoundActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.requires_input_bounds = [3]
        self.recurjac = options.get('recurjac', False)

    @staticmethod
    def relu_grad(preact):
        return (preact > 0).float()

    def forward(self, g, g_relu, g_relu_rev, preact):
        if g.ndim == preact.ndim + 1:
            preact = preact.unsqueeze(1)
        return g * relu_grad(preact)

    def interval_propagate(self, *v):
        g_lower, g_upper = v[0]
        preact_lower, preact_upper = v[3]
        relu_grad_lower = relu_grad(preact_lower)
        relu_grad_upper = relu_grad(preact_upper)
        if g_lower.ndim == relu_grad_lower.ndim + 1:
            relu_grad_lower = relu_grad_lower.unsqueeze(1)
            relu_grad_upper = relu_grad_upper.unsqueeze(1)
        lower = torch.min(g_lower * relu_grad_lower, g_lower * relu_grad_upper)
        upper = torch.max(g_upper * relu_grad_lower, g_upper * relu_grad_upper)
        return lower, upper

    def bound_backward(self, last_lA, last_uA, g, g_relu, g_relu_rev, preact,
                       **kwargs):
        mask_active = (preact.lower > 0).float()
        mask_inactive = (preact.upper < 0).float()
        mask_unstable = 1 - mask_active - mask_inactive

        if self.recurjac and self.inputs[0].perturbed:
            upper_grad = preact.upper >= 0
            lower_interval = self.inputs[0].lower * upper_grad
            upper_interval = self.inputs[0].upper * upper_grad
        else:
            lower_interval = upper_interval = None

        def _bound_oneside(last_A, pos_interval=None, neg_interval=None):
            if last_A is None:
                return None, None, None, 0

            if isinstance(last_A, torch.Tensor):
                if self.recurjac and self.inputs[0].perturbed:
                    mask_unstable_grad = (
                        (self.inputs[0].lower < 0) * (self.inputs[0].upper > 0))
                    last_A_unstable = last_A * mask_unstable_grad
                    bias = (
                        last_A_unstable.clamp(min=0) * pos_interval
                        + last_A_unstable.clamp(max=0) * neg_interval)
                    bias = bias.reshape(
                        bias.shape[0], bias.shape[1], -1).sum(dim=-1)
                    last_A = last_A * torch.logical_not(mask_unstable_grad)
                else:
                    bias = 0
                A = last_A * mask_active
                A_pos = last_A.clamp(min=0) * mask_unstable
                A_neg = last_A.clamp(max=0) * mask_unstable
                return A, A_pos, A_neg, bias
            elif isinstance(last_A, Patches):
                last_A_patches = last_A.patches

                if self.recurjac and self.inputs[0].perturbed:
                    mask_unstable_grad = (
                        (self.inputs[0].lower < 0) * (self.inputs[0].upper > 0))
                    mask_unstable_grad_unfold = _maybe_unfold(
                        mask_unstable_grad, last_A)
                    last_A_unstable = (
                        last_A.to_matrix(mask_unstable_grad.shape)
                        * mask_unstable_grad)
                    bias = (
                        last_A_unstable.clamp(min=0) * pos_interval
                        + last_A_unstable.clamp(max=0) * neg_interval)
                    # FIXME Clean up patches. This implementation does not seem
                    # to support general shapes.
                    assert bias.ndim == 5
                    bias = bias.sum(dim=[-1, -2, -3]).view(-1, 1)
                    last_A_patches = (
                        last_A_patches
                        * torch.logical_not(mask_unstable_grad_unfold))
                else:
                    bias = 0

                # need to unfold mask_active and mask_unstable
                # [batch, 1, in_c, in_H, in_W]
                mask_active_unfold = _maybe_unfold(mask_active, last_A)
                mask_unstable_unfold = _maybe_unfold(mask_unstable, last_A)
                # [spec, batch, 1, in_c, in_H, in_W]

                mask_active_unfold = mask_active_unfold.expand(last_A.shape)
                mask_unstable_unfold = mask_unstable_unfold.expand(last_A.shape)

                A = Patches(
                    last_A_patches * mask_active_unfold,
                    last_A.stride, last_A.padding, last_A.shape,
                    last_A.identity, last_A.unstable_idx, last_A.output_shape)

                A_pos_patches = last_A_patches.clamp(min=0) * mask_unstable_unfold
                A_neg_patches = last_A_patches.clamp(max=0) * mask_unstable_unfold

                A_pos = Patches(
                    A_pos_patches, last_A.stride, last_A.padding, last_A.shape,
                    last_A.identity, last_A.unstable_idx, last_A.output_shape)
                A_neg = Patches(
                    A_neg_patches, last_A.stride, last_A.padding, last_A.shape,
                    last_A.identity, last_A.unstable_idx, last_A.output_shape)

                return A, A_pos, A_neg, bias

        lA, lA_pos, lA_neg, lbias = _bound_oneside(
            last_lA, pos_interval=lower_interval, neg_interval=upper_interval)
        uA, uA_pos, uA_neg, ubias = _bound_oneside(
            last_uA, pos_interval=upper_interval, neg_interval=lower_interval)

        return (
            [(lA, uA), (lA_neg, uA_pos), (lA_pos, uA_neg), (None, None)],
            lbias, ubias)


class BoundConv2dGrad(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.stride = attr['stride']
        self.padding = attr['padding']
        self.dilation = attr['dilation']
        self.groups = attr['groups']
        self.output_padding = [
            attr.get('output_padding0', 0),
            attr.get('output_padding1', 0)
        ]
        self.has_bias = len(inputs) == 3
        self.mode = options.get('conv_mode', 'matrix')
        self.patches_start = True

    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.has_bias
        return F.conv_transpose2d(
            x[0], x[1], None,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, output_padding=self.output_padding)

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):
        assert not self.is_input_perturbed(1)

        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].lower

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0

            if isinstance(last_A, torch.Tensor):
                shape = last_A.size()
                next_A = F.conv2d(
                    last_A.reshape(shape[0] * shape[1], *shape[2:]),
                    weight, None, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups)
                next_A = next_A.view(
                    shape[0], shape[1], *next_A.shape[1:])
                if self.has_bias:
                    sum_bias = (last_A.sum((3, 4)) * x[2].lower).sum(2)
                else:
                    sum_bias = 0
                return next_A, sum_bias
            elif isinstance(last_A, Patches):
                # Here we build and propagate a Patch object with
                # (patches, stride, padding)
                assert self.stride == 1, 'The patches mode only supports stride = 1'
                if last_A.identity == 1:
                    # create a identity patch
                    # [out_dim, batch, out_c, out_h, out_w, in_dim, in_c, in_h, in_w]
                    patch_shape = last_A.shape
                    if last_A.unstable_idx is not None:
                        # FIXME Somehow the usage of unstable_idx seems to have
                        # been changed, and the previous code is no longer working.
                        raise NotImplementedError(
                            'Sparse patches for '
                            'BoundConv2dGrad is not supported yet.')
                        output_shape = last_A.output_shape
                        patches = torch.eye(
                            patch_shape[0]).to(weight)
                        patches = patches.view([
                            patch_shape[0], 1, 1, 1, 1, patch_shape[0], 1, 1])
                        # [out_dim, bsz, out_c, out_h, out_w, out_dim, in_c, in_h, in_w]
                        patches = patches.expand([
                            patch_shape[0], patch_shape[1], patch_shape[2],
                            output_shape[2], output_shape[3],
                            patch_shape[0], 1, 1])
                        patches = patches.transpose(0, 1)
                        patches = patches[
                            :,torch.tensor(list(range(patch_shape[0]))),
                            last_A.unstable_idx[0], last_A.unstable_idx[1],
                            last_A.unstable_idx[2]]
                        patches = patches.transpose(0, 1)
                    else:
                        # out_dim * out_c
                        patches = torch.eye(patch_shape[0]).to(weight)
                        patches = patches.view([
                            patch_shape[0], 1, 1, 1, patch_shape[0], 1, 1])
                        patches = patches.expand(patch_shape)
                else:
                    patches = last_A.patches

                if self.has_bias:
                    # bias is x[2] (lower and upper are the same), and has
                    # shape (c,).
                    # Patches either has
                    # [out_dim, batch, out_c, out_h, out_w, out_dim, c, h, w]
                    # or [unstable_size, batch, out_dim, c, h, w].
                    # sum_bias has shape (out_dim, batch, out_c, out_h, out_w)
                    # or (unstable_size, batch).
                    sum_bias = torch.einsum(
                        'sb...ochw,c->sb...', patches, x[2].lower)
                else:
                    sum_bias = 0

                flattened_patches = patches.reshape(
                    -1, patches.size(-3), patches.size(-2), patches.size(-1))
                # Pad to the full size
                pieces = F.conv2d(
                    flattened_patches, weight, stride=self.stride,
                    padding=weight.shape[2]-1)
                # New patch size:
                # (out_c, batch, out_h, out_w, c, h, w)
                # or (unstable_size, batch, c, h, w).
                pieces = pieces.view(
                    *patches.shape[:-3], pieces.size(-3), pieces.size(-2),
                    pieces.size(-1))

                # (left, right, top, bottom)
                padding = last_A.padding if last_A is not None else (0, 0, 0, 0)
                stride = last_A.stride if last_A is not None else 1

                if isinstance(padding, int):
                    padding = padding + weight.shape[2] - 1
                else:
                    padding = tuple(p + weight.shape[2] - 1 for p in padding)

                return Patches(
                    pieces, stride, padding, pieces.shape,
                    unstable_idx=last_A.unstable_idx,
                    output_shape=last_A.output_shape), sum_bias
            else:
                raise NotImplementedError()

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def interval_propagate(self, *v, C=None):
        assert not self.is_input_perturbed(1)

        norm = Interval.get_perturbation(v[0])[0]
        h_L, h_U = v[0]

        weight = v[1][0]
        bias = v[2][0] if self.has_bias else None

        if norm == torch.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv_transpose2d(
                diff, weight_abs, None, stride=self.stride,
                padding=self.padding, dilation=self.dilation,
                groups=self.groups, output_padding=self.output_padding)
        else:
            raise NotImplementedError
        center = F.conv_transpose2d(
            mid, weight, bias, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
            output_padding=self.output_padding)
        upper = center + deviation
        lower = center - deviation
        return lower, upper
