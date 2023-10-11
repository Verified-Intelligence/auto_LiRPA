""" Resize operator """
import itertools

import torch

from .base import *
import numpy as np
from .solver_utils import grb
from ..patches import unify_shape, create_valid_mask, is_shape_used
from .gradient_modules import Conv2dGrad


class BoundResize(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        # only support nearest mode for now
        assert attr["mode"] == "nearest"
        self.mode = attr["mode"]
        self.scale_factor = None

    def forward(self, x, size=None, scale_factor=None):
        # currently, forwarding size is not supported.
        assert isinstance(size, torch.Tensor) and len(size.tolist()) == 0
        # currently, only support enlarge tensor size by an integer factor.
        assert len(scale_factor.tolist()) == 4 and np.array([tmp.is_integer() and tmp > 0 for tmp in scale_factor.tolist()]).all()
        assert (scale_factor[0:2].to(torch.long) == 1).all(), 'only support resize on the H and W dim'
        self.scale_factor = tuple([int(tmp) for tmp in scale_factor][2:])
        if x.ndim == 4:
            final = F.interpolate(
                x, None, self.scale_factor, mode=self.mode)
        else:
            raise NotImplementedError(
                "Interpolation in 3D or interpolation with parameter size has not been implmented.")
        return final

    def interval_propagate(self, *v):
        l, u = zip(*v)
        return Interval.make_interval(self.forward(*l), self.forward(*u), v[0])

    def bound_forward(self, dim_in, *inp):
        x = inp[0]
        lw, lb, uw, ub = x.lw, x.lb, x.uw, x.ub
        new_lw, new_lb, new_uw, new_ub = \
            torch.nn.functional.upsample(lw, scale_factor=([1] * (lw.ndim - 4)) + list(self.scale_factor), mode=self.mode), \
            torch.nn.functional.upsample(lb, scale_factor=([1] * (lb.ndim - 4)) + list(self.scale_factor), mode=self.mode), \
            torch.nn.functional.upsample(uw, scale_factor=([1] * (uw.ndim - 4)) + list(self.scale_factor), mode=self.mode), \
            torch.nn.functional.upsample(ub, scale_factor=([1] * (ub.ndim - 4)) + list(self.scale_factor), mode=self.mode)
        return LinearBound(
            lw = new_lw,
            lb = new_lb,
            uw = new_uw,
            ub = new_ub)

    def bound_backward(self, last_lA, last_uA, *x, **kwargs):

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            assert type(last_A) is Patches or last_A.ndim == 5
            # in case the kernel size cannot be divided by scale_factor, we round up the shape
            split_shape = tuple((torch.tensor(
                last_A.shape)[-2:] / torch.tensor(self.scale_factor)).ceil().to(torch.long).tolist())
            new_shape = last_A.shape[:-2] + split_shape
            if not type(last_A) is Patches:
                # classical mode is simple to handle by
                # sum the grid elements by using avg_pool2d with divisor_override=1
                return torch.nn.functional.avg_pool2d(
                    last_A.reshape(-1, *last_A.shape[-2:]), kernel_size=self.scale_factor, stride=self.scale_factor,
                    divisor_override=1).reshape(new_shape)
            else:
                # for patches mode
                assert type(last_A) is Patches
                assert self.scale_factor[0] == self.scale_factor[1]
                if self.scale_factor[0] == 1:
                    # identity upsampling
                    return last_A
                if isinstance(last_A.padding, int) and last_A.padding % self.scale_factor[0] == 0 and last_A.stride % self.scale_factor[0] == 0 and last_A.inserted_zeros == 0:
                    # an easy case where patch sliding windows coincides with the nearest sampling scaling windows
                    # in this case, we divide each patch to size of scale_factor sub-matrices,
                    # and sum up each sub-matrices respectively
                    # print(last_A.shape)
                    padding = last_A.shape[-1] % self.scale_factor[-1]
                    new_patches = torch.nn.functional.pad(last_A.patches, (0, padding, 0, padding))
                    new_patches = torch.nn.functional.avg_pool2d(
                        new_patches.reshape(-1, *new_patches.shape[-2:]), kernel_size=self.scale_factor,
                        stride=self.scale_factor, divisor_override=1).reshape(new_shape)
                    return last_A.create_similar(patches=new_patches,
                                                 stride=last_A.stride//self.scale_factor[0],
                                                 padding=last_A.padding//self.scale_factor[0],
                                                 )
                else:
                    """
                        The following part is created and mainly maintained by Linyi
                        Time complexity = O(A.numel * scale_factor + outH * kerH + outW * kerW + A.numel * kerH * kerW)
                        With Python loop complexity = O(outH + outW + kerH * kerW * scale_factor^2)
                    """
                    # preparation: unify shape
                    if last_A.padding:
                        padding = unify_shape(last_A.padding)
                    else:
                        padding = (0,0,0,0)
                    # padding = (left, right, top, bottom)
                    if last_A.output_padding:
                        output_padding = unify_shape(last_A.output_padding)
                    else:
                        output_padding = (0,0,0,0)
                    # output_padding = (left, right, top, bottom)

                    """
                        Step 0: filter out valid entries that maps to real cells of input
                        Like with inserted zeros = 2, [x 0 0 x 0 0 x]. Only "x" cells are kept
                        Borrowed from one_d generation from Conv patches
                    """
                    one_d_unfolded_r = create_valid_mask(self.output_shape,
                                                         last_A.patches.device,
                                                         last_A.patches.dtype,
                                                         last_A.patches.shape[-2:],
                                                         last_A.stride,
                                                         last_A.inserted_zeros,
                                                         last_A.padding,
                                                         last_A.output_padding,
                                                         last_A.unstable_idx)
                    patches = last_A.patches * one_d_unfolded_r

                    """
                        Step 1: compute the coordinate mapping from patch coordinates to input coordinates
                        Time complexity: O(outH + outW)
                        note: last_A shape is [outC, batch, outH, outW, inC, kerH, kerW]
                        We create H_idx_map and W_idx_map of shape [outH] and [outW] respectively,
                        recording the start idx of row/column for patches at position [.,.,.,.,.,i,j]
                        in H_idx_map[i] and W_idx_map[j]
                    """
                    ker_size_h, ker_size_w = last_A.shape[-2], last_A.shape[-1]
                    if last_A.unstable_idx is None:
                        # we can get the real output H and W from shape[2] and shape [3]
                        out_h, out_w = last_A.shape[2], last_A.shape[3]
                    else:
                        # it seems to be stored in output_shape
                        out_h, out_w = last_A.output_shape[-2], last_A.output_shape[-1]
                    h_idx_map = torch.arange(0, out_h) * last_A.stride - padding[-2] + output_padding[-2] * last_A.stride
                    h_idx_map = h_idx_map.to(last_A.device)
                    w_idx_map = torch.arange(0, out_w) * last_A.stride - padding[-4] + output_padding[-4] * last_A.stride
                    w_idx_map = w_idx_map.to(last_A.device)

                    r"""
                        Step 2: compute the compressed patches
                        Time complexity: O(outH * kerH + outW * kerW + A.numel * kerH * kerW)
                        Upsampling needs to sum up A cells in scale_factor * scale_factor sub-blocks
                        Example: when scale factor is 2
                        [ a b c d
                          e f g h    ---\    [ a+b+e+f c+d+g+h
                          i j k l    ---/      i+j+m+n k+l+o+p]
                          m n o p]
                        In patches mode, we need to sum up cells in each patch accordingly.
                        The summing mechanism could change at different locations.
                        For each spatial dimension, we create a binary sum_mask tensor [outH, ker_size_h, new_ker_size_h]
                            to select the cells to sum up
                        Example:
                        For [a b c d] -> [a+b c+d], with 3x3 patch covering [0..2] and [2..4].
                        The first patch needs to sum to [a+b c]; the second patch needs to sum to [b c+d]
                        So we have sum_mask
                        [ for patch 1: [[1, 1, 0],    (first entry sums up index 0 and 1)
                                        [0, 0, 1]]^T, (second entry sums up index 2)
                          for patch 2: [[1, 0, 0],    (first entry sums up index 0)
                                        [0, 1, 1]]^T  (second entry sums up index 1 and 2)
                        ]
                        With the mask, we can now compute the new patches with einsum:
                            [outC, batch, outH, outW, inC, kerH, kerW] * [outH, kerH, new_kerH] -> [outC, batch, outH, outW, inC, new_kerH, kerW]
                    """
                    tot_scale_fac = ((last_A.inserted_zeros + 1) * self.scale_factor[0], (last_A.inserted_zeros + 1) * self.scale_factor[1])
                    new_ker_size_h, new_ker_size_w = \
                        (tot_scale_fac[0] + ker_size_h - 2) // tot_scale_fac[0] + 1, \
                        (tot_scale_fac[1] + ker_size_w - 2) // tot_scale_fac[1] + 1

                    min_h_idx, max_h_idx = h_idx_map[0], h_idx_map[-1] + ker_size_h
                    shrank_h_idx = (torch.arange(min_h_idx, max_h_idx) + last_A.inserted_zeros).div(tot_scale_fac[0], rounding_mode='floor')
                    if last_A.unstable_idx is None:
                        # with nonsparse index, create full-sized sum musk for rows
                        ker_h_indexer = torch.arange(0, ker_size_h).to(last_A.device)
                        sum_mask_h = torch.zeros(last_A.shape[2], ker_size_h, new_ker_size_h).to(last_A.device)
                        for i in range(last_A.shape[2]):
                            sum_mask_h[i, ker_h_indexer, \
                                shrank_h_idx[h_idx_map[i] - min_h_idx: h_idx_map[i] - min_h_idx + ker_size_h] - shrank_h_idx[h_idx_map[i] - min_h_idx]] = 1
                            # set zero to those in padding area
                            padding_place_mask = (ker_h_indexer + h_idx_map[i] < 0)
                            sum_mask_h[i, padding_place_mask] = 0
                    else:
                        # with sparse index, create sparse sum musk
                        sum_mask_h = torch.zeros(last_A.shape[0], ker_size_h, new_ker_size_h).to(last_A.device)

                        row_nos = last_A.unstable_idx[1]
                        unstable_loc_indexer = torch.arange(0, row_nos.shape[0]).to(last_A.device)

                        for k in range(ker_size_h):
                            place_in_new_ker = shrank_h_idx[h_idx_map[row_nos] - min_h_idx + k] - shrank_h_idx[h_idx_map[row_nos] - min_h_idx]
                            sum_mask_h[unstable_loc_indexer, k, place_in_new_ker] = 1
                            # set zero to those in padding area
                            padding_place_mask = (h_idx_map[row_nos] + k < 0)
                            sum_mask_h[padding_place_mask, k] = 0

                    min_w_idx, max_w_idx = w_idx_map[0], w_idx_map[-1] + ker_size_w
                    shrank_w_idx = (torch.arange(min_w_idx, max_w_idx) + last_A.inserted_zeros).div(tot_scale_fac[1], rounding_mode='floor')
                    if last_A.unstable_idx is None:
                        # with nonsparse index, create full-sized sum musk for columns
                        ker_w_indexer = torch.arange(0, ker_size_w).to(last_A.device)
                        sum_mask_w = torch.zeros(last_A.shape[3], ker_size_w, new_ker_size_w).to(last_A.device)
                        for i in range(last_A.shape[3]):
                            sum_mask_w[i, ker_w_indexer, \
                                shrank_w_idx[w_idx_map[i] - min_w_idx: w_idx_map[i] - min_w_idx + ker_size_w] - shrank_w_idx[w_idx_map[i] - min_w_idx]] = 1
                            # set zero to those in padding area
                            padding_place_mask = (ker_w_indexer + w_idx_map[i] < 0)
                            sum_mask_w[i, padding_place_mask] = 0
                    else:
                        # with sparse index, create sparse sum musk
                        sum_mask_w = torch.zeros(last_A.shape[0], ker_size_w, new_ker_size_w).to(last_A.device)

                        col_nos = last_A.unstable_idx[2]
                        unstable_loc_indexer = torch.arange(0, col_nos.shape[0]).to(last_A.device)

                        for k in range(ker_size_w):
                            place_in_new_ker = shrank_w_idx[w_idx_map[col_nos] - min_w_idx + k] - shrank_w_idx[w_idx_map[col_nos] - min_w_idx]
                            sum_mask_w[unstable_loc_indexer, k, place_in_new_ker] = 1
                            # set zero to those in padding area
                            padding_place_mask = (w_idx_map[col_nos] + k < 0)
                            sum_mask_w[padding_place_mask, k] = 0

                    if last_A.unstable_idx is None:
                        # nonsparse aggregation
                        new_patches = torch.einsum("ObhwIij,hix,wjy->ObhwIxy", patches, sum_mask_h, sum_mask_w)
                    else:
                        # sparse aggregation
                        new_patches = torch.einsum("NbIij,Nix,Njy->NbIxy", patches, sum_mask_h, sum_mask_w)

                    """
                        Step 3: broadcasting the new_patches by repeating elements,
                            since later we would need to apply insert_zeros
                        For example, scale_factor = 3, repeat patch [a,b] to [a,a,a,b,b,b]
                        Time complexity: O(A.numel * scale_factor)
                    """
                    ext_new_ker_size_h, ext_new_ker_size_w = \
                        new_ker_size_h * tot_scale_fac[0], new_ker_size_w * tot_scale_fac[1]
                    ext_new_patches = torch.zeros(list(new_patches.shape[:-2]) +
                                                  [ext_new_ker_size_h, ext_new_ker_size_w], device=new_patches.device)
                    for i in range(ext_new_ker_size_h):
                        for j in range(ext_new_ker_size_w):
                            ext_new_patches[..., i, j] = new_patches[..., i // tot_scale_fac[0], j // tot_scale_fac[1]]

                    """
                        Step 4: compute new padding, stride, shape, insert_zeros, and output_padding
                    """
                    # stride should be the same after upsampling, stride is an integer
                    # new_stride = last_A.stride
                    # padding can change much, the beginning should extend by (scale - 1) entries,
                    # the ending should extend by (ext_new_ker_size - ker_size) entries
                    # padding = (left, right, top, bottom)
                    new_padding = (padding[0] + (self.scale_factor[1] - 1) * (last_A.inserted_zeros + 1),
                                   padding[1] + ext_new_ker_size_w - ker_size_w,
                                   padding[2] + (self.scale_factor[0] - 1) * (last_A.inserted_zeros + 1),
                                   padding[3] + ext_new_ker_size_h - ker_size_h)
                    if new_padding[0] == new_padding[1] and new_padding[1] == new_padding[2] and new_padding[2] == new_padding[3]:
                        # simplify to an int
                        new_padding = new_padding[0]
                    # only support uniform scaling on H and W now, i.e., self.scale_factor[0] == self.scale_factor[1]
                    inserted_zeros = tot_scale_fac[0] - 1
                    # output padding seems not to change
                    # new_output_padding = last_A.output_padding

                    """
                        Package and create
                    """
                    # sparse tensor doesn't support einsum which is necessary for subsequent computes, so deprecated
                    # if inserted_zeros >= 3:
                    #     # mask unused cells
                    #     input_shape = list(self.output_shape)
                    #     input_shape[-2], input_shape[-1] = input_shape[-2] // self.scale_factor[-2], \
                    #         input_shape[-1] // self.scale_factor[-1]
                    #     one_unfolded = create_valid_mask(input_shape, ext_new_patches.device,
                    #                                       ext_new_patches.dtype, ext_new_patches.shape[-2:],
                    #                                       last_A.stride, inserted_zeros, new_padding,
                    #                                       last_A.output_padding,
                    #                                       last_A.unstable_idx if last_A.unstable_idx else None)
                    #     ext_new_patches = (ext_new_patches * one_unfolded).to_sparse()

                    # print the shape change after upsampling, if needed
                    # print(f'After upsampling, '
                    #       f'{last_A.patches.shape} (pad={padding}, iz={last_A.inserted_zeros}, s={last_A.stride}) -> '
                    #       f'{ext_new_patches.shape} (pad={new_padding}, iz={inserted_zeros}, s={last_A.stride})')
                    ret_patches_A = last_A.create_similar(patches=ext_new_patches,
                                                          padding=new_padding,
                                                          inserted_zeros=inserted_zeros)
                    if self.input_shape[-2] < ret_patches_A.shape[-2] and self.input_shape[-1] < ret_patches_A.shape[-2] \
                            and not is_shape_used(ret_patches_A.output_padding):
                        # using matrix mode could be more memory efficient
                        ret_matrix_A = ret_patches_A.to_matrix(self.input_shape)
                        # print(f'After upsampling, to_matrix: {ret_matrix_A.shape}')
                        ret_matrix_A = ret_matrix_A.transpose(0, 1)
                        return ret_matrix_A
                    else:
                        return ret_patches_A

        last_lA = _bound_oneside(last_lA)
        last_uA = _bound_oneside(last_uA)
        return [(last_lA, last_uA), (None, None), (None, None)], 0, 0
