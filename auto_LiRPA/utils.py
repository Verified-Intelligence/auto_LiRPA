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
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import logging
import time
import torch
import torch.nn as nn
import os
import sys
import appdirs
from collections import defaultdict, namedtuple
from functools import reduce
import operator
import warnings
from typing import Tuple
from .patches import Patches


logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('AUTOLIRPA_DEBUG', 0) else logging.INFO)

warnings.simplefilter("once")

# Special identity matrix. Avoid extra computation of identity matrix multiplication in various places.
eyeC = namedtuple('eyeC', 'shape device')
OneHotC = namedtuple('OneHotC', 'shape device index coeffs')
BatchedCrownC = namedtuple('BatchedCrownC', 'type')

def onehotc_to_dense(one_hot_c: OneHotC, dtype: torch.dtype) -> torch.Tensor:
    shape = one_hot_c.shape  # [spec, batch, C, H, W]
    dim = int(prod(shape[2:]))
    dense = torch.zeros(
        size=(shape[0], shape[1], dim), device=one_hot_c.device, dtype=dtype)
    # one_hot_c.index has size (spec, batch), its values are the index of the one-hot non-zero elements in A.
    # one_hot_c.coeffs is the value of the non-zero element.
    dense = torch.scatter(
        dense, dim=2, index=one_hot_c.index.unsqueeze(-1),
        src=one_hot_c.coeffs.unsqueeze(-1))
    dense = dense.view(shape[0], shape[1], *shape[2:])
    return dense

# Benchmarking mode disable some expensive assertions.
Benchmarking = True

reduction_sum = lambda x: x.sum(dim=tuple(range(1, x.dim())), keepdim=True)
reduction_mean = lambda x: x.mean(dim=tuple(range(1, x.dim())), keepdim=True)
reduction_max = lambda x: x.amax(dim=tuple(range(1, x.dim())), keepdim=True)
reduction_min = lambda x: x.amin(dim=tuple(range(1, x.dim())), keepdim=True)

MIN_HALF_FP = 5e-8  # 2**-24, which is the smallest value that float16 can be represented


def reduction_str2func(reduction_func):
    if type(reduction_func) == str:
        if reduction_func == 'min':
            return reduction_min
        elif reduction_func == 'max':
            return reduction_max
        elif reduction_func == 'sum':
            return reduction_sum
        elif reduction_func == 'mean':
            return reduction_mean
        else:
            raise NotImplementedError(f'Unknown reduction_func {reduction_func}')
    else:
        return reduction_func

def stop_criterion_placeholder(threshold=0):
    return lambda x: RuntimeError("BUG: bound optimization stop criterion not specified.")

def stop_criterion_min(threshold=0):
    return lambda x: (x.min(1, keepdim=True).values > threshold)

def stop_criterion_all(threshold=0):
    # The dimension of x should be (batch, spec). The spec dimension
    # This was used in the incomplete verifier, where the spec dimension can
    # present statements in an OR clause.
    return lambda x: (x > threshold).all(dim=1, keepdim=True)

def stop_criterion_max(threshold=0):
    return lambda x: (x.max(1, keepdim=True).values > threshold)

def stop_criterion_batch(threshold=0):
    # may unexpected broadcast, pay attention to the shape of threshold
    # x shape: batch, number_bounds; threshold shape: batch, number_bounds
    return lambda x: (x > threshold)

def stop_criterion_batch_any(threshold=0):
    """If any spec >= rhs, then this sample can be stopped;
       if all samples can be stopped, stop = True, o.w., False.
    """
    # may unexpected broadcast, pay attention to the shape of threshold
    # x shape: batch, number_bounds; threshold shape: batch, number_bounds
    return lambda x: (x > threshold).any(dim=1, keepdim=True)

def stop_criterion_general(or_spec_size, threshold=0):
    """
    If any spec in a group >= rhs, then this group can be stopped;
    if all groups can be stopped, stop = True, o.w., False.
    Args:
        or_clause_indices: [num_clause]. the indices of the belonging OR clauses for AND clauses.
        num_or: the number of OR clauses.
        threshold: [batch, num_clause]. The threshold for each spec. sum(or_clause_indices) == num_clauses.
    """
    def stop_criterion_per_or(x):
        # get the indices of OR clauses assigned to their corresponding atom clauses, [num_clause]
        num_or = or_spec_size.shape[0]
        or_clause_indices = torch.repeat_interleave(
            torch.arange(num_or, device=or_spec_size.device), or_spec_size
        ).view(1, -1).expand(x.shape)
        # get the result for each spec. [batch, num_clause]
        result_per_spec = (x > threshold) 
        # get the number of verified ANDs for each OR clause. [batch, num_or]
        num_verified_and_per_or = torch.scatter_reduce(result_per_spec[:, :num_or], 1, or_clause_indices, result_per_spec, 'sum', include_self=False)
        # result of any spec in a OR (group of ANDs) is True (sum >= 1) -> result of the OR is True.
        return num_verified_and_per_or >= 1
    # if all OR clauses are True, then return True. [batch, 1]
    return lambda x: stop_criterion_per_or(x).all(dim=1, keepdim=True)

def stop_criterion_batch_topk(threshold=0, k=1314):
    # x shape: batch, number_bounds; threshold shape: batch, number_bounds
    return lambda x: (torch.kthvalue(x, k, dim=-1, keepdim=True).values > threshold).any(dim=1)

def multi_spec_keep_func_all(x):
    return torch.all(x, dim=-1)


user_data_dir = appdirs.user_data_dir('auto_LiRPA')
if not os.path.exists(user_data_dir):
    try:
        os.makedirs(user_data_dir)
    except:
        logger.error('Failed to create directory {}'.format(user_data_dir))


class MultiAverageMeter(object):
    """Computes and stores the average and current value for multiple metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_meter = defaultdict(float)
        self.lasts = defaultdict(float)
        self.counts_meter = defaultdict(int)
        self.batch_size = 1

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def update(self, key, val, n=None):
        if val is None:
            return
        if n is None:
            n = self.batch_size
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.lasts[key] = val
        self.sum_meter[key] += val * n
        self.counts_meter[key] += n

    def last(self, key):
        return self.lasts[key]

    def avg(self, key):
        if self.counts_meter[key] == 0:
            return 0.0
        else:
            return self.sum_meter[key] / self.counts_meter[key]

    def __repr__(self):
        s = ""
        for k in self.sum_meter:
            s += "{}={:.4f} ".format(k, self.avg(k))
        return s.strip()


class MultiTimer(object):
    """Count the time for each part of training."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.timer_starts = defaultdict(float)
        self.timer_total = defaultdict(float)
    def start(self, key):
        if self.timer_starts[key] != 0:
            raise RuntimeError("start() is called more than once")
        self.timer_starts[key] = time.time()
    def stop(self, key):
        if key not in self.timer_starts:
            raise RuntimeError("Key does not exist; please call start() before stop()")
        self.timer_total[key] += time.time() - self.timer_starts[key]
        self.timer_starts[key] = 0
    def total(self, key):
        return self.timer_total[key]
    def __repr__(self):
        s = ""
        for k in self.timer_total:
            s += "{}_time={:.3f} ".format(k, self.timer_total[k])
        return s.strip()


class Flatten(nn.Flatten):
    """Legacy Flatten class.

    It was previously created when nn.Flatten was not supported. Simply use
    nn.Flatten in the future."""
    pass


class Unflatten(nn.Module):
    def __init__(self, wh):
        super().__init__()
        self.wh = wh # width and height of the feature maps
    def forward(self, x):
        return x.view(x.size(0), -1, self.wh, self.wh)


class Max(nn.Module):

    def __init__(self):
        super(Max, self).__init__()

    def forward(self, x, y):
        return torch.max(x, y)


class Min(nn.Module):

    def __init__(self):
        super(Min, self).__init__()

    def forward(self, x, y):
        return torch.min(x, y)


def scale_gradients(optimizer, gradient_accumulation_steps, grad_clip=None):
    parameters = []
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            parameters.append(param)
            if param.grad is not None:
                param.grad.data /= gradient_accumulation_steps
    if grad_clip is not None:
        return torch.nn.utils.clip_grad_norm_(parameters, grad_clip)


# unpack tuple, dict, list into one single list
# TODO: not sure if the order matches graph.inputs()
def unpack_inputs(inputs, device=None):
    if isinstance(inputs, dict):
        inputs = list(inputs.values())
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        res = []
        for item in inputs:
            res += unpack_inputs(item, device=device)
        return res
    else:
        if device is not None:
            inputs = inputs.to(device)
        return [inputs]


def isnan(x):
    if isinstance(x, Patches):
        return False
    return torch.isnan(x).any()


def prod(x):
    return reduce(operator.mul, x, 1)


def batched_index_select(input, dim, index):
    # Assuming the input has a batch dimension.
    # index has dimensin [spec, batch].
    if input.ndim == 4:
        # Alphas for fully connected layers, shape [2, spec, batch, neurons]
        index = index.unsqueeze(-1).unsqueeze(0).expand(input.size(0), -1, -1, input.size(3))
    elif input.ndim == 6:
        # Alphas for fully connected layers, shape [2, spec, batch, c, h, w].
        index = index.view(1, index.size(0), index.size(1), *([1] * (input.ndim - 3))).expand(input.size(0), -1, -1, *input.shape[3:])
    elif input.ndim == 3:
        # Weights.
        input = input.expand(index.size(0), -1, -1)
        index = index.unsqueeze(-1).expand(-1, -1, input.size(2))
    elif input.ndim == 2:
        # Bias.
        input = input.expand(index.size(0), -1)
    else:
        raise ValueError
    return torch.gather(input, dim, index)


def get_spec_matrix(X, y, num_classes):
    with torch.no_grad():
        c = (torch.eye(num_classes).type_as(X)[y].unsqueeze(1)
            - torch.eye(num_classes).type_as(X).unsqueeze(0))
        I = (~(y.unsqueeze(1) == torch.arange(num_classes).type_as(y).unsqueeze(0)))
        c = (c[I].view(X.size(0), num_classes - 1, num_classes))
    return c


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, a list with tensors in shape (N, D).

    Code borrowed from:
        https://github.com/pytorch/pytorch/issues/35674
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='trunc')

    return list(reversed(coord))


class AutoBatchSize:
    def __init__(self, init_batch_size, device, vram_ratio=0.9, enable=True):
        self.batch_size = init_batch_size
        self.max_actual_batch_size = 0
        self.device = device
        self.vram_ratio = vram_ratio
        self.enable = enable

    def record_actual_batch_size(self, actual_batch_size):
        """Record the actual batch size used.

        It may be smaller than self.batch_size, especially for the early batches.
        """
        self.max_actual_batch_size = max(self.max_actual_batch_size, actual_batch_size)

    def update(self):
        """Check if the batch size can be enlarged."""
        if not self.enable:
            return None
        # Only try to update the batch size if the current batch size has
        # been actually used, as indicated by `max_actual_batch_size`
        if self.device == 'cpu' or self.max_actual_batch_size < self.batch_size:
            return None
        total_vram = torch.cuda.get_device_properties(self.device).total_memory
        current_vram = torch.cuda.memory_reserved(self.device)
        if current_vram * 2 >= total_vram * self.vram_ratio:
            return None
        new_batch_size = self.batch_size * 2
        self.batch_size = new_batch_size
        logger.debug('Automatically updated batch size to %d', new_batch_size)
        return {
            'current_vram': current_vram,
            'total_vram': total_vram,
        }


def sync_params(model_ori: torch.nn.Module,
                model: 'BoundedModule',
                loss_fusion: bool = False):
    """Sync the parameters from a BoundedModule to the original model."""
    state_dict_loss = model.state_dict()
    state_dict = model_ori.state_dict()
    for name in state_dict_loss:
        v = state_dict_loss[name]
        if name.endswith('.param'):
            name = name[:-6]
        elif name.endswith('.buffer'):
            name = name[:-7]
        else:
            raise NameError(name)
        name_ori = model[name].ori_name
        if loss_fusion:
            assert name_ori.startswith('model.')
            name_ori = name_ori[6:]
        assert name_ori in state_dict
        state_dict[name_ori] = v
    model_ori.load_state_dict(state_dict)
    return state_dict


def reduce_broadcast_dims(A, target_shape, left_extra_dims=1):
    """
    When backward propagating tensors that are automatically broadcasted,
    we need to reduce the broadcasted dimensions to match the input shape.
    This can be useful for backward bound propagation and backward gradient
    computation.

    Args:
        A: The input tensor.
        target_shape: The target shape to reduce to.
        left_extra_dims: The number of dimensions that A should have but the target
            shape doesn't have. These dimensions are usually added to the left of the
            target shape and don't need to be reduced (e.g. spec).

    Example:
        x1 has shape [a1, a2, a3, a4], x2 has shape [a2, 1, a4], y = x1 * x2.
        Two types of broadcasting here:
            1. Adding additional dimensions to x2 to match the dimension of x1.
            2. Broadcasting along existing dimensions length 1.
        In backward computation from y to x2, we need to reduce (sum) the A matrix
        to match the shape of x2. The first dimension of A is usually for spec, so
        the shape usually aligns from the second dimension.
    """
    # Step 1: Dimension doesn't exist in target shape but exists in A.
    # cnt_sum is the number of dimensions that are broadcast.
    # (The additional dimensions in A that are not in target shape).
    cnt_sum = (A.ndim - left_extra_dims) - len(target_shape)
    # The broadcast dimensions must be the first dimensions in A
    # (except the extra dimensions and batch dimension).
    dims = list(range(left_extra_dims + 1, cnt_sum + left_extra_dims + 1))
    if dims:
        A = torch.sum(A, dim=dims, keepdim=False)
    # Step 2: Dimension exists in target shape, broadcast from 1.
    # FIXME (05/11/2022): the following condition is not always correct.
    # We should not rely on checking dimension is "1" or not.
    dims = [i + left_extra_dims for i in range(left_extra_dims, len(target_shape))
            if target_shape[i] == 1 and A.shape[i + left_extra_dims] != 1]
    if dims:
        A = torch.sum(A, dim=dims, keepdim=True)
    # Check the final shape - it should be compatible.
    assert A.shape[2:] == target_shape[1:]  # skip the spec and batch dimension.
    return A


@torch.jit.script
def matmul_maybe_batched(a: torch.Tensor, b: torch.Tensor, both_batched: bool):
    # Basically just matmul, but we need to handle the batch dimension.
    if both_batched:
        return torch.einsum("b...ij,b...jk->b...ik", a, b)
    else:
        return a.matmul(b)

def transfer(tensor, device=None, dtype=None, non_blocking=False):
    """Transfer a tensor to a specific device or dtype."""
    if device:
        tensor = tensor.to(device, non_blocking=non_blocking)
    if dtype:
        tensor = tensor.to(dtype)

    return tensor


def clone_sub_A_dict(A_dict, out_in_keys: Tuple):
    """
    Deep copy the A_dict structure for specific out_in_keys.
    Args:
        A_dict: The A_dict to be copied.
        out_in_keys: The (out_key, in_key) pairs to be copied.
    Returns:
        A new A_dict with all tensors cloned.
    """
    # Structure: A_dict[out_key][in_key][key]
    # key in [lA, uA, lbias, ubias, unstable_idx]
    # lA, uA are tensors or Patches
    # (there're also types like eyeC, OneHotC, not supported here)
    # lbias, ubias are tensors
    # unstable_idx is tensor or tuple of tensors

    out_key, in_key = out_in_keys
    src_subdict = A_dict[out_key][in_key]
    cloned_subdict = {}

    for key, val in src_subdict.items():
        if val is None:
            cloned_subdict[key] = None
            continue

        if isinstance(val, (torch.Tensor, Patches)):
            cloned_subdict[key] = val.detach().clone()
        elif isinstance(val, tuple):
            cloned_subdict[key] = tuple(v.detach().clone() for v in val)
        else:
            raise NotImplementedError(f'Unsupported A type {type(val)} for copying.')
    return cloned_subdict


def clone_full_A_dict(A_dict):
    """
    Deep copy the A_dict structure.
    Args:
        A_dict: The A_dict to be copied.
    Returns:
        A new A_dict with all tensors cloned.
    """
    new_A_dict = {}
    for out_key, in_dict in A_dict.items():
        new_A_dict[out_key] = {}
        for in_key in in_dict:
            new_A_dict[out_key][in_key] = clone_sub_A_dict(A_dict, (out_key, in_key))
    return new_A_dict