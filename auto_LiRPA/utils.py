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
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('AUTOLIRPA_DEBUG', 0) else logging.INFO)

warnings.simplefilter("once")

# Special identity matrix. Avoid extra computation of identity matrix multiplication in various places.
eyeC = namedtuple('eyeC', 'shape device')
OneHotC = namedtuple('OneHotC', 'shape device index coeffs')

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

reduction_sum = lambda x: x.sum(1, keepdim=True)
reduction_mean = lambda x: x.mean(1, keepdim=True)
reduction_max = lambda x: x.max(1, keepdim=True).values
reduction_min = lambda x: x.min(1, keepdim=True).values

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


def fill_template(out, template):
    if template is None:
        return out.popleft()
    elif isinstance(template, (list, tuple)):
        res = []
        for t in template:
            res.append(fill_template(t))
        return tuple(res) if isinstance(template, tuple) else res
    elif isinstance(template, dict):
        res = {}
        for key in template:
            res[key] = fill_template(template[key])
        return res
    else:
        raise NotImplementedError
