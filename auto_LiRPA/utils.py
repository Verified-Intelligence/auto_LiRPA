import logging
import pickle
import time
import torch
import os
import appdirs
from oslo_concurrency import lockutils, processutils
from collections import defaultdict, Sequence, namedtuple

# Special identity matrix. Avoid extra computation of identity matrix multiplication in various places.
eyeC = namedtuple('eyeC', 'shape device')
# Linear bounds with coefficients. Used for forward bound propagation.
LinearBound = namedtuple('LinearBound', ('lw', 'lb', 'uw', 'ub', 'lower', 'upper', 'from_input'), defaults=(None,) * 7)

logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# for debugging
if False:
    file_handler = logging.FileHandler('debug.log')
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

user_data_dir = appdirs.user_data_dir('auto_LiRPA')
if not os.path.exists(user_data_dir):
    os.makedirs(user_data_dir)
lockutils.set_defaults(os.path.join(user_data_dir, '.lock'))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MultiAverageMeter(object):
    """Computes and stores the average and current value for multiple metrics"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum_meter = defaultdict(float)
        self.lasts = defaultdict(float)
        self.counts_meter = defaultdict(int)
    def update(self, key, val, n=1):
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

def scale_gradients(optimizer, gradient_accumulation_steps, grad_clip=None):    
    parameters = []
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            parameters.append(param)
            if param.grad is not None:
                param.grad.data /= gradient_accumulation_steps
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)                

def recursive_map (seq, func):
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(item, func))
        else:
            yield func(item)

# unpack tuple, dict, list into one single list
# TODO: not sure if the order matches graph.inputs()
def unpack_inputs(inputs):
    if isinstance(inputs, dict):
        inputs = list(inputs.values())
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        res = []
        for item in inputs: 
            res += unpack_inputs(item)
        return res
    else:
        return [inputs]

def isnan(x):
    return torch.isnan(x).any()