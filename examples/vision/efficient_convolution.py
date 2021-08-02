"""
Demonstration of efficient convolutional network implementation in auto_LiRPA.

auto_LiRPA library supports an efficient algorithm for computing bounds for
convolutional networks. The "patches" mode implementation makes full backward
bounds (CROWN) for convolutional layers significantly faster by using more
efficient GPU operators.  The convolution mode can be set by the "conv_mode"
key in the bound_opts parameter when constructing your BoundeModule object and
the new "patches" mode is enabled by default.  In this example we show the
differences between "patches" mode and the old "matrix" mode in memory
consumption, on a relatively large ResNet network.

"""

import sys
import torch
import random
import numpy as np
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import models

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
conv_mode = sys.argv[1] if len(sys.argv) > 1 else 'patches' # conv_mode can be set as 'matrix' or 'patches'

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

## Step 1: Define the model
# model_ori = models.model_resnet(width=1, mult=4)
# model_ori = models.ResNet18(in_planes=2)
# model_ori = models.vnncomp_resnet2b()
model_ori = models.vnncomp_resnet4b()
model_ori = model_ori.to(device=device)

## Step 2: Prepare dataset as usual.
# test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
normalize = torchvision.transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, 
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize]))
# For illustration we only use 1 image from dataset
N = 1
n_classes = 10
image = torch.Tensor(test_data.data[:N]).reshape(N,3,32,32)
# Convert to float between 0. and 1.
image = image.to(torch.float32) / 255.0
if device == 'cuda':
    image = image.cuda()

## Step 3: wrap model with auto_LiRPA.
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
# The new "patches" conv_mode provides an more efficient implementation for convolutional neural networks.
model = BoundedModule(model_ori, image, bound_opts={"conv_mode": conv_mode}, device=device) 

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.1
norm = 2
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = model(image)

# Compute bounds
torch.cuda.empty_cache()
print('Using {} mode to compute convolution.'.format(conv_mode))
lb, ub = model.compute_bounds(IBP=False, C=None, method='backward')

## Step 5: Final output
# pred = pred.detach().cpu().numpy()
lb = lb.detach().cpu().numpy()
ub = ub.detach().cpu().numpy()
for i in range(N):
    # print("Image {} top-1 prediction {}".format(i, label[i]))
    for j in range(n_classes):
        print("f_{j}(x_0): {l:8.5f} <= f_{j}(x_0+delta) <= {u:8.5f}".format(j=j, l=lb[i][j], u=ub[i][j]))
    print()

# Print the GPU memory usage
print('Memory usage in "{}" mode:'.format(conv_mode))
print(torch.cuda.memory_summary())
