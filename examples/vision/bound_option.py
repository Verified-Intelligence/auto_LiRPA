"""
A simple example for bounding neural network outputs with different bound options on ReLU activation functions.

"""
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

## Step 1: Define computational graph by implementing forward()
# This simple model comes from https://github.com/locuslab/convex_adversarial
def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

model = mnist_model()
# Optionally, load the pretrained weights.
checkpoint = torch.load(
    os.path.join(os.path.dirname(__file__), 'pretrained/mnist_a_adv.pth'),
    map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

## Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST(
    './data', train=False, download=True,
    transform=torchvision.transforms.ToTensor())
# For illustration we only use one image from dataset
N = 1
n_classes = 10
image = test_data.data[:N].view(N,1,28,28)
true_label = test_data.targets[:N]
# Convert to float
image = image.to(torch.float32) / 255.0
if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

## Step 3: wrap model with auto_LiRPA
# Use default bound_option
lirpa_model_default = BoundedModule(model, torch.empty_like(image), device=image.device)
# Use same-slope option for ReLU functions
lirpa_model_sameslope = BoundedModule(model, torch.empty_like(image), device=image.device, 
                                      bound_opts={'relu': 'same-slope'})
print('Running on', image.device)

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.3
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = lirpa_model_default(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

print()
print('Demonstration 1.1: Bound computation and comparisons of different options.')
## Step 5: Compute bounds for final output
print('Bounding method:', 'backward (CROWN)')
print('Bounding option:', 'Default (adaptive)')
lb, ub = lirpa_model_default.compute_bounds(x=(image,), method='backward')
for i in range(N):
    print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
    for j in range(n_classes):
        indicator = '(ground-truth)' if j == true_label[i] else ''
        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
print()

print('Bounding option:', 'same-slope')
lb, ub = lirpa_model_sameslope.compute_bounds(x=(image,), method='backward')
for i in range(N):
    print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
    for j in range(n_classes):
        indicator = '(ground-truth)' if j == true_label[i] else ''
        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
print()

print('Demonbstration 1.2: same-slope option is also available with CROWN-Optimized')
print('Bounding method:', 'CROWN-Optimized (alpha-CROWN)')
print('Bounding option:', 'Default (adaptive)')
lb, ub = lirpa_model_default.compute_bounds(x=(image,), method='CROWN-Optimized')
for i in range(N):
    print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
    for j in range(n_classes):
        indicator = '(ground-truth)' if j == true_label[i] else ''
        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
print()

print('Bounding option:', 'same-slope')
lb, ub = lirpa_model_sameslope.compute_bounds(x=(image,), method='CROWN-Optimized')
for i in range(N):
    print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
    for j in range(n_classes):
        indicator = '(ground-truth)' if j == true_label[i] else ''
        print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
print()


print('Demonstration 2: Obtaining linear coefficients of the lower and upper bounds.')
print('With same-slope option, two linear coefficients should be the same.')
# There are many bound coefficients during CROWN bound calculation; here we are interested in the linear bounds
# of the output layer, with respect to the input layer (the image).
required_A = defaultdict(set)
required_A[lirpa_model_sameslope.output_name[0]].add(lirpa_model_sameslope.input_name[0])

print("Bounding method:", 'backward')
print("Bounding option:", 'same-slope')
lb, ub, A_dict = lirpa_model_sameslope.compute_bounds(x=(image,), method='backward', return_A=True, needed_A_dict=required_A)
lower_A, lower_bias = A_dict[lirpa_model_sameslope.output_name[0]][lirpa_model_sameslope.input_name[0]]['lA'], A_dict[lirpa_model_sameslope.output_name[0]][lirpa_model_sameslope.input_name[0]]['lbias']
upper_A, upper_bias = A_dict[lirpa_model_sameslope.output_name[0]][lirpa_model_sameslope.input_name[0]]['uA'], A_dict[lirpa_model_sameslope.output_name[0]][lirpa_model_sameslope.input_name[0]]['ubias']
print(f'lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}')
print(f'lower bound bias term size (batch, output_dim): {list(lower_bias.size())}')
print(f'upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}')
print(f'upper bound bias term size (batch, output_dim): {list(upper_bias.size())}')
print()
print(f'lower bound linear coefficients should be the same as upper bound linear coefficients: {(lower_A - upper_A).abs().max() < 1e-5}')
print()
