"""Examples of computing Jacobian bounds.

We use a small model with two convolutional layers and dense layers respectively.
The width of the model has been reduced for the demonstration here. And we use
data from CIFAR-10.

We show examples of:
- Computing Jacobian bounds
- Computing Linf local Lipschitz constants
- Computing JVP bounds
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten


def build_model(in_ch=3, in_dim=32, width=32, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(width * (in_dim-4)**2, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def compute_jacobians(model_ori, x0, bound_opts=None, device='cpu'):
    """Compute Jacobians given a model and an input."""

    results = [[] for _ in range(3)]

    model_ori = model_ori.to(device)
    x0 = x0.to(device)
    print('Model:', model_ori)

    # Example 1: Convert the model for Jacobian bound computation
    model = BoundedModule(model_ori, x0, bound_opts=bound_opts, device=device)
    model.augment_gradient_graph(x0)

    # Sanity check to ensure that the new graph matches the original gradient computation
    y = model_ori(x0.requires_grad_(True))
    ret_ori = torch.autograd.grad(y.sum(), x0)[0].view(1, -1)
    # After running augment_gradient_graph, the model takes an additional input
    # (the second input) which is a linear mapping applied on the output of the
    # model before computing the gradient. It is the same as "grad_outputs" in
    # torch.autograd.grad, which is "the 'vector' in the vector-Jacobian product".
    # Here, setting torch.ones(1, 10) is equivalent to computing the gradients for
    # y.sum() above.
    ret_new = model(x0, torch.ones(1, 10).to(x0))
    assert torch.allclose(ret_ori, ret_new)

    for eps in [0, 1./255, 4./255]:
        # The input region considered is an Linf ball with radius eps around x0.
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
        # Compute the Linf locaal Lipscphitz constant
        lower, upper = model.compute_jacobian_bounds(x)
        print(f'Gap between upper and lower Jacobian bound for eps={eps:.5f}',
            (upper - lower).max())
        if eps == 0:
            assert torch.allclose(ret_new, lower.sum(dim=0, keepdim=True))
            assert torch.allclose(ret_new, upper.sum(dim=0, keepdim=True))
        results[0].append((lower.detach(), upper.detach()))

    # Example 2: Convert the model for Linf local Lipschitz constant computation
    model = BoundedModule(model_ori, x0, bound_opts=bound_opts, device=device)
    # Set norm=np.inf for Linf local Lipschitz constant
    model.augment_gradient_graph(x0, norm=np.inf)

    # Sanity check to ensure that the new graph matches the original gradient computation
    y = model_ori(x0.requires_grad_(True))
    ret_ori = torch.autograd.grad(y.sum(), x0)[0].abs().sum().view(-1)
    ret_new = model(x0, torch.ones(1, 10).to(x0)).view(-1)
    assert torch.allclose(ret_ori, ret_new)

    for eps in [0, 1./255, 4./255]:
        # The input region considered is an Linf ball with radius eps around x0.
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
        # Compute the Linf locaal Lipschitz constant
        result = model.compute_jacobian_bounds(x)
        print(f'Linf local Lipschitz constant for eps={eps:.5f}', result)
        results[1].append(result.detach())

    # Example 3: Convert the model for Jacobian-Vector Product (JVP) computation
    model = BoundedModule(model_ori, x0, bound_opts=bound_opts, device=device)
    vector = torch.rand_like(x0)
    # Set vector for JVP computation
    model.augment_gradient_graph(x0, vector=vector)

    # Sanity check to ensure that the new graph matches the original JVP
    def func(x0):
        return model_ori(x0.requires_grad_(True))
    ret_ori = torch.autograd.functional.jvp(func, x0, vector)[-1].view(-1)
    ret_new = torch.zeros(10).to(x0)
    for i in range(10):
        c = F.one_hot(torch.tensor([i], dtype=torch.long), 10).to(x0)
        ret_new[i] = model(x0, c)
    assert torch.allclose(ret_ori, ret_new)

    for eps in [0, 1./255, 4./255]:
        # The input region considered is an Linf ball with radius eps around x0.
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
        # Compute the JVP
        lower, upper = model.compute_jacobian_bounds(x)
        print(f'JVP lower bound for eps={eps:.5f}', lower.view(-1))
        print(f'JVP upper bound for eps={eps:.5f}', upper.view(-1))
        results[2].append((lower.detach(), upper.detach()))

    return results


def run_jacobian_examples():
    torch.manual_seed(0)

    # Create a small model and load pre-trained parameters.
    model_ori = build_model(width=4, linear_size=32)
    model_ori.load_state_dict(torch.load('pretrained/cifar_2c2f.pth'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare the dataset
    test_data = datasets.CIFAR10('./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2009, 0.2009, 0.2009])]))
    x0 = test_data[0][0].unsqueeze(0)

    return compute_jacobians(model_ori, x0, device=device)


if __name__ == '__main__':
    run_jacobian_examples()