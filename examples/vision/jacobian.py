"""Examples of computing Jacobian bounds.

We show examples of:
- Computing Jacobian bounds
- Computing Linf local Lipschitz constants
- Computing JVP bounds
"""

import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from auto_LiRPA.jacobian import JacobianOP, GradNorm


def build_model(in_ch=3, in_dim=32):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(in_ch*in_dim**2, 100),
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )
    return model


def example_jacobian(model_ori, x0, bound_opts, device):
    """Example: computing Jacobian bounds."""

    class JacobianWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            y = self.model(x)
            return JacobianOP.apply(y, x)

    model = BoundedModule(JacobianWrapper(model_ori), x0, bound_opts=bound_opts, device=device)

    def func(x0):
        return model_ori(x0.requires_grad_(True))
    ret_ori = torch.autograd.functional.jacobian(func, x0).squeeze(2)
    ret_new = model(x0)
    assert torch.allclose(ret_ori, ret_new)

    ret = []
    for eps in [0, 1./255, 4./255]:
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
        lower, upper = model.compute_jacobian_bounds(x)
        print(f'Gap between upper and lower Jacobian bound for eps={eps:.5f}',
            (upper - lower).max())
        if eps == 0:
            assert torch.allclose(
                ret_new.view(-1),
                lower.sum(dim=0, keepdim=True).view(-1))
            assert torch.allclose(
                ret_new.view(-1),
                upper.sum(dim=0, keepdim=True).view(-1))
        ret.append((lower.detach(), upper.detach()))

    return ret


def example_local_lipschitz(model_ori, x0, bound_opts, device):
    """Example: computing Linf local Lipschitz constant."""

    class LocalLipschitzWrapper(nn.Module):
        def __init__(self, model, mask):
            super().__init__()
            self.model = model
            self.mask = mask
            self.grad_norm = GradNorm(norm=1)

        def forward(self, x):
            y = self.model(x)
            y_selected = y.matmul(self.mask)
            jacobian = JacobianOP.apply(y_selected, x)
            lipschitz = self.grad_norm(jacobian)
            return lipschitz

    mask = torch.zeros(10, 1, device=device)
    mask[1, 0] = 1
    model = BoundedModule(LocalLipschitzWrapper(model_ori, mask=mask), (x0),
                          bound_opts=bound_opts, device=device)

    y = model_ori(x0.requires_grad_(True))
    ret_ori = torch.autograd.grad(y[:, 1].sum(), x0)[0].abs().flatten(1).sum(dim=-1).view(-1)
    ret_new = model(x0, mask).view(-1)
    assert torch.allclose(ret_ori, ret_new)

    ret = []
    for eps in [0, 1./255, 4./255]:
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
        lip = []
        for i in range(mask.shape[0]):
            mask.zero_()
            mask[i, 0] = 1
            ub = model.compute_jacobian_bounds((x, mask), bound_lower=False)[1]
            lip.append(ub)
        lip = torch.concat(lip).max()
        print(f'Linf local Lipschitz constant for eps={eps:.5f}: {lip.item()}')
        ret.append(lip.detach())

    return ret


def example_jvp(model_ori, x0, bound_opts, device):
    """Example: computing Jacobian-Vector Product."""

    class JVPWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.grad_norm = GradNorm(norm=1)

        def forward(self, x, v):
            y = self.model(x)
            jacobian = JacobianOP.apply(y, x).flatten(2)
            jvp = (jacobian * v.flatten(1).unsqueeze(1)).sum(dim=-1)
            return jvp

    vector = torch.rand_like(x0)
    model = BoundedModule(JVPWrapper(model_ori), (x0, vector),
                          bound_opts=bound_opts, device=device)

    def func(x0):
        return model_ori(x0.requires_grad_(True))
    ret_ori = torch.autograd.functional.jvp(func, x0, vector)[-1].view(-1)
    ret_new = model(x0, vector)
    assert torch.allclose(ret_ori, ret_new)

    ret = []
    for eps in [0, 1./255, 4./255]:
        x = BoundedTensor(x0, PerturbationLpNorm(norm=np.inf, eps=eps))
        lb, ub = model.compute_jacobian_bounds((x, vector))
        print(f'JVP lower bound for eps={eps:.5f}: {lb}')
        print(f'JVP upper bound for eps={eps:.5f}: {ub}')
        ret.append((lb, ub))

    return ret


def compute_jacobians(model_ori, x0, bound_opts=None, device='cpu'):
    results = [[] for _ in range(3)]

    model_ori = model_ori.to(device)
    x0 = x0.to(device)
    print('Model:', model_ori)

    results[0] = example_jacobian(model_ori, x0, bound_opts, device)
    results[1] = example_local_lipschitz(model_ori, x0, bound_opts, device)
    results[2] = example_jvp(model_ori, x0, bound_opts, device)

    return results


if __name__ == '__main__':
    torch.manual_seed(0)

    # Create a small model and load pre-trained parameters.
    model_ori = build_model(in_dim=8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x0 = torch.randn(1, 3, 8, 8, device=device)

    compute_jacobians(model_ori, x0, device=device)
