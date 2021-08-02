# Test bounds on a 1 layer linear network.

import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import torch.nn.functional as F
import numpy as np

n_classes = 3
N = 10
torch.manual_seed(0)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, n_classes)


    def forward(self, x):
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (2, 2))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc(x.view(x.size(0), -1))
        return x

original_model = LinearModel()

input_data = torch.randn((N, 3, 64, 64))

def compute_bounds(eps, norm, IBP, method):
    model = BoundedModule(original_model, torch.empty_like(input_data))
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    ptb_data = BoundedTensor(input_data, ptb)
    pred = model(ptb_data)
    label = torch.argmax(pred, dim=1).cpu().detach().numpy()
    # Compute bounds.
    lb, ub = model.compute_bounds(IBP=IBP, method=method)


if __name__ == '__main__':
    compute_bounds(0.1, np.inf, True, None)
    compute_bounds(0.1, np.inf, False, 'backward')
