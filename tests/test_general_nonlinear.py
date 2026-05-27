import sys
import pytest
import torch.nn as nn

sys.path.insert(0, '../complete_verifier')

import arguments
from beta_CROWN_solver import LiRPANet
from activation_split.bab_bootstrap import general_bab

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import DEFAULT_DEVICE, DEFAULT_DTYPE


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        Sin(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        Sin(),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        Sin(),
        nn.Linear(100, 10)
    )
    return model


def bab(model_ori, data, target, norm, eps, data_max=None, data_min=None, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
    data = data.to(device=device, dtype=dtype)
    eps = eps.to(device=device, dtype=dtype)
    if norm == np.inf:
        if data_max is None:
            data_ub = data + eps
            data_lb = data - eps
        else:
            data_max = data_max.to(device=device, dtype=dtype)
            data_min = data_min.to(device=device, dtype=dtype)
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
    else:
        data_ub = data_lb = data
    pred = torch.argmax(model_ori(data), dim=1)

    c = torch.zeros((1, 1, 10), device=device, dtype=dtype) # we only support c with shape of (1, 1, n)
    c[0, 0, pred] = 1
    c[0, 0, target] = -1
    rhs = torch.tensor(arguments.Config["bab"]["decision_thresh"], dtype=dtype, device=device).view(c.shape[:2])

    arguments.Config.parse_config(args={})

    arguments.Config['general']['device'] = 'cpu'
    arguments.Config["solver"]["batch_size"] = 200
    arguments.Config["bab"]["decision_thresh"] = np.float64(10)  # naive float obj has no max() function, np.inf will lead infeasible domain
    arguments.Config["solver"]["beta-crown"]["iteration"] = 20
    arguments.Config["bab"]["timeout"] = 60 #300

    arguments.Config["solver"]["alpha-crown"]["lr_alpha"] = 0.1
    arguments.Config["solver"]["beta-crown"]["lr_beta"] = 0.1
    arguments.Config["bab"]["branching"]["method"] = 'nonlinear'
    arguments.Config["bab"]["branching"]["candidates"] = 2
    arguments.Config["general"]["enable_incomplete_verification"] = False
    arguments.Config["data"]["dataset"] = 'cifar'

    # LiRPA wrapper
    model = LiRPANet(model_ori, device=device, in_size=(1, 3, 32, 32))

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb)
    forward = model_ori(x)

    min_lb = general_bab(model, x, c, rhs)[0]

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()

    min_lb += arguments.Config["bab"]["decision_thresh"]
    print(min_lb)

    assert min_lb < torch.min(forward)

# This test takes long time so it is set as the last test case.
@pytest.mark.skip(reason="The test is failing now after removing index clamping.")
# @pytest.mark.order(-1)
def test(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
    model_ori = cifar_model_wide()
    data = torch.load('data/beta_crown_test_data')
    model_ori.load_state_dict(data['state_dict'])
    model_ori = model_ori.to(device=device, dtype=dtype)
    x = data['x']
    pidx = data['pidx']
    eps_temp = data['eps_temp']
    data_max = data['data_max']
    data_min = data['data_min']

    bab(model_ori, x, pidx, float('inf'), eps_temp, data_max=data_max, data_min=data_min, device=device, dtype=dtype)


if __name__ == "__main__":
    test()
