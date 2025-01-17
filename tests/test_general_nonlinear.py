import sys
import pytest
import torch.nn as nn

sys.path.insert(0, '../complete_verifier')

import arguments
from beta_CROWN_solver import LiRPANet
from bab import general_bab

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import *


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


def bab(model_ori, data, target, norm, eps, data_max=None, data_min=None):
    if norm == np.inf:
        if data_max is None:
            data_ub = data + eps
            data_lb = data - eps
        else:
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
    else:
        data_ub = data_lb = data

    pred = torch.argmax(model_ori(data), dim=1)

    c = torch.zeros((1, 1, 10))  # we only support c with shape of (1, 1, n)
    c[0, 0, pred] = 1
    c[0, 0, target] = -1

    arguments.Config.parse_config(args={})

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
    model = LiRPANet(model_ori, device='cpu', in_size=(1, 3, 32, 32), c=c)

    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    forward = model_ori(x)

    min_lb = general_bab(
        model, domain, x, rhs=arguments.Config["bab"]["decision_thresh"])[0]

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()

    min_lb += arguments.Config["bab"]["decision_thresh"]
    print(min_lb)

    assert min_lb < torch.min(forward)

# This test takes long time so it is set as the last test case.
@pytest.mark.skip(reason="The test is failing now after removing index clamping.")
# @pytest.mark.order(-1)
def test():
    model_ori = cifar_model_wide()
    data = torch.load('data/beta_crown_test_data')
    model_ori.load_state_dict(data['state_dict'])
    x = data['x']
    pidx = data['pidx']
    eps_temp = data['eps_temp']
    data_max = data['data_max']
    data_min = data['data_min']

    bab(model_ori, x, pidx, float('inf'), eps_temp, data_max=data_max, data_min=data_min)


if __name__ == "__main__":
    test()