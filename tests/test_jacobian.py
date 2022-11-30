"""Test Jacobian bounds."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16**2, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.flatten(x, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2, 3, 7, stride=1, padding=0)
        self.fc1 = nn.Linear(48, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class TestJacobian(TestCase):
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(
            methodName, seed=1, ref_path='data/jacobian_test_data',
            generate=generate)

    def test(self):
        image = torch.randn(1, 1, 16, 16)
        model = CNN()
        model = BoundedModule(model, image, device='cpu')
        ptb = PerturbationLpNorm(eps=0.1)
        x = BoundedTensor(image, ptb)
        output = model(x)
        print(output)
        model.augment_gradient_graph(x)
        ret = model.compute_jacobian_bounds(
            x, labels=torch.tensor([1], dtype=torch.long))
        print(ret)
        self.result = [ret]
        self.check()


if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestJacobian(generate=False)
    testcase.setUp()
    testcase.test()
