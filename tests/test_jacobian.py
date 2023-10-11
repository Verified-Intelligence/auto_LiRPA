# pylint: disable=wrong-import-position
"""Test Jacobian bounds."""

import sys
sys.path.append('../examples/vision')
from jacobian import compute_jacobians
import torch
import torch.nn as nn
from testcase import TestCase
from auto_LiRPA.utils import Flatten


class TestJacobian(TestCase):
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(
            methodName, seed=1, ref_path='data/jacobian_test_data',
            generate=generate)

    def test(self):
        in_dim, width, linear_size = 8, 2, 8
        model = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(width * (in_dim-4)**2, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, 10)
        )
        x0 = torch.randn(1, 3, in_dim, in_dim)
        self.result = compute_jacobians(
            model, x0, bound_opts={'optimize_bound_args': {'iteration': 2}})
        self.check()


if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestJacobian(generate=False)
    testcase.setUp()
    testcase.test()
