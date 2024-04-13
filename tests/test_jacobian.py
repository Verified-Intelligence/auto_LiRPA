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
        in_dim, linear_size = 8, 100
        model = nn.Sequential(
            Flatten(),
            nn.Linear(3*in_dim**2, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, 10),
        )
        x0 = torch.randn(1, 3, in_dim, in_dim)
        self.result = compute_jacobians(model, x0)
        self.check()


if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestJacobian(generate=False)
    testcase.setUp()
    testcase.test()
