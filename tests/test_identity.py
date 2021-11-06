"""Test a model with an nn.Identity layer only"""
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase  

class TestIdentity(TestCase): 
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test(self):
        model = nn.Sequential(nn.Identity())
        x = torch.randn(2, 10)
        model = BoundedModule(model, x)
        y = model(x)
        eps = 0.1
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        x = BoundedTensor(x, ptb)
        y_l, y_u = model.compute_bounds()
        self.assertTensorEqual(x, y)
        self.assertTensorEqual(y_l, x - eps)
        self.assertTensorEqual(y_u, x + eps)

if __name__ == '__main__':
    testcase = TestIdentity()
    testcase.test()        
