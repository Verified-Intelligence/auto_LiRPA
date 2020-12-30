# Test bounds on a 1 layer linear network.

import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase

n_classes = 3
N = 10

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class TestLinearModel(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, seed=0)
        self.original_model = LinearModel()

    def compute_and_compare_bounds(self, eps, norm, IBP, method):
        input_data = torch.randn((N, 256))
        model = BoundedModule(self.original_model, torch.empty_like(input_data))
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        ptb_data = BoundedTensor(input_data, ptb)
        pred = model(ptb_data)
        label = torch.argmax(pred, dim=1).cpu().detach().numpy()
        # Compute bounds.
        lb, ub = model.compute_bounds(IBP=IBP, method=method)
        # Compute dual norm.
        if norm == 1:
            q = np.inf
        elif norm == np.inf:
            q = 1.0
        else:
            q = 1.0 / (1.0 - (1.0 / norm))
        # Compute reference manually.
        weight, bias = list(model.parameters())
        norm = weight.norm(p=q, dim=1)
        expected_pred = input_data.matmul(weight.t()) + bias
        expected_ub = eps * norm + expected_pred
        expected_lb = -eps * norm + expected_pred

        # Check equivalence.
        self.assertEqual(expected_pred, pred)
        self.assertEqual(expected_ub, ub)
        self.assertEqual(expected_lb, lb)

    def test_Linf_forward(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=False, method='forward')

    def test_Linf_backward(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=False, method='backward')

    def test_Linf_IBP(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=True, method=None)

    def test_Linf_backward_IBP(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=True, method='backward')

    def test_L2_forward(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=1.0, norm=2, IBP=False, method='forward')

    def test_L2_backward(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=1.0, norm=2, IBP=False, method='backward')

    def test_L2_IBP(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=1.0, norm=2, IBP=True, method=None)

    def test_L2_backward_IBP(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=1.0, norm=2, IBP=True, method='backward')

    def test_L1_forward(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=3.0, norm=1, IBP=False, method='forward')

    def test_L1_backward(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=3.0, norm=1, IBP=False, method='backward')

    def test_L1_IBP(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=3.0, norm=1, IBP=True, method=None)

    def test_L1_backward_IBP(self):
        with np.errstate(divide='ignore'):
            self.compute_and_compare_bounds(eps=3.0, norm=1, IBP=True, method='backward')

