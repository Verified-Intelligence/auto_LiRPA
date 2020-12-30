# Test bounds on a 1 layer CNN network.

import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from test_linear_model import TestLinearModel

input_dim = 8
out_channel = 2
N = 10

class LinearCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channel, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, input_dim //2 * input_dim // 2 * out_channel)
        return x

class TestLinearCNNModel(TestLinearModel): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName)
        self.original_model = LinearCNNModel()

    def compute_and_compare_bounds(self, eps, norm, IBP, method):
        input_data = torch.randn((N, 1, input_dim, input_dim))
        model = BoundedModule(self.original_model, torch.empty_like(input_data))
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        ptb_data = BoundedTensor(input_data, ptb)
        pred = model(ptb_data)
        label = torch.argmax(pred, dim=1).cpu().detach().numpy()
        # Compute bounds.
        lb, ub = model.compute_bounds(IBP=IBP, method=method)
        # Compute reference.
        conv_weight, conv_bias = list(model.parameters())
        conv_bias = conv_bias.view(1, out_channel, 1, 1)
        matrix_eye = torch.eye(input_dim * input_dim).view(input_dim * input_dim, 1, input_dim, input_dim)
        # Obtain equivalent weight and bias for convolution.
        weight = self.original_model.conv(matrix_eye) - conv_bias # Output is (batch, channel, weight, height).
        weight = weight.view(input_dim * input_dim, -1) # Dimension is (flattened_input, flattened_output).
        bias = conv_bias.repeat(1, 1, input_dim //2, input_dim //2).view(-1)
        flattend_data = input_data.view(N, -1)
        # Compute dual norm.
        if norm == 1:
            q = np.inf
        elif norm == np.inf:
            q = 1.0
        else:
            q = 1.0 / (1.0 - (1.0 / norm))
        # Manually compute bounds.
        norm = weight.t().norm(p=q, dim=1)
        expected_pred = flattend_data.matmul(weight) + bias
        expected_ub = eps * norm + expected_pred
        expected_lb = -eps * norm + expected_pred
        # Check equivalence.
        if method == 'backward' or method == 'forward':
            self.assertEqual(expected_pred, pred)
            self.assertEqual(expected_ub, ub)
            self.assertEqual(expected_lb, lb)
