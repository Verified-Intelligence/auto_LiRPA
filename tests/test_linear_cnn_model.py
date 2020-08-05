# Test bounds on a 1 layer CNN network.

import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import pytest

input_dim = 8
out_channel = 2
N = 10
torch.manual_seed(0)

class LinearCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channel, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, input_dim //2 * input_dim // 2 * out_channel)
        return x

original_model = LinearCNNModel()

input_data = torch.randn((N, 1, input_dim, input_dim))

def compute_and_compare_bounds(eps, norm, IBP, method):
    model = BoundedModule(original_model, torch.empty_like(input_data))
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
    weight = original_model.conv(matrix_eye) - conv_bias # Output is (batch, channel, weight, height).
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
    torch.allclose(expected_pred, pred)
    torch.allclose(expected_ub, ub)
    torch.allclose(expected_lb, lb)

@pytest.mark.skip(reason="Not Implemented")
def test_Linf_forward():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=False, method='forward')

def test_Linf_backward():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=False, method='backward')

def test_Linf_IBP():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=True, method=None)

def test_Linf_backward_IBP():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=0.3, norm=np.inf, IBP=True, method='backward')

@pytest.mark.skip(reason="Not Implemented")
def test_L2_forward():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=1.0, norm=2, IBP=False, method='forward')

def test_L2_backward():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=1.0, norm=2, IBP=False, method='backward')

def test_L2_IBP():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=1.0, norm=2, IBP=True, method=None)

def test_L2_backward_IBP():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=1.0, norm=2, IBP=True, method='backward')

@pytest.mark.skip(reason="Not Implemented")
def test_L1_forward():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=3.0, norm=1, IBP=False, method='forward')

def test_L1_backward():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=3.0, norm=1, IBP=False, method='backward')

def test_L1_IBP():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=3.0, norm=1, IBP=True, method=None)

def test_L1_backward_IBP():
    with np.errstate(divide='ignore'):
        compute_and_compare_bounds(eps=3.0, norm=1, IBP=True, method='backward')


if __name__ == '__main__':
    test_Linf_forward()
    test_Linf_backward()
    test_Linf_IBP()
    test_Linf_backward_IBP()
    test_L2_forward()
    test_L2_backward()
    test_L2_IBP()
    test_L2_backward_IBP()
    test_L1_forward()
    test_L1_backward()
    test_L1_IBP()
    test_L1_backward_IBP()

