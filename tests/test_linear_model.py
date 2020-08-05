# Test bounds on a 1 layer linear network.

import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

n_classes = 3
N = 10
torch.manual_seed(0)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

original_model = LinearModel()

input_data = torch.randn((N, 256))

def compute_and_compare_bounds(eps, norm, IBP, method):
    model = BoundedModule(original_model, torch.empty_like(input_data))
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
    torch.allclose(expected_pred, pred)
    torch.allclose(expected_ub, ub)
    torch.allclose(expected_lb, lb)

    """
    pred = pred.detach().cpu().numpy()
    lb = lb.detach().cpu().numpy()
    ub = ub.detach().cpu().numpy()

    print("Computed:")
    for i in range(N):
        for j in range(n_classes):
            print("f_{j}(x_0) = {fx0:8.3f},   {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(j=j, fx0=pred[i][j], l=lb[i][j], u=ub[i][j]))

    # Expected output:
    print("Expected:")
    for i in range(N):
        for j in range(n_classes):
            print("f_{j}(x_0) = {fx0:8.3f},   {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(j=j, fx0=expected_pred[i][j], l=expected_lb[i][j], u=expected_ub[i][j]))
    """

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

