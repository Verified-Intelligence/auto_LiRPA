import random

import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

# with open('data/vision_test_data.pickle', 'rb') as handle:
#     data = pickle.load(handle)
data = torch.load('data/vision_test_data')


class cnn_4layer_test(nn.Module):
    def __init__(self):
        super(cnn_4layer_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.shortcut = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(192, 10)

    def forward(self, x):
        x_ = x
        x = F.relu(self.conv1(self.bn(x)))
        x += self.shortcut(x_)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def verify_bounds(model, x, IBP, method, forward_ret, lb_name, ub_name):
    lb, ub = model(method_opt="compute_bounds", x=(x,), IBP=IBP, method=method)
    assert torch.allclose(lb, data[lb_name], 1e-4)
    assert torch.allclose(ub, data[ub_name], 1e-4)
    assert ((lb - data[lb_name]).pow(2).sum() < 1e-12)
    assert ((ub - data[ub_name]).pow(2).sum() < 1e-12)

    # test gradient backward propagation
    loss = (ub - lb).abs().sum()
    loss.backward()

    # grad_sum = sum((p.grad.abs().sum()) for n, p in model.named_parameters() if p.grad is not None)
    # gradient w.r.t input only
    grad = x.grad
    # data[lb_name[:-2]+'grad'] = grad.detach().data.clone()
    assert torch.allclose(grad, data[lb_name[:-2] + 'grad'], 1e-4)
    assert ((grad - data[lb_name[:-2] + 'grad']).pow(2).sum() < 1e-12)


def test():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)

    model_ori = cnn_4layer_test().eval()
    model_ori.load_state_dict(data['model'])
    dummy_input = data['data']
    inputs = (dummy_input,)

    model = BoundedModule(model_ori, inputs)
    forward_ret = model(dummy_input)
    model_ori.eval()

    assert torch.isclose(model_ori(dummy_input), model_ori(dummy_input), 1e-8).all()

    # Linf
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.01)
    x = BoundedTensor(dummy_input, ptb)
    x.requires_grad_()

    verify_bounds(model, x, IBP=True, method=None, forward_ret=forward_ret, lb_name='l_inf_IBP_lb',
                  ub_name='l_inf_IBP_ub')  # IBP
    verify_bounds(model, x, IBP=True, method='backward', forward_ret=forward_ret, lb_name='l_inf_CROWN-IBP_lb',
                  ub_name='l_inf_CROWN-IBP_ub')  # CROWN-IBP
    verify_bounds(model, x, IBP=False, method='backward', forward_ret=forward_ret, lb_name='l_inf_CROWN_lb',
                  ub_name='l_inf_CROWN_ub')  # CROWN

    # L2
    ptb = PerturbationLpNorm(norm=2, eps=0.01)
    x = BoundedTensor(dummy_input, ptb)
    x.requires_grad_()

    verify_bounds(model, x, IBP=True, method=None, forward_ret=forward_ret, lb_name='l_2_IBP_lb',
                  ub_name='l_2_IBP_ub')  # IBP
    verify_bounds(model, x, IBP=True, method='backward', forward_ret=forward_ret, lb_name='l_2_CROWN-IBP_lb',
                  ub_name='l_2_CROWN-IBP_ub')  # CROWN-IBP
    verify_bounds(model, x, IBP=False, method='backward', forward_ret=forward_ret, lb_name='l_2_CROWN_lb',
                  ub_name='l_2_CROWN_ub')  # CROWN

    # torch.save(data, 'data/vision_test_data')


if __name__ == "__main__":
    test()
