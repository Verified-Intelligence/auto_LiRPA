import copy
import random

import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedParameter
from auto_LiRPA.perturbations import *

# with open('data/weight_perturbation_test_data.pickle', 'rb') as handle:
#     data = pickle.load(handle)
data = torch.load('data/weight_perturbation_test_data')


class mlp_3layer(nn.Module):
    def __init__(self, in_ch=1, in_dim=28, width=1, pert_weight=True, pert_bias=True, norm=2):
        super(mlp_3layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 64 * width)
        self.fc2 = nn.Linear(64 * width, 64 * width)
        self.fc3 = nn.Linear(64 * width, 10)

        eps = 0.01
        global ptb
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        if pert_weight:
            self.fc1.weight = BoundedParameter(self.fc1.weight.data, ptb)
            self.fc2.weight = BoundedParameter(self.fc2.weight.data, ptb)
            self.fc3.weight = BoundedParameter(self.fc3.weight.data, ptb)

        if pert_bias:
            self.fc1.bias = BoundedParameter(self.fc1.bias.data, ptb)
            self.fc2.bias = BoundedParameter(self.fc2.bias.data, ptb)
            self.fc3.bias = BoundedParameter(self.fc3.bias.data, ptb)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def verify_bounds(model, x, IBP, method, forward_ret, lb_name, ub_name):
    lb, ub = model(method_opt="compute_bounds", x=(x,), IBP=IBP, method=method)
    assert torch.allclose(lb, data[lb_name], 1e-3)
    assert torch.allclose(ub, data[ub_name], 1e-3)
    assert ((lb - data[lb_name]).pow(2).sum() < 1e-9)
    assert ((ub - data[ub_name]).pow(2).sum() < 1e-9)

    # test gradient backward propagation
    loss = (ub - lb).abs().sum()
    loss.backward()

    # grad_sum = sum((p.grad.abs().sum()) for n, p in model.named_parameters() if p.grad is not None)
    # gradient w.r.t input only
    grad = x.grad
    # data[lb_name+'_grad'] = grad.detach().data.clone()
    assert torch.allclose(grad, data[lb_name + '_grad'], 1e-3)
    assert ((grad - data[lb_name + '_grad']).pow(2).sum() < 1e-9)


def test():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)

    model_ori = mlp_3layer(pert_weight=True, pert_bias=True).eval()
    model_ori.load_state_dict(data['model'])
    state_dict = copy.deepcopy(model_ori.state_dict())
    dummy_input = data['data'].requires_grad_()
    inputs = (dummy_input,)

    model = BoundedModule(model_ori, inputs)
    forward_ret = model(dummy_input)
    model_ori.eval()

    assert torch.isclose(model_ori(dummy_input), model_ori(dummy_input), 1e-8).all()

    def verify_model(pert_weight=True, pert_bias=True, norm=np.inf, lb_name='', ub_name=''):
        model_ori_ = mlp_3layer(pert_weight=pert_weight, pert_bias=pert_bias, norm=norm).eval()
        model_ori_.load_state_dict(state_dict)
        model_ = BoundedModule(model_ori_, inputs)

        ptb.eps = 0.01

        verify_bounds(model_, dummy_input, IBP=True, method='backward', forward_ret=forward_ret,
                      lb_name=lb_name + '_CROWN-IBP', ub_name=ub_name + '_CROWN-IBP')  # CROWN-IBP
        verify_bounds(model_, dummy_input, IBP=False, method='backward', forward_ret=forward_ret,
                      lb_name=lb_name + '_CROWN', ub_name=ub_name + '_CROWN')  # CROWN

    # Linf
    verify_model(pert_weight=True, pert_bias=True, norm=np.inf, lb_name='l_inf_weights_bias_lb',
                 ub_name='l_inf_weights_bias_ub')
    verify_model(pert_weight=True, pert_bias=False, norm=np.inf, lb_name='l_inf_weights_lb', ub_name='l_inf_weights_ub')
    verify_model(pert_weight=False, pert_bias=True, norm=np.inf, lb_name='l_inf_bias_lb', ub_name='l_inf_bias_ub')

    # L2
    verify_model(pert_weight=True, pert_bias=True, norm=2, lb_name='l_2_weights_bias_lb', ub_name='l_2_weights_bias_ub')
    verify_model(pert_weight=True, pert_bias=False, norm=2, lb_name='l_2_weights_lb', ub_name='l_2_weights_ub')
    verify_model(pert_weight=False, pert_bias=True, norm=2, lb_name='l_2_bias_lb', ub_name='l_2_bias_ub')
    # torch.save(data, 'data/weight_perturbation_test_data')


if __name__ == "__main__":
    test()
