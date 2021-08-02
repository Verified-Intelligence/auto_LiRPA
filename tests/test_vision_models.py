import random
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase

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

class TestVisionModels(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName, seed=1234, ref_path='data/vision_test_data')
        self.result = {}

    def verify_bounds(self, model, x, IBP, method, forward_ret, lb_name, ub_name):
        lb, ub = model(method_opt="compute_bounds", x=(x,), IBP=IBP, method=method)
        self.result[lb_name] = lb
        self.result[ub_name] = ub
        assert torch.allclose(lb, self.reference[lb_name], 1e-4), (lb - self.reference[lb_name]).abs().sum()
        assert torch.allclose(ub, self.reference[ub_name], 1e-4), (ub - self.reference[ub_name]).abs().sum()
        assert ((lb - self.reference[lb_name]).pow(2).sum() < 1e-9), (lb - self.reference[lb_name]).pow(2).sum()
        assert ((ub - self.reference[ub_name]).pow(2).sum() < 1e-9), (ub - self.reference[ub_name]).pow(2).sum()
        # test gradient backward propagation
        loss = (ub - lb).abs().sum()
        loss.backward()
        grad = x.grad
        self.result[lb_name[:-2] + 'grad'] = grad
        assert torch.allclose(grad, self.reference[lb_name[:-2] + 'grad'], 1e-4, 1e-6)
        assert (grad - self.reference[lb_name[:-2] + 'grad']).pow(2).sum() < 1e-9

    def test_bounds(self):
        np.random.seed(123) # FIXME inconsistent seeds

        model_ori = cnn_4layer_test().eval()
        model_ori.load_state_dict(self.reference['model'])
        dummy_input = self.reference['data']
        inputs = (dummy_input,)

        model = BoundedModule(model_ori, inputs)
        forward_ret = model(dummy_input)
        model_ori.eval()

        assert torch.allclose(model_ori(dummy_input), model(dummy_input), 1e-4, 1e-6)

        # Linf
        ptb = PerturbationLpNorm(norm=np.inf, eps=0.01)
        x = BoundedTensor(dummy_input, ptb)
        x.requires_grad_()

        self.verify_bounds(model, x, IBP=True, method=None, forward_ret=forward_ret, lb_name='l_inf_IBP_lb',
                    ub_name='l_inf_IBP_ub')  # IBP
        self.verify_bounds(model, x, IBP=True, method='backward', forward_ret=forward_ret, lb_name='l_inf_CROWN-IBP_lb',
                    ub_name='l_inf_CROWN-IBP_ub')  # CROWN-IBP
        self.verify_bounds(model, x, IBP=False, method='backward', forward_ret=forward_ret, lb_name='l_inf_CROWN_lb',
                    ub_name='l_inf_CROWN_ub')  # CROWN

        # L2
        ptb = PerturbationLpNorm(norm=2, eps=0.01)
        x = BoundedTensor(dummy_input, ptb)
        x.requires_grad_()

        self.verify_bounds(model, x, IBP=True, method=None, forward_ret=forward_ret, lb_name='l_2_IBP_lb',
                    ub_name='l_2_IBP_ub')  # IBP
        self.verify_bounds(model, x, IBP=True, method='backward', forward_ret=forward_ret, lb_name='l_2_CROWN-IBP_lb',
                    ub_name='l_2_CROWN-IBP_ub')  # CROWN-IBP
        self.verify_bounds(model, x, IBP=False, method='backward', forward_ret=forward_ret, lb_name='l_2_CROWN_lb',
                    ub_name='l_2_CROWN_ub')  # CROWN

        if self.generate:
            for item in self.result:
                self.reference = self.result[item]
            self.save()
