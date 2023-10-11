"""Test two dimensional activation functions (e.g., min, max, etc)"""
import tqdm
import torch
import torch.nn as nn
from testcase import TestCase
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import logger


# Wrap the computation with a nn.Module
class test_model(nn.Module):
    def __init__(self, act_func):
        super().__init__()
        self.act_func = act_func

    def forward(self, x, y):
        return self.act_func(x, y)


def mul(x, y):
    return x * y


class Test2DActivation(TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def create_test(self, act_func, low_x, high_x, low_y, high_y,
                    ntests=10000, nsamples=1000, method='IBP'):
        print(f'Testing activation {act_func}')

        model = test_model(act_func)
        image = torch.zeros(2, ntests)
        bounded_model = BoundedModule(model, (image[0], image[1]), device=torch.device('cpu'))

        # Generate randomly bounded inputs.
        p_x = torch.rand(1, ntests) * (high_x - low_x) + low_x
        q_x = torch.rand(1, ntests) * (high_x - low_x) + low_x
        input_lb_x = torch.min(p_x, q_x)
        input_ub_x = torch.max(p_x, q_x)
        input_center_x = (input_lb_x + input_ub_x) / 2.0
        ptb_x = PerturbationLpNorm(x_L=input_lb_x, x_U=input_ub_x)
        ptb_data_x = BoundedTensor(input_center_x, ptb_x)

        p_y = torch.rand(1, ntests) * (high_y - low_y) + low_y
        q_y = torch.rand(1, ntests) * (high_y - low_y) + low_y
        input_lb_y = torch.min(p_y, q_y)
        input_ub_y = torch.max(p_y, q_y)
        input_center_y = (input_lb_y + input_ub_y) / 2.0
        ptb_y = PerturbationLpNorm(x_L=input_lb_y, x_U=input_ub_y)
        ptb_data_y = BoundedTensor(input_center_y, ptb_y)

        # Generate reference results.
        range_xy = torch.linspace(start=low_x, end=high_x, steps=nsamples+1)
        table = torch.empty([range_xy.shape[0], range_xy.shape[0]])
        for i in range(range_xy.shape[0]):
            x = range_xy[i]
            table_y = act_func(x, torch.linspace(start=low_y, end=high_y, steps=nsamples+1))
            table[i] = table_y
        def lookup(l_x, u_x, l_y, u_y):
            assert torch.all(u_x <= high_x)
            assert torch.all(l_x >= low_x)
            assert torch.all(u_y <= high_y)
            assert torch.all(l_y >= low_y)
            shape = l_x.size()
            l_x = l_x.squeeze()
            u_x = u_x.squeeze()
            l_y = l_y.squeeze()
            u_y = u_y.squeeze()
            # select all sample points between l and u.
            low_index_x = torch.ceil((l_x - low_x) / (high_x - low_x) * nsamples).int()  # Make sure we do not have index 0.
            high_index_x = torch.floor((u_x - low_x) / (high_x - low_x) * nsamples).int()
            low_index_y = torch.ceil((l_y - low_y) / (high_y - low_y) * nsamples).int()  # Make sure we do not have index 0.
            high_index_y = torch.floor((u_y - low_y) / (high_y - low_y) * nsamples).int()
            real_lb = torch.empty_like(l_x)
            real_ub = torch.empty_like(u_x)
            for i, (li_x, hi_x) in enumerate(zip(low_index_x, high_index_x)):
                li_y = low_index_y[i]
                hi_y = high_index_y[i]
                if li_x == hi_x + 1 or li_y == hi_y + 1:
                    # Not enough precision. l and u are too close so we cannot tell.
                    real_lb[i] = float("inf")
                    real_ub[i] = float("-inf")
                else:
                    selected = table[li_x : hi_x+1, li_y : hi_y+1].reshape(-1)
                    real_lb[i] = torch.min(selected)
                    real_ub[i] = torch.max(selected)
            real_lb = real_lb.view(*shape)
            real_ub = real_ub.view(*shape)
            return real_lb, real_ub
        # These are reference results. IBP results should be very close to these. Linear bound results can be looser than these.
        ref_forward = model(input_center_x, input_center_y)
        ref_output_lb, ref_output_ub = lookup(input_lb_x, input_ub_x, input_lb_y, input_ub_y)

        # Get bounding results.
        forward = bounded_model(ptb_data_x, ptb_data_y)
        output_lb, output_ub = bounded_model.compute_bounds(x=(ptb_data_x, ptb_data_y), method = method)

        # Compare.
        assert torch.allclose(forward, ref_forward)
        for i in tqdm.tqdm(range(ntests)):
            show = False
            if output_ub[0,i] < ref_output_ub[0,i] - 1e-5:
                logger.warning(f'upper bound is wrong {ref_output_ub[0,i] - output_ub[0,i]}')
                show = True
            if output_lb[0,i] > ref_output_lb[0,i] + 1e-5:
                logger.warning(f'lower bound is wrong {output_lb[0,i] - ref_output_lb[0,i]}')
                show = True
            if show:
                logger.warning(f'input_lb_x={input_lb_x[0,i]:8.3f}, input_ub_x={input_ub_x[0,i]:8.3f},input_lb_y={input_lb_y[0,i]:8.3f}, input_ub_y={input_ub_y[0,i]:8.3f}, lb={output_lb[0,i]:8.3f}, ref_lb={ref_output_lb[0,i]:8.3f}, ub={output_ub[0,i]:8.3f}, ref_ub={ref_output_ub[0,i]:8.3f}')
        assert torch.all(output_ub + 1e-5 >= ref_output_ub)
        assert torch.all(output_lb - 1e-5 <= ref_output_lb)

    def test_max(self):
        self.create_test(act_func=torch.max, low_x=-10, high_x=5, low_y=-1, high_y=10, method='IBP')
        self.create_test(act_func=torch.max, low_x=-10, high_x=5, low_y=-1, high_y=10, method='CROWN')

    def test_min(self):
        self.create_test(act_func=torch.min, low_x=-10, high_x=5, low_y=-1, high_y=10, method='IBP')
        self.create_test(act_func=torch.min, low_x=-10, high_x=5, low_y=-1, high_y=10, method='CROWN')

    def test_mul(self):
        self.create_test(act_func=mul, low_x=-10, high_x=5, low_y=-1, high_y=10, method='IBP')
        self.create_test(act_func=mul, low_x=-10, high_x=5, low_y=-1, high_y=10, method='CROWN')

if __name__ == '__main__':
    testcase = Test2DActivation()
    testcase.test_max()
    testcase.test_min()
    testcase.test_mul()
