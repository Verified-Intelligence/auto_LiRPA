"""Test one dimensional activation functions (e.g., ReLU, tanh, exp, sin, etc)"""
import pytest
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

    def forward(self, x):
        return self.act_func(x)

def pow_2(x):
    return torch.pow(x, 2)

def pow_3(x):
    return torch.pow(x, 3)

class GELUOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op('custom::Gelu', x)

    @staticmethod
    def forward(ctx, x):
        return torch.nn.functional.gelu(x)

def GELU(x):
    return GELUOp.apply(x)


class Test1DActivation(TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def create_test(self, act_func, low, high, ntests=1000, nsamples=1000,
                    method='IBP'):
        print(f'Testing activation {act_func} (method {method})')

        model = test_model(act_func)
        image = torch.zeros(1, ntests)
        bounded_model = BoundedModule(
            model, image, bound_opts={
                'optimize_bound_args': {'iteration': 2},
            })

        # Generate randomly bounded inputs.
        p = torch.rand(1, ntests) * (high - low ) + low
        q = torch.rand(1, ntests) * (high - low ) + low
        input_lb = torch.min(p, q)
        input_ub = torch.max(p, q)
        input_center = (input_lb + input_ub) / 2.0
        ptb = PerturbationLpNorm(norm=float("inf"), eps=None, x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor(input_center, ptb)

        # Generate reference results.
        table = act_func(torch.linspace(start=low, end=high, steps=nsamples+1))
        def lookup(l, u):
            assert torch.all(u <= high)
            assert torch.all(l >= low)
            shape = l.size()
            l = l.squeeze()
            u = u.squeeze()
            # select all sample points between l and u.
            low_index = torch.ceil((l - low) / (high - low) * nsamples).int()  # Make sure we do not have index 0.
            high_index = torch.floor((u - low) / (high - low) * nsamples).int()
            real_lb = torch.empty_like(l)
            real_ub = torch.empty_like(u)
            for i, (li, hi) in enumerate(zip(low_index, high_index)):
                if li == hi + 1:
                    # Not enough precision. l and u are too close so we cannot tell.
                    real_lb[i] = float("inf")
                    real_ub[i] = float("-inf")
                else:
                    selected = table[li : hi+1]
                    real_lb[i] = torch.min(selected)
                    real_ub[i] = torch.max(selected)
            real_lb = real_lb.view(*shape)
            real_ub = real_ub.view(*shape)
            return real_lb, real_ub

        # These are reference results. IBP results should be very close to these.
        # Linear bound results can be looser than these.
        ref_forward = model(input_center)
        ref_output_lb, ref_output_ub = lookup(input_lb, input_ub)

        # Get bounding results.
        forward = bounded_model(ptb_data)
        output_lb, output_ub = bounded_model.compute_bounds(
            x=(ptb_data,), method=method)
        bounded_model.set_bound_opts({
            'optimize_bound_args': {'iteration': 2, 'init_alpha': True},
        })

        # Compare.
        assert torch.allclose(forward, ref_forward)
        for i in range(ntests):
            show = False
            if output_ub[0,i] < ref_output_ub[0,i] - 1e-5:
                logger.warning(f'upper bound is wrong {ref_output_ub[0,i] - output_ub[0,i]}')
                show = True
            if output_lb[0,i] > ref_output_lb[0,i] + 1e-5:
                logger.warning(f'lower bound is wrong {output_lb[0,i] - ref_output_lb[0,i]}')
                show = True
            if show:
                logger.warning(f'input_lb={input_lb[0,i]:8.3f}, input_ub={input_ub[0,i]:8.3f}, lb={output_lb[0,i]:8.3f}, ref_lb={ref_output_lb[0,i]:8.3f}, ub={output_ub[0,i]:8.3f}, ref_ub={ref_output_ub[0,i]:8.3f}')
        assert torch.all(output_ub + 1e-5 >= ref_output_ub)
        assert torch.all(output_lb - 1e-5 <= ref_output_lb)

    @pytest.mark.skip(reason="Known issue: https://github.com/Verified-Intelligence/Verifier_Development/issues/164")
    def test_tan(self):
        # Test tan(x) in different periods.
        for i in range(-5, 5):
            self.create_test(
                act_func=torch.tan,
                low=-0.5*torch.pi + i*torch.pi + 1e-20,
                high=0.5*torch.pi + i*torch.pi - 1e-20, method='IBP')
            self.create_test(
                act_func=torch.tan,
                low=-0.5*torch.pi + i*torch.pi + 1e-20,
                high=0.5*torch.pi + i*torch.pi - 1e-20, method='CROWN')

    def test_acts(self):
        for act_func in [torch.nn.functional.relu,
                         torch.sin, torch.cos,
                         torch.tanh, torch.arctan,
                         torch.exp, pow_2, pow_3,
                         torch.sign, GELU]:
            low, high = -10, 10
            if act_func == torch.reciprocal:
                # So far only positive values are supported.
                low = 0.01
            self.create_test(act_func=act_func, low=low, high=high, method='IBP')
            self.create_test(act_func=act_func, low=low, high=high, method='CROWN')
            if act_func not in [torch.exp, torch.sign, torch.sin, torch.cos]:
                # Use optimized bounds
                self.create_test(act_func=act_func, low=low, high=high,
                                 method='CROWN-Optimized')
            if act_func in [torch.sin, torch.cos]:
                test_samples = 10
                for _ in range(test_samples):
                    self.create_test(act_func=act_func, low=low, high=high, method='CROWN-Optimized')

        print('Testing activations with large input range')
        for act_func in [torch.sin, torch.tanh,
                        pow_3, GELU]:
            low, high = -600, 600
            self.create_test(act_func=act_func, low=low, high=high, method='CROWN')


if __name__ == '__main__':
    testcase = Test1DActivation()
    testcase.test_acts()
