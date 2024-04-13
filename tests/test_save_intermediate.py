import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase


class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 500),
            nn.Linear(500, 200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class TestSave(TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test(self, gen_ref=False):
        image = torch.randn(1, 3, 32, 32)
        image = image.to(torch.float32) / 255.0
        model = test_model()

        bounded_model = BoundedModule(
            model, image, bound_opts={
                'optimize_bound_args': {'iteration': 2},
            })

        ptb = PerturbationLpNorm(eps=3/255)
        x = BoundedTensor(image, ptb)
        bounded_model.compute_bounds(x=(x,), method='CROWN-Optimized')
        save_dict = bounded_model.save_intermediate(
            save_path='data/test_save_data' if gen_ref else None)

        if gen_ref:
            torch.save(save_dict, 'data/test_save_data')
            return

        ref_dict = torch.load('data/test_save_data')

        for node in ref_dict.keys():
            assert torch.allclose(ref_dict[node][0], save_dict[node][0], atol=1e-5)
            assert torch.allclose(ref_dict[node][1], save_dict[node][1], atol=1e-5)


if __name__ == '__main__':
    testcase = TestSave()
    testcase.test()
