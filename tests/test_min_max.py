import os
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import *
from testcase import TestCase

class Test_Model(nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1)
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1)
        )

        self.seq3 = nn.Sequential(
            nn.Conv2d(32, 8, 2, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8*4*4,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.seq3(torch.max(self.seq1(x), self.seq2(x)))

class TestMinMax(TestCase):
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName,
            seed=1, ref_path='data/min_max_test_data', generate=generate)

    def test(self):
        for conv_mode in ['patches', 'matrix']:
            model = Test_Model()
            checkpoint = torch.load(
                os.path.join(os.path.dirname(__file__), '../examples/vision/pretrained/test_min_max.pth'),
                map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)

            test_data = torchvision.datasets.MNIST(
                './data', train=False, download=True,
                transform=torchvision.transforms.ToTensor())

            N = 2
            image = test_data.data[:N].view(N,1,28,28)
            image = image.to(torch.float32) / 255.0

            lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device, bound_opts={"conv_mode": conv_mode})

            eps = 0.3
            ptb = PerturbationLpNorm(eps = eps)
            image = BoundedTensor(image, ptb)

            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
            lb, ub = lirpa_model.compute_bounds(x=(image,), method='CROWN-Optimized')

            self.result = (lb, ub)
            self.setUp()
            self.check()

if __name__ == "__main__":
    testcase = TestMinMax(generate=False)
    testcase.test()