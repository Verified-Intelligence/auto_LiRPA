import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import sys
sys.path.append('../examples/vision')
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


def reset_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class cnn_4layer_b(nn.Module):
    def __init__(self, paddingA, paddingB):
        super().__init__()
        self.padA = nn.ZeroPad2d(paddingA)
        self.padB = nn.ZeroPad2d(paddingB)

        self.conv1 = nn.Conv2d(3, 32, (5,5), stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 128, (4,4), stride=2, padding=1)

        self.linear = None
        self.fc = nn.Linear(250, 10)

    def forward(self, x):
        x = self.padA(x)
        x = self.conv1(x)
        x = self.conv2(self.padB(F.relu(x)))
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        if self.linear is None:
            self.linear = nn.Linear(x.size(1), 250)
        x = self.linear(x)
        return self.fc(F.relu(x))


class TestDistinctPatches(TestCase):
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):        
        super().__init__(methodName,
            seed=1234, ref_name='distinct_patches_test_data',
            generate=generate,
            device=device, dtype=dtype)

        self.cases = [(2,1,2,1), (0,0,0,0), (1,3,3,1), (2,2,3,1)]

        # Original: downloads full CIFAR10 dataset (not xdist-safe)
        # test_data = torchvision.datasets.CIFAR10(
        #     "./data", train=False, download=True,
        #     transform=torchvision.transforms.Compose([
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(
        #             mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        #     ]))
        # imgs = torch.from_numpy(test_data.data[:1]).reshape(1,3,32,32).float() / 255.0
        cifar_data = np.load(os.path.join(os.path.dirname(__file__), 'data', 'test_samples', 'cifar10_test_1.npy'))
        imgs = torch.from_numpy(cifar_data[:1]).reshape(1,3,32,32).float() / 255.0
        self.single_img = imgs.to(dtype=self.default_dtype, device=self.default_device)

    def run_conv_mode(self, model, img, conv_mode):
        model(img)  # dummy run to initialize shapes
        model_lirpa = BoundedModule(
            model, img, device=self.default_device,
            bound_opts={"conv_mode": conv_mode}
        )
        ptb = PerturbationLpNorm(norm = np.inf, eps = 0.03)
        img_perturbed = BoundedTensor(img, ptb)

        lb, ub = model_lirpa.compute_bounds(
            x=(img_perturbed,), IBP=False, C=None, method='backward'
        )
        return lb, ub

    def test(self):
        self.result = []
        for paddingA in self.cases:
            for paddingB in self.cases:
                print("Testing", paddingA, paddingB)
                reset_seed()
                model_ori = cnn_4layer_b(paddingA, paddingB).to(
                    device=self.default_device, dtype=self.default_dtype
                )

                lb_patch, ub_patch = self.run_conv_mode(
                    model_ori, self.single_img, conv_mode='patches'
                )
                self.result.append((lb_patch, ub_patch))

                if self.generate:
                    # We only compare with matrix mode when generating reference results
                    lb_matrix, ub_matrix = self.run_conv_mode(
                        model_ori, self.single_img, conv_mode='matrix'
                    )
                    # Check equality
                    assert torch.allclose(lb_patch, lb_matrix), "Lower bounds differ!"
                    assert torch.allclose(ub_patch, ub_matrix), "Upper bounds differ!"
        
        self.check()


if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestDistinctPatches(generate=False)
    testcase.test()