import os
import sys
import torch
import numpy as np
import models
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE
sys.path.append('../examples/vision')



class TestResnetPatches(TestCase):
    def __init__(self, methodName='runTest', generate=False, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName,
            seed=1234, ref_name='resnet_patches_test_data',
            generate=generate,
            device=device, dtype=dtype)

    def test(self):
        model_oris = [
            models.model_resnet(width=1, mult=2),
            models.ResNet18(in_planes=2)
        ]
        self.result = []

        for model_ori in model_oris:
            conv_mode = 'patches' # conv_mode can be set as 'matrix' or 'patches'

            # Original: downloads full CIFAR10 dataset (not xdist-safe)
            # normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            # test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True,
            #                 transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize]))
            # image = torch.Tensor(test_data.data[:N]).reshape(N,3,32,32)
            cifar_data = np.load(os.path.join(os.path.dirname(__file__), 'data', 'test_samples', 'cifar10_test_1.npy'))
            N = 1

            image = torch.Tensor(cifar_data[:N]).reshape(N,3,32,32)
            image = image.to(device=self.default_device,
                             dtype=self.default_dtype) / 255.0

            model_ori = model_ori.to(
                device=self.default_device, dtype=self.default_dtype)
            model = BoundedModule(model_ori, image, bound_opts={"conv_mode": conv_mode}, device=self.default_device)

            ptb = PerturbationLpNorm(norm = np.inf, eps = 0.03)
            image = BoundedTensor(image, ptb)
            pred = model(image)
            lb, ub = model.compute_bounds(IBP=False, C=None, method='backward')
            self.result += [lb, ub]

        self.check()

if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestResnetPatches(generate=False)
    testcase.test()