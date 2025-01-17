import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase


class cnn_model(nn.Module):
    def __init__(self, layers, padding, stride, linear):
        super(cnn_model, self).__init__()
        self.module_list = []
        channel = 1
        length = 28
        for i in range(layers):
            self.module_list.append(nn.Conv2d(channel, 3, 4, stride = stride, padding = padding))
            channel = 3
            length = (length + 2 * padding - 4)//stride + 1
            assert length > 0
            self.module_list.append(nn.ReLU())
        self.module_list.append(nn.Flatten())
        if linear:
            self.module_list.append(nn.Linear(3 * length * length, 256))
            self.module_list.append(nn.Linear(256, 10))
        self.model = nn.Sequential(*self.module_list)

    def forward(self, x):
        x = self.model(x)
        return x

class TestConv(TestCase):
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName,
            seed=1, ref_path=None,
            generate=generate)

    def test(self):
        models = [1, 2, 3]
        paddings = [1, 2]
        strides = [1, 3]

        N = 2
        n_classes = 10
        image = torch.randn(N, 1, 28, 28)
        image = image.to(torch.float32) / 255.0

        for layer_num in models:
            for padding in paddings:
                for stride in strides:
                    for linear in [True, False]:
                        model_ori = cnn_model(layer_num, padding, stride, linear)
                        print('Model:', model_ori)

                        model = BoundedModule(model_ori, image, bound_opts={"conv_mode": "patches"})
                        eps = 0.3
                        ptb = PerturbationLpNorm(x_L=image-eps, x_U=image+eps)
                        image = BoundedTensor(image, ptb)
                        pred = model(image)
                        lb, ub = model.compute_bounds()

                        model = BoundedModule(model_ori, image, bound_opts={"conv_mode": "matrix"})
                        pred = model(image)
                        lb_ref, ub_ref = model.compute_bounds()

                        if linear:
                            assert lb.shape == ub.shape == torch.Size((N, n_classes))
                        self.assertEqual(lb, lb_ref)
                        self.assertEqual(ub, ub_ref)

                        if not linear and layer_num == 1:
                            pred = model(image)
                            lb_forward, ub_forward = model.compute_bounds(method='forward')
                            self.assertEqual(lb, lb_forward)
                            self.assertEqual(ub, ub_forward)
                            pred = model(image)
                            lb_forward, ub_forward = model.compute_bounds(method='dynamic-forward+backward')
                            self.assertEqual(lb, lb_forward)
                            self.assertEqual(ub, ub_forward)

if __name__ == '__main__':
    testcase = TestConv()
    testcase.test()
