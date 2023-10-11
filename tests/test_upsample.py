from collections import defaultdict

from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

from testcase import TestCase

class Model(nn.Module):

    def __init__(self,
                 input_dim=5, image_size=4,
                 scale_factor=2, conv_kernel_size=3, stride=1, padding=1,
                 conv_in_channels=16, conv_out_channels=4):
        super(Model, self).__init__()
        self.conv_in_channels = conv_in_channels
        self.input_dim = input_dim
        self.image_size = image_size

        self.fc1 = nn.Linear(input_dim, conv_in_channels * image_size * image_size)
        self.upsample = nn.Upsample(scale_factor=(scale_factor, scale_factor), mode='nearest')
        # H = W = 4 * scale_factor now
        self.conv1 = nn.Conv2d(in_channels=conv_in_channels, out_channels=conv_out_channels,
                               kernel_size=(conv_kernel_size, conv_kernel_size), stride=(stride, stride), padding=padding)
        # H = W = (4 * scale + 2 * pad - ker + s) // s
        size_after_conv = (4 * scale_factor + 2 * padding - conv_kernel_size + stride) // stride
        assert size_after_conv > 0, "0 size after convolution, please use more padding, more scale_factor," \
                                    "smaller kernel, or smaller stride"
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(size_after_conv * size_after_conv * conv_out_channels, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_z):
        f1 = self.fc1(input_z)
        d1 = f1.reshape(-1, self.conv_in_channels, self.image_size, self.image_size)
        d2 = self.upsample(d1)
        d3 = self.conv1(d2)
        d4 = self.relu(d3)
        f2 = self.flatten(d4)
        f3 = self.fc2(f2)
        # out = self.sigmoid(f3)
        return f3

class ModelReducedCGAN(nn.Module):
    def __init__(self):
        """
            The network has the same architecture with merged bn CGAN upsampling one except reduced channel nums
        """
        super(ModelReducedCGAN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.relu5 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.relu6 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(4 * 2 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_z):
        f1 = self.fc1(input_z)
        f2 = f1.reshape(-1, 2, 4, 4)
        f3 = self.up1(f2)
        f4 = self.conv1(f3)
        f5 = self.relu1(f4)
        f6 = self.up2(f5)
        f7 = self.conv2(f6)
        f8 = self.relu2(f7)
        f9 = self.up3(f8)
        f10 = self.conv3(f9)
        f11 = self.relu3(f10)
        f12 = self.conv4(f11)
        f13 = self.conv5(f12)
        f14 = self.relu4(f13)
        f15 = self.conv6(f14)
        f16 = self.relu5(f15)
        f17 = self.conv7(f16)
        f18 = self.relu6(f17)
        f19 = self.conv8(f18)
        f20 = self.relu7(f19)
        f21 = f20.reshape(f20.shape[0], -1)
        f22 = self.fc2(f21)
        # f23 = self.sigmoid(f22)
        return f22



def recursive_allclose(a, b: dict, verbose=False, prefix=''):
    """
        Recursively check whether every corresponding tensors in two dicts are close
    :param a: dict a
    :param b: dict b
    :param prefix: reserved for path tracking in recursive calling for error printing
    :return: bool: all_close or not
    """
    tot_tensor = 0
    tot_dict = 0
    for k in a:
        if isinstance(a[k], torch.Tensor):
            if k == 'unstable_idx': continue
            if verbose:
                print(f'recursive_allclose(): Checking {prefix}{k}')
            assert k in b and isinstance(b[k], torch.Tensor) or isinstance(b[k], Patches), f'recursive_allclose(): Tensor not found in path {prefix}{k}'
            if isinstance(b[k], torch.Tensor):
                assert torch.allclose(a[k].reshape(-1), b[k].reshape(-1), 1e-4, 1e-5), f'recursive_allclose(): Inconsistency found in path {prefix}{k}'
            tot_tensor += 1
        elif isinstance(a[k], dict):
            assert k in b and isinstance(b[k], dict), f'recursive_allclose(): dict not found in path {prefix}{k}'
            recursive_allclose(a[k], b[k], verbose, prefix + k)
            tot_dict += 1
    tot_b_tensor = sum([1 if isinstance(v, torch.Tensor) or isinstance(v, Patches) and k != 'unstable_idx' else 0 for k, v in b.items()])
    tot_b_dict = sum([1 if isinstance(v, dict) else 0 for v in b.values()])
    assert tot_tensor == tot_b_tensor, f'recursive_allclose(): Extra tensors found in path {prefix}'
    assert tot_dict == tot_b_dict, f'recursive_allclose(): Extra recursive paths found in path {prefix}'
    return True


class TestUpSample(TestCase):
    def __init__(self, methodName='runTest', generate=False, device='cpu'):
        super().__init__(methodName, seed=1, ref_path=None, generate=generate)
        self.device = device

    def test(self, seed=123):
        for kernel_size in [3,5]:
            for scaling_factor in [2,3,4]:
                for stride in [1,2]:
                    for padding in [1]:
                        self.test_instance(kernel_size, scaling_factor, stride, padding, seed=seed)

    def test_instance(self, kernel_size=3, scaling_factor=2, stride=1, padding=1, seed=123):
        self.set_seed(seed)

        print(f'kernel_size = {kernel_size}, scaling_factor = {scaling_factor}, stride = {stride}, padding = {padding}')
        random_input = torch.randn((1,5)).to(torch.device(self.device)) * 1000.
        eps = 0.3

        model_ori = Model(scale_factor=scaling_factor,
                          conv_kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        z1_clean = random_input.detach().clone().requires_grad_(requires_grad=True)

        z1 = BoundedTensor(random_input, ptb)
        model_mat = BoundedModule(model_ori, (random_input,), device=self.device,
                                  bound_opts={"conv_mode": "matrix"})
        pred_of_mat = model_mat(z1)
        lb_m, ub_m, A_m = model_mat.compute_bounds(return_A=True, needed_A_dict={model_mat.output_name[0]: model_mat.input_name[0]}, )

        model_pat = BoundedModule(model_ori, (random_input,), device=self.device,
                                  bound_opts={"conv_mode": "patches"})
        pred_of_patch = model_pat(z1)
        lb_p, ub_p, A_p = model_pat.compute_bounds(return_A=True, needed_A_dict={
            model_pat.output_name[0]: model_pat.input_name[0]}, )

        assert torch.allclose(pred_of_mat, pred_of_patch, 1e-5)
        assert torch.allclose(lb_m, lb_p, 1e-5)
        assert torch.allclose(ub_m, ub_p, 1e-5)
        assert recursive_allclose(A_m, A_p, verbose=True)

class TestReducedCGAN(TestCase):

    def __init__(self, methodName='runTest', generate=False, device='cpu'):
        super().__init__(methodName, seed=1, ref_path=None, generate=generate)
        self.device = device

    def test(self, seed=456):
        self.set_seed(seed)
        input = torch.tensor([[0.583, -0.97, -0.97, 0.598, 0.737]]).to(torch.device(self.device))
        eps = 0.1

        model_ori = ModelReducedCGAN()

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        z1_clean = input.detach().clone().requires_grad_(requires_grad=True)

        z1 = BoundedTensor(input, ptb)
        model_mat = BoundedModule(model_ori, (input,), device=self.device,
                                  bound_opts={"conv_mode": "matrix"})
        pred_of_mat = model_mat(z1)

        needed_A_dict = defaultdict(set)
        for node in model_mat.nodes():
            needed_A_dict[node.name] = set()

        lb_m, ub_m, A_m = model_mat.compute_bounds((z1,), return_A=True, needed_A_dict=needed_A_dict, method='crown')

        model_pat = BoundedModule(model_ori, (input,), device=self.device,
                                  bound_opts={"conv_mode": "patches", "sparse_features_alpha": False})
        pred_of_patch = model_pat(z1)
        lb_p, ub_p, A_p = model_pat.compute_bounds((z1,), return_A=True, needed_A_dict=needed_A_dict, method='crown')

        # print(pred_of_mat, pred_of_patch)
        assert torch.allclose(pred_of_mat, pred_of_patch, 1e-5)
        assert torch.allclose(lb_m, lb_p, 1e-5)
        assert torch.allclose(ub_m, ub_p, 1e-5)
        assert recursive_allclose(A_m, A_p, verbose=True)

if __name__ == '__main__':
    # should use device = 'cpu' for GitHub CI
    testcase = TestUpSample(generate=False, device='cpu')
    testcase.test(seed=123)

    # """
    #     following test is much stronger, but runs within 30s only on GPUs
    #     so commented it out for CI testing now
    #     required GPU memory: 1.5 GiB
    # """
    # testhardcase = TestReducedCGAN(generate=False, device='cuda')
    # testhardcase.test(seed=456)


