import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA.perturbations import *
from test_vision_models import TestVisionModels

class cnn_4layer_test_hardtanh(nn.Module):
    def __init__(self, in_ch, in_dim, width=2, linear_size=256):
        super(cnn_4layer_test_hardtanh, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size)
        self.fc2 = nn.Linear(linear_size, 10)

    def forward(self, x):
        x = F.hardtanh(self.conv1(x))
        x = F.hardtanh(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.hardtanh(self.fc1(x))
        x = self.fc2(x)

        return x

class TestCustomVisionModel(TestVisionModels):
    def __init__(self, methodName='runTest', ref_path='data/vision_clip_test_data', model=cnn_4layer_test_hardtanh(in_ch=1, in_dim=28), generate=False):
        super().__init__(methodName, ref_path, model, generate)

    def test_bounds(self, bound_opts=None, optimize=False):
        if bound_opts is None:
            bound_opts = {'hardtanh': 'same-slope'}
        super().test_bounds(bound_opts=bound_opts, optimize=optimize)

if __name__ == "__main__":
    t = TestCustomVisionModel()
    t.setUp()
    t.test_bounds()
