"""Test optimized bounds in simple_verification."""
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from testcase import TestCase

# This simple model comes from https://github.com/locuslab/convex_adversarial
def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

class TestSimpleVerification(TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test(self):
      model = mnist_model()
      checkpoint = torch.load(
        '../examples/vision/pretrained/mnist_a_adv.pth',
        map_location=torch.device('cpu'))
      model.load_state_dict(checkpoint)

      test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=torchvision.transforms.ToTensor())
      N = 2
      image = test_data.data[:N].view(N,1,28,28)
      image = image.to(torch.float32) / 255.0
      if torch.cuda.is_available():
          image = image.cuda()
          model = model.cuda()

      lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
      ptb = PerturbationLpNorm(0.3)
      image = BoundedTensor(image, ptb)

      method = 'CROWN-Optimized (alpha-CROWN)'
      lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
      _, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
      self.assertEqual(ub[0][7], torch.tensor(12.5080))

if __name__ == '__main__':
    testcase = TestSimpleVerification()
    testcase.test()
