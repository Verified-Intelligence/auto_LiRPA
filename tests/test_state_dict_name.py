import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule
from testcase import TestCase


class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(784, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return x


class cnn_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = BoundedModule(FeatureExtraction(), torch.empty((1, 1, 28, 28)))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


class cnn_MNIST_nobound(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureExtraction()
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


class TestStateDictName(TestCase): 
    def __init__(self, methodName='runTest', generate=False):
        super().__init__(methodName)

    def test(self):
        nobound_model = cnn_MNIST_nobound()

        model = cnn_MNIST()
        state_dict = model.state_dict()
        dummy = torch.randn((1, 1, 28, 28))
        ret1 = model(dummy)

        # create second model and load state_dict to test load_state_dict() whether works proper
        model = cnn_MNIST()
        model.load_state_dict(state_dict, strict=False)
        ret2 = model(dummy)
        self.assertEqual(ret1, ret2)
        self.assertEqual(nobound_model.state_dict().keys(), model.state_dict().keys())

        print('expected', nobound_model.state_dict().keys())
        print('got', model.state_dict().keys())
        

if __name__ == '__main__':
    # Change to generate=True when genearting reference results
    testcase = TestStateDictName(generate=False)
    testcase.test()
