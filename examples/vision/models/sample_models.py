import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_2layer_MNIST(nn.Module):
    def __init__(self):
        super(cnn_2layer_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class mlp_3layer_MNIST(nn.Module):
    def __init__(self):
        super(mlp_3layer_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class cnn_2layer_CIFAR(nn.Module):
    def __init__(self):
        super(cnn_2layer_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=2)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc(x)
        return x


class cnn_4layer_CIFAR(nn.Module):
    def __init__(self):
        super(cnn_4layer_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1)

        self.fc1 = nn.Linear(10368, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 10368)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


Models = {
    'cnn_2layer_MNIST': cnn_2layer_MNIST,
    'mlp_3layer_MNIST': mlp_3layer_MNIST,
    'cnn_2layer_CIFAR': cnn_2layer_CIFAR,
    'cnn_4layer_CIFAR': cnn_4layer_CIFAR
}

if __name__ == "__main__":
    model = cnn_4layer_CIFAR()
    dummy = torch.randn(8, 3, 32, 32)
    print(model(dummy).shape)
