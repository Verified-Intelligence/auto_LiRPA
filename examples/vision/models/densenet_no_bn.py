'''DenseNet in PyTorch.
https://github.com/kuangliu/pytorch-cifar
'''


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv(F.relu(x))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=True)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        # out_planes = int(math.floor(num_planes*reduction))
        # self.trans3 = Transition(num_planes, out_planes)
        # num_planes = out_planes

        # self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        # num_planes += nblocks[3]*growth_rate

        # self.bn = nn.BatchNorm2d(num_planes)
        self.linear1 = nn.Linear(9216, 512)
        self.linear2 = nn.Linear(512, num_classes)


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        # out = self.dense4(out)
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        return out

def Densenet_cifar_wobn(in_ch=3, in_dim=56):
    return DenseNet(Bottleneck, [2,4,6], growth_rate=16)


if __name__ == "__main__":
    net = Densenet_cifar_wobn()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(net)
    print(y)
