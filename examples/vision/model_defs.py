

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
from pdb import set_trace as st
import numpy as np 
import math
import collections


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# MLP model, each layer has the same number of neuron
# parameter in_dim: input image dimension, 784 for MNIST and 1024 for CIFAR
# parameter layer: number of layers
# parameter neuron: number of neurons per layer
def model_mlp_uniform(in_dim, layer, neurons, out_dim = 10):
    assert layer >= 2
    neurons = [neurons] * (layer - 1)
    return model_mlp_any(in_dim, neurons, out_dim)

# MLP model, each layer has the different number of neurons
# parameter in_dim: input image dimension, 784 for MNIST and 1024 for CIFAR
# parameter neurons: a list of neurons for each layer
def model_mlp_any(in_dim, neurons, out_dim = 10):
    assert len(neurons) >= 1
    # input layer
    units = [Flatten(), nn.Linear(in_dim, neurons[0])]
    prev = neurons[0]
    # intermediate layers
    for n in neurons[1:]:
        units.append(nn.ReLU())
        units.append(nn.Linear(prev, n))
        prev = n
    # output layer
    units.append(nn.ReLU())
    units.append(nn.Linear(neurons[-1], out_dim))
    #print(units)
    return nn.Sequential(*units)

def model_mlp_after_flatten(in_dim, neurons, out_dim = 10):
    assert len(neurons) >= 1
    # input layer
    units = [nn.Linear(in_dim, neurons[0])]
    prev = neurons[0]
    # intermediate layers
    for n in neurons[1:]:
        units.append(nn.ReLU())
        units.append(nn.Linear(prev, n))
        prev = n
    # output layer
    units.append(nn.ReLU())
    units.append(nn.Linear(neurons[-1], out_dim))
    #print(units)
    return nn.Sequential(*units)

def model_cnn_1layer(in_ch, in_dim, width): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 8*width, 4, stride=4),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),10),
    )
    return model


# CNN, small 2-layer (kernel size fixed to 4) TODO: use other kernel size
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter width: width multiplier
def model_cnn_2layer(in_ch, in_dim, width, linear_size=128): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

# CNN, relatively small 3-layer
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter kernel_size: convolution kernel size, 3 or 5
# parameter width: width multiplier
def model_cnn_3layer(in_ch, in_dim, kernel_size, width):
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, kernel_size=4, stride=4, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*h*h, width*64),
        nn.Linear(width*64, 10)
    )
    return model

def model_cnn_3layer_fixed(in_ch, in_dim, kernel_size, width, linear_size = None):
    if linear_size is None:
        linear_size = width * 64
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, kernel_size=4, stride=4, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*h*h, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

# CNN, relatively large 4-layer
# parameter in_ch: input image channel, 1 for MNIST and 3 for CIFAR
# parameter in_dim: input dimension, 28 for MNIST and 32 for CIFAR
# parameter width: width multiplier
# TODO: the model we used before is equvalent to width=8, TOO LARGE!
# TODO: use different kernel size in this model
def model_cnn_4layer(in_ch, in_dim, width, linear_size): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def model_cnn_10layer(in_ch, in_dim, width): 
    model = nn.Sequential(
        # input 32*32*3
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 32*32*4
        nn.Conv2d(4*width, 8*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 16*16*8
        nn.Conv2d(8*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 16*16*8
        nn.Conv2d(8*width, 16*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 8*8*16
        nn.Conv2d(16*width, 16*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 8*8*16
        nn.Conv2d(16*width, 32*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 4*4*32
        nn.Conv2d(32*width, 32*width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 4*4*32
        nn.Conv2d(32*width, 64*width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 2*2*64
        Flatten(),
        nn.Linear(2*2*64*width,10)
    )
    return model

# below are utilities for feature masking, not used
class FeatureMask2D(nn.Module):
    def __init__(self, in_ch, in_dim, keep = 1.0, seed = 0):
        super(FeatureMask2D, self).__init__()
        self.in_ch = in_ch
        self.in_dim = in_dim
        self.keep = keep
        self.seed = seed
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.weight = torch.rand((1, in_ch, in_dim, in_dim))
        torch.set_rng_state(state)
        self.weight.require_grad = False
        self.weight.data[:] = (self.weight.data <= keep)
    
    # we don't want to register self.weight as a parameter, as it is not trainable
    # but we need to be able to apply operations on it
    def _apply(self, fn):
        super(FeatureMask2D, self)._apply(fn)
        self.weight.data = fn(self.weight.data)

    def forward(self, x):
        return x * self.weight

    def extra_repr(self):
        return 'in_ch={}, in_dim={}, keep={}, seed={}'.format(self.in_ch, self.in_dim, self.keep, self.seed)

def add_feature_subsample(model, in_ch, in_dim, keep = 1.0, seed = 0):
    layers = list(model.children())
    # add a new masking layer
    mask_layer = FeatureMask2D(in_ch, in_dim, keep, seed)
    new_model = model.__class__()
    new_model.add_module("__mask_layer", mask_layer)
    for name, layer in model.named_modules():
        # print(name, layer)
        if name and '.' not in name:
            new_model.add_module(name, layer)
    return new_model

def remove_feature_subsample(model):
    layers = list(model.children())
    # remove the first layer and rebuild
    layers = layers[1:]
    return model.__class__(*layers)





# below are utilities for model converters, not used during training
class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        # super(nn.Conv2d, self).__init__( in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        super(DenseConv2d, self).__init__()
        self.weight = Parameter(torch.randn(out_channels, in_channels//groups, *kernel_size) )

        if bias is not None:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        
      
    def Denseforward(self, inputs):
        b, n, w, h = inputs.shape
        kernel = self.weight
        bias = self.bias
        I = torch.eye(n*w*h).view(n*w*h, n, w, h)
        W = F.conv2d(I, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        input_flat = inputs.view(b, -1)
        b1, n1, w1, h1 = W.shape
        out = torch.matmul(input_flat, W.view(b1, -1)).view(b, n1, w1, h1) 
        new_bias = bias.view(1,n1,1,1).repeat(1,1,w1,h1)


        if type(bias) != type(True): 
            # out2 = out + bias.view(1, n1, 1, 1)
            out2 = out + new_bias
        else:
            out2 = out
        self.dense_w = W.view(b1,-1).transpose(1,0)
        self.dense_bias = new_bias.view(-1)
        # print( ((gt - out2) **2).sum()) 
        # torch.matmul(input_flat, W.view(n*w*h, -1)).view(b, )
        return out2


    def forward(self, input):
        # out = F.conv2d(input, self.weight,self.bias, self.stride,
                        # self.padding, self.dilation, self.groups)
        out = self.Denseforward(input)
        return out

def convert_conv2d_dense(model):
    layers = list(model.children())
    new_model = model.__class__()
    new_layers = []
    # for name, layer in model.named_modules():
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            new_layer = DenseConv2d(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups, bias=layer.bias)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
        else:
            new_layer = layer
        # new_model.add_module(name, new_layer)
        # print(name, layer)
        new_layers.append(new_layer)
    return new_model.__class__(*new_layers)

def save_checkpoint(model, checkpoint_fname):

    layers = list(model.children())
    # for name, layer in model.named_modules():
    count = 0
    save_dict = {}
    for layer in layers:
        if isinstance(layer, DenseConv2d):
            save_dict["{}.weight".format(count+1)] = layer.dense_w
            save_dict["{}.bias".format(count+1)] = layer.dense_bias
        elif isinstance(layer, nn.Linear):
            save_dict["{}.weight".format(count)] = layer.weight
            save_dict["{}.bias".format(count)] = layer.bias
        count+=1
    save_dict = collections.OrderedDict(save_dict)
    torch.save({"state_dict" : save_dict}, checkpoint_fname)
    return save_dict

def load_checkpoint_to_mlpany(dense_checkpoint_file):
    checkpoint = torch.load(dense_checkpoint_file)["state_dict"]
    neurons=[]
    first = True
    for key in checkpoint:
        if key.endswith("weight"):
            h,w = checkpoint[key].shape
            if first:
                neurons.append(w)
                first=False
            print( h, w)
            neurons.append(h)
    print(neurons)
    neuron_list = " ".join([str(n) for n in neurons])
    print("python converter/torch2keras.py -i {} -o {} --flatten {}".format(dense_checkpoint_file, dense_checkpoint_file.replace(".pth", ".h5"), neuron_list))
    # align name 
    model = model_mlp_any(neurons[0], neurons[1:-1], out_dim = neurons[-1])
    mlp_state = model.state_dict()
   
    # for key in mlp_state:
    #     print( mlp_state[key].shape )
    #     print( checkpoint[key].shape)
    model.load_state_dict(checkpoint)
   
    return model






if __name__ == "__main__":
    # model = model_cnn_2layer(3, 32, 1)
    # print(model)
    # sub_model = add_feature_subsample(model, 3, 32, 0.5)
    # print(sub_model)

    checkpoint_fname = "mnist/cnn_2layer_width_1.pth"
    model = model_cnn_2layer(1, 28, 1)
    # print(model)
    input = torch.zeros(1, 1, 28 ,28 )
    x = model(input)    
    model = convert_conv2d_dense(model)
    x2 = model(input)
    save_checkpoint(model, checkpoint_fname.split(".pth")[0] + "_dense.pth")
    print(x2)

    checkpoint_fname =  "mnist/cnn_2layer_width_1_dense.pth"
    checkpoint = torch.load(checkpoint_fname)["state_dict"]
    load_checkpoint_to_mlpany(checkpoint)
    # model_mlp_any(784, neurons, out_dim = 10)
    x3 = model(input)
    print(x3)
