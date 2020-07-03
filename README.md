# auto_LiRPA: Automatic Linear Relaxation based Perturbation Analysis for Neural Networks

<p align="center">
<img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa.png" width="50%" height="50%">
</p>

## What's New?

June 30, 2020:
- Adding support to **complex vision models** including DenseNet, ResNeXt and WideResNet.
- **Loss fusion**, a technique that reduces training cost of tight LiRPA bounds 
(e.g. CROWN-IBP) to the same asympototic complexity of IBP, making LiRPA based certified 
defense scalable to large datasets (e.g., TinyImageNet with 200 labels).
- **Multi-GPU** support to scale LiRPA based training to large models and datasets.

## Introduction

**What is auto_LiRPA?** `auto_LiRPA` is a library for automatically deriving
and computing bounds with linear relaxation based perturbation analysis (LiRPA)
(e.g.  [CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)) for
neural networks.  LiRPA algorithms can provide *guaranteed* upper and lower
bounds for a neural network function with perturbed inputs.  These bounds are
represented as linear functions with respect to the variable under
perturbation. LiRPA has become an important tool in robustness verification and
certified adversarial defense, and can become an useful tool for many other
tasks as well.

Our algorithm generalizes existing LiRPA algorithms for
feed-forward neural networks to a graph algorithm on general computational
graphs. We can compute LiRPA bounds on a computational graph defined by
PyTorch, without manual derivation. Our implementation is also automatically
**differentiable**, allowing optimizing network parameters to shape the bounds
into certain specifications (e.g., certified defense).

**Why we need auto_LiRPA?** We aim to facilitate the application of efficient
linear relaxation based perturbation analysis (LiRPA).  Existing works have
extended LiRPA from feed-forward networks to a few more network structures like
ResNet, RNN and Transformer, however they require manual derivation and
implementation of the bounds for each new type of network.  We allow automatic
bound derivation and computation for general computational graphs, in a similar
manner that gradients are obtained in modern deep learning frameworks -- users
only define the computation in a forward pass, and `auto_LiRPA` traverses through
the computational graph and derives bounds for any nodes on the graph.  With
`auto_LiRPA` we free users from deriving and implementing LiPRA for most common
tasks, and they can simply apply LiPRA as a tool for their own applications.
This is especially useful for users who are not experts of LiRPA and cannot
derive these bounds manually (LiRPA is significantly more complicated than
backpropagation).

We provide a wide range of examples of using `auto_LiRPA`.  See [More
Examples](#more-examples) below.  The main algorithm of `auto_LiRPA` is
discussed in [our paper](https://arxiv.org/abs/2002.12920).  Please refer to
the [the guide](doc/paper.md) for reproducing paper results. 

*Automatic Perturbation Analysis on General Computational Graphs*. Kaidi Xu\*,
Zhouxing Shi\*, Huan Zhang\*, Yihan Wang, Minlie Huang, Kai-Wei Chang, Bhavya Kailkhura, Xue
Lin, Cho-Jui Hsieh (\* equal contribution). https://arxiv.org/pdf/2002.12920

Please cite our paper if you use the `auto_LiRPA` library.  If you encounter
any problems with this library, feel free create an issue or pull request. We
welcome contributions in any form from anyone.

## Installation

Before you run any examples, please install `auto_LiRPA` first:

```
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
python setup.py install
```
If you intend to modify this library, use `python setup.py develop` instead).
This library is still under heavy development.  We are still working
on implementing more primitive operations on computational graphs.  These
operations are implemented in `auto_LiRPA/bound_ops.py`.  For example, if you
add a custom activation function that is not supported by our framework, you
can implement it in this file.

## Quick Start

First define your computation as a `nn.Module` and wrap it using
`auto_LiRPA.BoundedModule()`. Then, you can call the `compute_bounds` function
to obtain certified lower and upper bounds under perturbation:

```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Define computation as a nn.Module
class MyModel(nn.Module):
    def forward(self, x):
        # Define your computation here

model = MyModel()
my_input = load_a_batch_of_data()
# Wrap the model with auto_LiRPA
model = BoundedModule(model, my_input)
# Define perturbation
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with perturbation
my_input = BoundedTensor(my_input, ptb)
# Forward propagation using BoundedTensor
prediction = model(my_input)
# Compute LiRPA bounds
lb, ub = model.compute_bounds(method="backward")
```

Checkout
[examples/vision/simple_verification.py](examples/vision/simple_verification.py)
for a complete but very basic example.

## More Examples

We provide many [examples](examples) of using our `auto_LiRPA` library,
including robustness verification and certified robust training for fairly
complicated networks and specifications. Please first install required libraries
to run the examples:

```bash
cd examples
pip install -r requirements.txt
```

### Basic Bound Computation and Verification

We provide a very simple tutorial for `auto_LiRPA` at
[examples/vision/simple_verification.py](examples/vision/simple_verification.py).
This script is self-contained. It loads a simple CNN model and compute the
guaranteed lower and upper bounds using LiRPA for each output neuron under a L
infinity perturbation.

```bash
cd examples/vision
python simple_verification.py
```

### Basic Certified Training

We provide a [simple example of certified
training](examples/vision/simple_training.py). By default it uses
[CROWN-IBP](https://arxiv.org/pdf/1906.06316.pdf) to train a certifiably robust
model:

```bash
cd examples/vision
python simple_training.py
```

The default model is a small ResNet model for MNIST, used in [Scaling provable
adversarial defenses ](https://arxiv.org/pdf/1805.12514.pdf). You should get
less than 10% verified error (at Linf eps=0.3) after training.

For CIFAR-10, we provided some sample models in `examples/vision/models`:
e.g., [cnn_7layer_bn](./examples/vision/models/feedforward.py),
[DenseNet](./examples/vision/models/densenet.py),
[ResNet18](./examples/vision/models/resnet18.py),
[ResNeXt](./examples/vision/models/resnext.py). For example, to train a ResNeXt model on CIFAR,
use:

```bash
python cifar_training.py --batch_size 256 --model ResNeXt_cifar
```

See a list of supported models [here](./examples/vision/models/__init__.py).
This command uses multi-GPUs by default. You probably need to reduce batch size
if you have only 1 GPU. The CIFAR training implementation includes **loss
fusion**, a technique that can greatly reduce training time and memory usage of
LiRPA based certified defense.  More example of CIFAR-10 training can be found
in [doc/paper.md](doc/paper.md).

### Certified Training on Tiny-ImageNet with Loss Fusion

Loss fusion is essential for certified training on Tiny-ImageNet using LiRPA
based bounds (e.g., CROWN-IBP).  This technique leads to ~50X speeding up on
training time and also greatly reduces memory usage.

First, we need to prepare the data:

```bash
cd examples/vision/data/tinyImageNet
bash tinyimagenet_download.sh
```

To train WideResNet model on TinyImagenet:
```bash
cd examples/vision
python tinyimagenet_training.py --batch_size 100 --model wide_resnet_imagenet64
```
To evaluate the clean error and verified error:
```bash
# This is the model saved by the previous command.
MODEL=saved_models/wide_resnet_imagenet64_b100CROWN-IBP_epoch600_start=100,length=400,mid=0.4_ImageNet_0.0039
# Run evaluation.
python tinyimagenet_training.py --verify --model wide_resnet_imagenet64 --load $MODEL --eps 0.003921568627451
```

See more details in [doc/paper.md](doc/paper.md) for these examples.


### Certified Training for LSTM on MNIST

In [examples/sequence](examples/sequence), we have an example of training a
certifiably robust LSTM on MNIST, where an input image is perturbed within an
Lp-ball and sliced to several pieces each regarded as an input frame. To run
the example:

```bash
cd examples/sequence
python train.py
```

### Certified Training for Word Substitution Perturbation on Transformer and LSTM

In [examples/language](examples/language),  we show that our framework can
support perturbation specification of word substitution, beyond Lp-ball
perturbation. We perform certified training for Transformer and LSTM on a
sentiment classification task. 

First, [download data](http://download.huan-zhang.com/datasets/language/data_language.tar.gz) and extract them to `examples/language/data`:

```bash
cd examples/language
wget http://download.huan-zhang.com/datasets/language/data_language.tar.gz
tar xvf data_language.tar.gz
```

We use `$DIR$` to represent the directory for storing checkpoints. Then, to train a robust Transformer:

```bash
python train.py --dir=$DIR --robust --method=IBP+backward_train --train
python train.py --dir=$DIR --robust --method=IBP+backward # for verification
```

And to train a robust LSTM:

```bash
python train.py --dir=$DIR --model=lstm --lr=1e-3 --robust --method=IBP+backward_train --dropout=0.5 --train
python train.py --dir=$DIR --model=lstm --load=$DIR/ckpt_25 --robust --method=IBP+backward # for verification
```

### Certified Training for Weight Perturbation

We provide an example for training a robust network under **weight
perturbations** by applying LiRPA bounds on network weights rather than data
inputs (our algorithm considers general computational graphs, and model weights
are also inputs of a computational graph, so LiRPA bounds can be naturally
applied on weights).  This essentially obtains a network that has "flat"
optimization landscape (a small change in weight parameters do not change loss
too much).

```bash
cd examples/vision
python weights_training.py --norm 2 --bound_type CROWN-IBP
```

### Contribute Additional Examples

If you have an example based on `auto_LiRPA` that can be potentially helpful
for other users, you are encouraged to create a pull request so that we can
include your example here.  Any contributions from the community will be
greatly appreciated.
