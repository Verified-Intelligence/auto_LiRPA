# auto_LiRPA: Automatic Linear Relaxation based Perturbation Analysis for Neural Networks

![](https://travis-ci.com/KaidiXu/auto_LiRPA.svg?token=HM3jb55xV1sMRsVKBr8b&branch=master&status=started)

<p align="center">
<img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa_2.png" width="45%" height="45%" float="left">
<img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa.png" width="45%" height="45%" float="right">
</p>

## What's New?

- A **memory efficient** GPU implementation of backward (CROWN) bounds for 
**convolutional layers**. See [examples/vision/patch_convolution.py](examples/vision/patch_convolution.py)
for a comparison. (10/31/2020)
- We released our certified defense models for downscaled
[ImageNet](#imagenet-pretrained), [TinyImageNet](#imagenet-pretrained), [CIFAR-10](#cifar10-pretrained),
and [LSTM/Transformers](#language-pretrained). (08/20/2020)
- Adding support to **complex vision models** including DenseNet, ResNeXt and WideResNet. (06/30/2020)
- **Loss fusion**, a technique that reduces training cost of tight LiRPA bounds 
(e.g. CROWN-IBP) to the same asympototic complexity of IBP, making LiRPA based certified 
defense scalable to large datasets (e.g., TinyImageNet, downscaled ImageNet). (06/30/2020)
- **Multi-GPU** support to scale LiRPA based training to large models and datasets. (06/30/2020)
- Initial release. (02/28/2020)

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

**Supported Features:** We support **backward/forward mode perturbation analysis** 
and interval bound propagation (**IBP**, which can be seen as a degenerate case 
of LiRPA) on general computational graphs, as well as hybrid approaches such as 
IBP+Backward (CROWN-IBP), Forward+Backward.

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
discussed in [our NeurIPS 2020 paper](https://arxiv.org/abs/2002.12920).  Please refer to
the [the guide](doc/paper.md) for reproducing paper results. 

[Automatic Perturbation Analysis for Scalable Certified Robustness and
Beyond](https://arxiv.org/pdf/2002.12920). Kaidi Xu\*, Zhouxing Shi\*, Huan
Zhang\*, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin,
Cho-Jui Hsieh (\* equal contribution). NeurIPS 2020.

Please cite our paper if you use the `auto_LiRPA` library.  If you encounter
any problems with this library, feel free create an issue or pull request. We
welcome contributions in any form from anyone.

## Installation

Python 3.7+ is required. Pytorch 1.4, 1.5 and 1.6 are supported.
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
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds
lb, ub = model.compute_bounds(x=(my_input,), method="backward")
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

We also provide an L0-norm option in `simple_training.py` and an example to use
L0-norm certified training to train an MLP model.  The IBP bounds for L0-norm
is provided in [Chiang et
al.](https://openreview.net/forum?id=HyeaSkrYPH&noteId=HyeaSkrYPH), but here we
also use the tighter backward mode perturbation analysis for L0-norm which is
the first time in literature.

```bash
cd examples/vision
python simple_training.py --model mlp_3layer --norm 0 --eps 1
```

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
LiRPA based certified defense.

<a id="cifar10-pretrained"></a>
**Pretrained models for CIFAR-10:** We released our CIFAR-10 certified defense models
[here](http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/cifar/).  To compute
verified error, please run:

```bash
python cifar_training.py --verify  --model cnn_7layer_bn --load saved_models/cnn_7layer_bn_cifar --eps 0.03137254901961
```

More example of CIFAR-10 training can be found
in [doc/paper.md](doc/paper.md).


### Certified Training on Downscaled ImageNet and TinyImageNet with Loss Fusion

Loss fusion is essential for certified training on Tiny-ImageNet (200 classes)
or downscaled ImageNet (1000 classes) using LiRPA based bounds (e.g.,
CROWN-IBP).  This technique leads to ~50X speeding up on training time and also
greatly reduces memory usage.

First, we need to prepare the data, for Tiny-ImageNet:

```bash
cd examples/vision/data/tinyImageNet
bash tinyimagenet_download.sh
```

To train the WideResNet model on Tiny-Imagenet:

```bash
cd examples/vision
python tinyimagenet_training.py --batch_size 100 --model wide_resnet_imagenet64
```

For downscaled ImageNet, please download raw images (Train and Val, 64x64, npz format) from 
[Image-Net.org](http://image-net.org/download-images) to `example/vision/data/ImageNet64/raw_data`,
decompress them and then run data preprocessing:

```bash
cd examples/vision/data/ImageNet64
python imagenet_data_loader.py
```

To train the WideResNet model on downscaled Imagenet:

```bash
cd examples/vision
python imagenet_training.py --batch_size 100 --model wide_resnet_imagenet64_1000class
```

<a id="imagenet-pretrained"></a> 
**Pretrained models for ImageNet:** We released our certified defense models (trained with loss fusion) for
[Tiny-Imagenet](http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/imagenet-200/)
and
[downscaled Imagenet](http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/imagenet-1000/).
To evaluate the clean error and verified error:

```bash
# This is the model saved path.
MODEL=saved_models/wide_resnet_imagenet64_1000	
# Run evaluation.
python imagenet_training.py --verify --model wide_resnet_imagenet64_1000class --load $MODEL --eps 0.003921568627451
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

We use `$DIR` to represent the directory for storing checkpoints. Then, to train a robust Transformer:

```bash
python train.py --dir=$DIR --robust --method=IBP+backward_train --train
python train.py --load=$DIR/ckpt_25 --robust --method=IBP+backward # for verification
```

And to train a robust LSTM:

```bash
python train.py --dir=$DIR --model=lstm --lr=1e-3 --robust --method=IBP+backward_train --dropout=0.5 --train
python train.py --model=lstm --load=$DIR/ckpt_25 --robust --method=IBP+backward # for verification
```

<a id="language-pretrained"></a> 
**Pretrained models for Transformer/LSTM:** We provide our certified defense models for
[Transformer](http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_transformer)
and [LSTM](http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_lstm). 
To directly evaluate them:

```bash
# Download and evaluate our trained Transformer
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_transformer
python train.py --load=ckpt_transformer --robust --method=IBP+backward
# Download and evaluate our trained LSTM
wget http://web.cs.ucla.edu/~zshi/files/auto_LiRPA/trained/ckpt_lstm
python train.py --model=lstm --load=ckpt_lstm --robust --method=IBP+backward
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
