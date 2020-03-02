# auto_LiRPA: Automatic Linear Relaxation based Perturbation Analysis for Neural Networks

<p align="center">
<img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa.png" width="50%" height="50%">
</p>

**What is auto_LiRPA?** `auto_LiRPA` is a library for automatically deriving
and computing bounds with linear relaxation based perturbation analysis (LiRPA)
(e.g.  [CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)) for
neural networks.  LiRPA algorithms can provide *guaranteed* upper and lower
bounds for a neural network function with perturbed inputs.  These bounds are
represented as linear functions with respect to the variable under
perturbation. LiRPA has become an important tool in robustness verification and
certified adversarial defense, and we believe that it can become an useful tool
for many other tasks as well.

Our algorithm generalizes existing LiRPA algorithms for
feed-forward neural networks to a graph algorithm on general computational
graphs. We can compute LiRPA bounds on a computational graph defined by
PyTorch, without manual derivation. Our implementation is also automatically
differentiable, allowing optimizing network parameters to reshape the bounds
into certain specifications.

**Why we need auto_LiRPA?** The aim of this library is to facilitate the
application of efficient linear relaxation based perturbation analysis (LiRPA).
Existing works have extended LiRPA from feed-forward networks to a few more
network structures like ResNet, RNN and Transformer, however they require
manual derivation and implementation of the bounds for each new type of
network.  Our framework allows automatic bound derivation and computation for
general computational graphs, in a similar manner that gradients are obtained
in modern deep learning frameworks -- users only define the computation in
forward pass, and `auto_LiRPA` traverses through the computational graph and
derives bounds for any nodes on the graph.  With `auto_LiRPA` we free users
from deriving and implementing LiPRA for most common tasks, and they can simply
apply LiPRA as a tool for their own applications.  This is especially useful
for users who are not experts of LiRPA and cannot derive these bounds manually
(LiRPA is significantly more complicated than backpropagation).

**How to use it?** We provide a wide range of examples of using `auto_LiRPA`.
See [More Examples](#more-examples) below. If you have an example based on
`auto_LiRPA` that can be potentially helpful for other users, you are
encouraged to create a pull request so that we can include your example here.
Any contributions from the community will be greatly appreciated.

The main algorithm of `auto_LiRPA` is discussed in [our
paper](https://arxiv.org/abs/2002.12920).  We demonstrated several complicated
applications including certified training with non-Lp norm perturbations
(synonym perturbations) on Transformer, and training a neural network with
better flatness (less sensitive to weight perturbations).  Please refer to the
[guidance for reproducing paper results](doc/paper.md) for details, and cite
our paper if you find `auto_LiRPA` useful:

*Automatic Perturbation Analysis on General Computational Graphs*. Kaidi Xu\*,
Zhouxing Shi\*, Huan Zhang\*, Minlie Huang, Kai-Wei Chang, Bhavya Kailkhura, Xue
Lin, Cho-Jui Hsieh (\* equal contribution). https://arxiv.org/pdf/2002.12920

This repository is mainly maintained by Kaidi Xu (<xu.kaid@husky.neu.edu>) and
Zhouxing Shi (<zhouxingshichn@gmail.com>). Feel free to contact us or open an
issue if you have problems when using our library.

## Installation

Before you run any examples, please install `auto_LiRPA` first:

```
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
python setup.py install
```

Note that this library is still under heavy development.  We are still working
on implementing more primitive operations on computational graphs.  These
operations are implemented in `auto_LiRPA/bound_ops.py`.  For example, if you
add a custom activation function that is not supported by our framework, you
can implement it in this file.

If you encounter any problems with this library, feel free create an issue or
pull request. We welcome contributions in any form from anyone.

## Quick Start

First define your computation as a `nn.Module` and wrap it using
`auto_LiRPA.BoundGeneral()`. Then, you can call the `compute_bounds` function
to obtain certified lower and upper bounds under perturbation:

```python
# Define computation as a nn.Module
class MyModel(nn.Module):
    def forward(self, x):
        # Define your computation here

model = MyModel()
my_input = load_a_batch_of_data()
# Wrap model with auto_LiRPA
model = auto_LiRPA.BoundGeneral(model, my_input)
# Define perturbation
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Compute LiRPA bounds
lb, ub = model.compute_bounds(ptb=ptb, x=my_input, method="backward")
```

Checkout
[examples/vision/simple_verification.py](examples/vision/simple_verification.py)
for a complete but very basic example.

## More Examples

We provide many [examples](examples) of using our `auto_LiRPA` library,
including robustness verification and certified robust training for fairly
complicated networks and specifications.

### Basic Bound Computation and Verification

We provide a very simple tutorial for `auto_LiRPA` at
[examples/vision/simple_verification.py](examples/vision/simple_verification.py).
This script is self-contained. It loads a simple CNN model and compute the
guaranteed lower and upper bounds using LiRPA for each output neuron under a L
infinity perturbation.

```bash
python -m examples.vision.simple_verification
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

The default model is a small model, and you can get around 10%-11% verified
error (at Linf eps=0.3) after training.

### Certified Training for LSTM on MNIST

In [examples/sequence](examples/sequence), we have an example of training a
certifiably robust LSTM on MNIST, where an input image is perturbed within an
Lp-ball and sliced to several pieces each regarded as an input frame. To run
the example:

```bash
python -m examples.sequence.train
```

### Certified Training for Word Substitution Perturbation on Transformer and LSTM

In [examples/language](examples/language),  we show that our framework can
support perturbation specification of word substitution, beyond Lp-ball
perturbation. We perform certified training for Transformer and LSTM on a
sentiment classification task. 

To train a robust Transformer:

```bash
python -m examples.language.train --model=transformer --robust --ibp --method=backward --train
```

And to train a robust LSTM:

```bash
python -m examples.language.train --model=lstm --grad_clip=5.0 --lr=0.001 --robust --ibp --method=backward --train
```

### Certified Training for Weight Perturbation

We provide an example for training a robust network under weight perturbations
by applying LiRPA bounds on network weights rather than inputs.  This
essentially obtains a network that has "flat" optimization landscape (a small
change in weight parameters do not change loss too much).

```bash
cd examples/vision
python train_general.py --config config/mnist_crown_L2.json --path_prefix saved_models  --model_subset 2
```

See more details in [doc/paper.md](doc/paper.md) for this example.

