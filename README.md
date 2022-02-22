# auto_LiRPA: Automatic Linear Relaxation based Perturbation Analysis for Neural Networks

![](https://travis-ci.com/KaidiXu/auto_LiRPA.svg?token=HM3jb55xV1sMRsVKBr8b&branch=master&status=started)
[![Documentation Status](https://readthedocs.org/projects/auto-lirpa/badge/?version=latest)](https://auto-lirpa.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://PaperCode.cc/AutoLiRPA-Demo)
[![Video Introduction](https://img.shields.io/badge/play-video-red.svg)](http://PaperCode.cc/AutoLiRPA-Video)
[![BSD license](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<p align="center">
<a href="http://PaperCode.cc/AutoLiRPA-Video"><img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa_2.png" width="45%" height="45%" float="left"></a>
<a href="http://PaperCode.cc/AutoLiRPA-Video"><img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa_1.png" width="45%" height="45%" float="right"></a>
</p>

## What's New?

- Our neural network verification tool [α,β-CROWN](https://github.com/huanzhang12/alpha-beta-CROWN.git) ([alpha-beta-CROWN](https://github.com/huanzhang12/alpha-beta-CROWN.git)) **won** [VNN-COMP 2021](https://sites.google.com/view/vnn2021) **with the highest total score**, outperforming 11 SOTA verifiers. α,β-CROWN uses the `auto_LiRPA` library as its core bound computation library.
- Support for [custom operators](https://auto-lirpa.readthedocs.io/en/latest/custom_op.html). (01/02/2022) 
- [Optimized CROWN/LiRPA](https://arxiv.org/pdf/2011.13824.pdf) bound (α-CROWN) for ReLU, **sigmoid**, **tanh**, and **maxpool** activation functions, which can significantly outperform regular CROWN bounds. See [simple_verification.py](examples/vision/simple_verification.py#L59) for an example. (07/31/2021)
- Handle split constraints for ReLU neurons ([β-CROWN](https://arxiv.org/pdf/2103.06624.pdf)) for complete verifiers. (07/31/2021)
- A memory efficient GPU implementation of backward (CROWN) bounds for 
convolutional layers. (10/31/2020)
- Certified defense models for downscaled ImageNet, TinyImageNet, CIFAR-10, LSTM/Transformer. (08/20/2020)
- Adding support to **complex vision models** including DenseNet, ResNeXt and WideResNet. (06/30/2020)
- **Loss fusion**, a technique that reduces training cost of tight LiRPA bounds 
(e.g. CROWN-IBP) to the same asympototic complexity of IBP, making LiRPA based certified 
defense scalable to large datasets (e.g., TinyImageNet, downscaled ImageNet). (06/30/2020)
- **Multi-GPU** support to scale LiRPA based training to large models and datasets. (06/30/2020)
- Initial release. (02/28/2020)

## Introduction

`auto_LiRPA` is a library for automatically deriving and computing bounds with
linear relaxation based perturbation analysis (LiRPA) (e.g.
[CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)) for
neural networks, which is an useful tool for formal robustness verification. We
generalize existing LiRPA algorithms for feed-forward neural networks to a
graph algorithm on general computational graphs, defined by PyTorch.
Additionally, our implementation is also automatically **differentiable**,
allowing optimizing network parameters to shape the bounds into certain
specifications (e.g., certified defense). You can find [a video ▶️ introduction
here](http://PaperCode.cc/AutoLiRPA-Video).

Our library supports the following algorithms:

* Backward mode LiRPA bound propagation ([CROWN](https://arxiv.org/pdf/1811.00866.pdf)/[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf))
* Backward mode LiRPA bound propagation with optimized bounds ([α-CROWN](https://arxiv.org/pdf/2011.13824.pdf))
* Backward mode LiRPA bound propagation with split constraints ([β-CROWN](https://arxiv.org/pdf/2103.06624.pdf))
* Forward mode LiRPA bound propagation ([Xu et al., 2020](https://arxiv.org/pdf/2002.12920))
* Forward mode LiRPA bound propagation with optimized bounds (similar to [α-CROWN](https://arxiv.org/pdf/2011.13824.pdf))
* Interval bound propagation ([IBP](https://arxiv.org/pdf/1810.12715.pdf))
* Hybrid approaches, e.g., Forward+Backward, IBP+Backward ([CROWN-IBP](https://arxiv.org/pdf/1906.06316.pdf)), [α,β-CROWN](https://github.com/huanzhang12/alpha-beta-CROWN.git) ([alpha-beta-CROWN](https://github.com/huanzhang12/alpha-beta-CROWN.git))

Our library allows automatic bound derivation and computation for general
computational graphs, in a similar manner that gradients are obtained in modern
deep learning frameworks -- users only define the computation in a forward
pass, and `auto_LiRPA` traverses through the computational graph and derives
bounds for any nodes on the graph.  With `auto_LiRPA` we free users from
deriving and implementing LiPRA for most common tasks, and they can simply
apply LiPRA as a tool for their own applications.  This is especially useful
for users who are not experts of LiRPA and cannot derive these bounds manually
(LiRPA is significantly more complicated than backpropagation).

## Technical Background in 1 Minute

Deep learning frameworks such as PyTorch represent neural networks (NN) as
a computational graph, where each mathematical operation is a node and edges
define the flow of computation:

<p align="center">
<a href="http://PaperCode.cc/AutoLiRPA-Video"><img src="http://www.huan-zhang.com/images/upload/lirpa/auto_LiRPA_background_1.png" width="80%"></a>
</p>

Normally, the inputs of a computation graph (which defines a NN) are data and
model weights, and PyTorch goes through the graph and produces model prediction
(a bunch of numbers):

<p align="center">
<a href="http://PaperCode.cc/AutoLiRPA-Video"><img src="http://www.huan-zhang.com/images/upload/lirpa/auto_LiRPA_background_2.png" width="80%"></a>
</p>

Our `auto_LiRPA` library conducts perturbation analysis on a computational
graph, where the input data and model weights are defined within some
user-defined ranges.  We get guaranteed output ranges (bounds):

<p align="center">
<a href="http://PaperCode.cc/AutoLiRPA-Video"><img src="http://www.huan-zhang.com/images/upload/lirpa/auto_LiRPA_background_3.png" width="80%"></a>
</p>

## Installation

Python 3.7+ is required. Pytorch 1.8 (LTS) is recommended, although a newer
version might also work. It is highly recommended to have a pre-installed PyTorch
that matches your system and our version requirement. See [PyTorch Get Started](https://pytorch.org/get-started). 
Then you can install `auto_LiRPA` via:

```bash
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
python setup.py install
```

If you intend to modify this library, use `python setup.py develop` instead.

## Quick Start

First define your computation as a `nn.Module` and wrap it using
`auto_LiRPA.BoundedModule()`. Then, you can call the `compute_bounds` function
to obtain certified lower and upper bounds under input perturbations:

```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Define computation as a nn.Module.
class MyModel(nn.Module):
    def forward(self, x):
        # Define your computation here.

model = MyModel()
my_input = load_a_batch_of_data()
# Wrap the model with auto_LiRPA.
model = BoundedModule(model, my_input)
# Define perturbation. Here we add Linf perturbation to input data.
ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
# Make the input a BoundedTensor with the pre-defined perturbation.
my_input = BoundedTensor(my_input, ptb)
# Regular forward propagation using BoundedTensor works as usual.
prediction = model(my_input)
# Compute LiRPA bounds using the backward mode bound propagation (CROWN).
lb, ub = model.compute_bounds(x=(my_input,), method="CROWN")
```

Checkout
[examples/vision/simple_verification.py](examples/vision/simple_verification.py)
for a complete but very basic example.

<a href="http://PaperCode.cc/AutoLiRPA-Demo"><img align="left" width=64 height=64 src="https://colab.research.google.com/img/colab_favicon_256px.png"></a>
We also provide a [Google Colab Demo](http://PaperCode.cc/AutoLiRPA-Demo) including an example of computing verification
bounds for a 18-layer ResNet model on CIFAR-10 dataset. Once the ResNet model
is defined as usual in Pytorch, obtaining provable output bounds is as easy as
obtaining gradients through autodiff. Bounds are efficiently computed on GPUs.

## More Working Examples

We provide [a wide range of examples](doc/src/examples.md) of using `auto_LiRPA`: 

* [Basic Bound Computation and **Robustness Verification** of Neural Networks](doc/src/examples.md#basic-bound-computation-and-robustness-verification-of-neural-networks)
* [Basic **Certified Adversarial Defense** Training](doc/src/examples.md#basic-certified-adversarial-defense-training)
* [Large-scale Certified Defense Training on **ImageNet**](doc/src/examples.md#certified-adversarial-defense-on-downscaled-imagenet-and-tinyimagenet-with-loss-fusion)
* [Certified Adversarial Defense Training on Sequence Data with **LSTM**](doc/src/examples.md#certified-adversarial-defense-training-for-lstm-on-mnist)
* [Certifiably Robust Language Classifier using **Transformers**](doc/src/examples.md#certifiably-robust-language-classifier-with-transformer-and-lstm)
* [Certified Robustness against **Model Weight Perturbations**](doc/src/examples.md#certified-robustness-against-model-weight-perturbations-and-certified-defense)

## Full Documentations

For more documentations, please refer to:

* [Documentation homepage](https://auto-lirpa.readthedocs.io)
* [API documentation](https://auto-lirpa.readthedocs.io/en/latest/api.html)
* [Adding custom operators](https://auto-lirpa.readthedocs.io/en/latest/custom_op.html)
* [Guide](https://auto-lirpa.readthedocs.io/en/latest/paper.html) for reproducing [our NeurIPS 2020 paper](https://arxiv.org/abs/2002.12920)

## Publications

Please kindly cite our papers if you use the `auto_LiRPA` library. Full [BibTeX entries](doc/examples.md#bibtex-entries) can be found [here](doc/examples.md#bibtex-entries).

The general LiRPA based bound propagation algorithm was originally proposed in our paper:

* [Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond](https://arxiv.org/pdf/2002.12920).  
NeurIPS 2020  
Kaidi Xu\*, Zhouxing Shi\*, Huan Zhang\*, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh (\* Equal contribution)

The `auto_LiRPA` library is further extended to allow optimized bound (α-CROWN) and split constraints (β-CROWN):

* [Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers](https://arxiv.org/pdf/2011.13824.pdf).  
ICLR 2021  
Kaidi Xu\*, Huan Zhang\*, Shiqi Wang, Yihan Wang, Suman Jana, Xue Lin and Cho-Jui Hsieh (\* Equal contribution)

* [Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Verification](https://arxiv.org/pdf/2103.06624.pdf).  
NeurIPS 2021  
Shiqi Wang\*, Huan Zhang\*, Kaidi Xu\*, Suman Jana, Xue Lin, Cho-Jui Hsieh and Zico Kolter (\* Equal contribution)


## Developers and Copyright

| [Kaidi Xu](https://kaidixu.com/) | [Zhouxing Shi](https://shizhouxing.github.io/) | [Huan Zhang](https://huan-zhang.com/) | [Yihan Wang](https://yihanwang617.github.io/) | [Shiqi Wang](https://www.cs.columbia.edu/~tcwangshiqi/) |
|:--:|:--:| :--:| :--:| :--:|
| <img src="https://kaidixu.files.wordpress.com/2020/07/profile2-1.jpg" width="125" /> | <img src="https://shizhouxing.github.io/photo.jpg" width="115" /> | <img src="https://huan-zhang.appspot.com/images/Huan_Zhang_photo.jpg" width="125" /> | <img src="https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png" width="125" height="125" /> | <img src="https://www.cs.columbia.edu/~tcwangshiqi/images/shiqiwang.jpg" width="125" /> |

* Kaidi Xu (xu.kaid@northeastern.edu): main developer
* Zhouxing Shi (zshi@cs.ucla.edu): main developer
* Huan Zhang (huan@huan-zhang.com): team lead
* Yihan Wang (yihanwang@ucla.edu)
* Shiqi Wang (sw3215@columbia.edu): contact for beta-CROWN

We thank [commits](https://github.com/KaidiXu/auto_LiRPA/commits) and [pull requests](https://github.com/KaidiXu/auto_LiRPA/pulls) from community contributors.

Our library is released under the BSD 3-Clause license.
