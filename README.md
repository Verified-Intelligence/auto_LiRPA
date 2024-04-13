# auto_LiRPA: Automatic Linear Relaxation based Perturbation Analysis for Neural Networks

[![Documentation Status](https://readthedocs.org/projects/auto-lirpa/badge/?version=latest)](https://auto-lirpa.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://PaperCode.cc/AutoLiRPA-Demo)
[![Video Introduction](https://img.shields.io/badge/play-video-red.svg)](http://PaperCode.cc/AutoLiRPA-Video)
[![BSD license](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<p align="center">
<a href="http://PaperCode.cc/AutoLiRPA-Video"><img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa_2.png" width="45%" height="45%" float="left"></a>
<a href="http://PaperCode.cc/AutoLiRPA-Video"><img src="http://www.huan-zhang.com/images/upload/lirpa/auto_lirpa_1.png" width="45%" height="45%" float="right"></a>
</p>

## What's New?

- The [INVPROP algorithm](https://arxiv.org/pdf/2302.01404.pdf) allows to compute overapproximationsw of preimages (the set of inputs of an NN generating a given output set) and tighten bounds using output constraints. (03/2024)
- New activation functions (sin, cos, tan, GeLU) with optimizable bounds (α-CROWN) and [branch and bound support](https://files.sri.inf.ethz.ch/wfvml23/papers/paper_24.pdf) for non-ReLU activation functions. We achieve significant improvements on verifying neural networks with non-ReLU activation functions such as Transformer and LSTM networks. (09/2023)
- [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) ([alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git)) (using `auto_LiRPA` as its core library) **won** [VNN-COMP 2023](https://sites.google.com/view/vnn2023). (08/2023)
- Bound computation for higher-order computational graphs to support bounding Jacobian, Jacobian-vector products, and [local Lipschitz constants](https://arxiv.org/abs/2210.07394). (11/2022)
- Our neural network verification tool [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) ([alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git)) (using `auto_LiRPA` as its core library) **won** [VNN-COMP 2022](https://sites.google.com/view/vnn2022). Our library supports the large CIFAR100, TinyImageNet and ImageNet models in VNN-COMP 2022. (09/2022)
- Implementation of **general cutting planes** ([GCP-CROWN](https://arxiv.org/pdf/2208.05740.pdf)), support of more activation functions and improved performance and scalability. (09/2022)
- Our neural network verification tool [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) ([alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git)) **won** [VNN-COMP 2021](https://sites.google.com/view/vnn2021) **with the highest total score**, outperforming 11 SOTA verifiers. α,β-CROWN uses the `auto_LiRPA` library as its core bound computation library. (09/2021)
- [Optimized CROWN/LiRPA](https://arxiv.org/pdf/2011.13824.pdf) bound (α-CROWN) for ReLU, **sigmoid**, **tanh**, and **maxpool** activation functions, which can significantly outperform regular CROWN bounds. See [simple_verification.py](examples/vision/simple_verification.py#L59) for an example. (07/31/2021)
- Handle split constraints for ReLU neurons ([β-CROWN](https://arxiv.org/pdf/2103.06624.pdf)) for complete verifiers. (07/31/2021)
- A memory efficient GPU implementation of backward (CROWN) bounds for
convolutional layers. (10/31/2020)
- Certified defense models for downscaled ImageNet, TinyImageNet, CIFAR-10, LSTM/Transformer. (08/20/2020)
- Adding support to **complex vision models** including DenseNet, ResNeXt and WideResNet. (06/30/2020)
- **Loss fusion**, a technique that reduces training cost of tight LiRPA bounds
(e.g. CROWN-IBP) to the same asymptotic complexity of IBP, making LiRPA based certified
defense scalable to large datasets (e.g., TinyImageNet, downscaled ImageNet). (06/30/2020)
- **Multi-GPU** support to scale LiRPA based training to large models and datasets. (06/30/2020)
- Initial release. (02/28/2020)

## Introduction

`auto_LiRPA` is a library for automatically deriving and computing bounds with
linear relaxation based perturbation analysis (LiRPA) (e.g.
[CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)) for
neural networks, which is a useful tool for formal robustness verification. We
generalize existing LiRPA algorithms for feed-forward neural networks to a
graph algorithm on general computational graphs, defined by PyTorch.
Additionally, our implementation is also automatically **differentiable**,
allowing optimizing network parameters to shape the bounds into certain
specifications (e.g., certified defense). You can find [a video ▶️ introduction
here](http://PaperCode.cc/AutoLiRPA-Video).

Our library supports the following algorithms:

* Backward mode LiRPA bound propagation ([CROWN](https://arxiv.org/pdf/1811.00866.pdf)/[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf))
* Backward mode LiRPA bound propagation with optimized bounds ([α-CROWN](https://arxiv.org/pdf/2011.13824.pdf))
* Backward mode LiRPA bound propagation with split constraints ([β-CROWN](https://arxiv.org/pdf/2103.06624.pdf)) for ReLU, and ([Shi et al. 2023](https://files.sri.inf.ethz.ch/wfvml23/papers/paper_24.pdf)) for general nonlinear functions
* Generalized backward mode LiRPA bound propagation with general cutting plane constraints ([GCP-CROWN](https://arxiv.org/pdf/2208.05740.pdf))
* Backward mode LiRPA bound propagation with bounds tightened using output constraints ([INVPROP](https://arxiv.org/pdf/2302.01404.pdf))
* Forward mode LiRPA bound propagation ([Xu et al., 2020](https://arxiv.org/pdf/2002.12920))
* Forward mode LiRPA bound propagation with optimized bounds (similar to [α-CROWN](https://arxiv.org/pdf/2011.13824.pdf))
* Interval bound propagation ([IBP](https://arxiv.org/pdf/1810.12715.pdf))
* Hybrid approaches, e.g., Forward+Backward, IBP+Backward ([CROWN-IBP](https://arxiv.org/pdf/1906.06316.pdf)), [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) ([alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git))
* MIP/LP formulation of neural networks

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

Python 3.7+ and PyTorch 1.11+ are required.
It is highly recommended to have a pre-installed PyTorch
that matches your system and our version requirement.
See [PyTorch Get Started](https://pytorch.org/get-started).
Then you can install `auto_LiRPA` via:

```bash
git clone https://github.com/Verified-Intelligence/auto_LiRPA
cd auto_LiRPA
pip install .
```

If you intend to modify this library, use `pip install -e .` instead.

Optionally, you may build and install native CUDA modules (CUDA toolkit required):
```bash
python auto_LiRPA/cuda_utils.py install
```

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
lb, ub = model.compute_bounds(x=(my_input,), method="backward")
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

* [Basic Bound Computation on a Toy Neural Network (simplest example)](examples/simple/toy.py)
* [Basic Bound Computation with **Robustness Verification** of Neural Networks as an example](doc/src/examples.md#basic-bound-computation-and-robustness-verification-of-neural-networks)
* [MIP/LP Formulation of Neural Networks](examples/simple/mip_lp_solver.py)
* [Basic **Certified Adversarial Defense** Training](doc/src/examples.md#basic-certified-adversarial-defense-training)
* [Large-scale Certified Defense Training on **ImageNet**](doc/src/examples.md#certified-adversarial-defense-on-downscaled-imagenet-and-tinyimagenet-with-loss-fusion)
* [Certified Adversarial Defense Training on Sequence Data with **LSTM**](doc/src/examples.md#certified-adversarial-defense-training-for-lstm-on-mnist)
* [Certifiably Robust Language Classifier using **Transformers**](doc/src/examples.md#certifiably-robust-language-classifier-with-transformer-and-lstm)
* [Certified Robustness against **Model Weight Perturbations**](doc/src/examples.md#certified-robustness-against-model-weight-perturbations-and-certified-defense)
* [Bounding **Jacobian** and **local Lipschitz constants**](examples/vision/jacobian_new.py)
* [Compute an Overapproximate of Neural Network **Preimage**](examples/simple/invprop.py)

`auto_LiRPA` has also been used in the following works:
* [**α,β-CROWN for complete neural network verification**](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
* [**Fast certified robust training**](https://github.com/shizhouxing/Fast-Certified-Robust-Training)
* [**Computing local Lipschitz constants**](https://github.com/shizhouxing/Local-Lipschitz-Constants)

## Full Documentations

For more documentations, please refer to:

* [Documentation homepage](https://auto-lirpa.readthedocs.io)
* [API documentation](https://auto-lirpa.readthedocs.io/en/latest/api.html)
* [Adding custom operators](https://auto-lirpa.readthedocs.io/en/latest/custom_op.html)
* [Guide](https://auto-lirpa.readthedocs.io/en/latest/paper.html) for reproducing [our NeurIPS 2020 paper](https://arxiv.org/abs/2002.12920)

## Publications

Please kindly cite our papers if you use the `auto_LiRPA` library. Full [BibTeX entries](doc/src/examples.md#bibtex-entries) can be found [here](doc/src/examples.md#bibtex-entries).

The general LiRPA based bound propagation algorithm was originally proposed in our paper:

* [Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond](https://arxiv.org/pdf/2002.12920).
NeurIPS 2020
Kaidi Xu\*, Zhouxing Shi\*, Huan Zhang\*, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh (\* Equal contribution)

The `auto_LiRPA` library is further extended to allow optimized bound (α-CROWN), split constraints (β-CROWN) general constraints (GCP-CROWN), and higher-order computational graphs:

* [Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers](https://arxiv.org/pdf/2011.13824.pdf).
ICLR 2021.
Kaidi Xu\*, Huan Zhang\*, Shiqi Wang, Yihan Wang, Suman Jana, Xue Lin and Cho-Jui Hsieh (\* Equal contribution).

* [Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Verification](https://arxiv.org/pdf/2103.06624.pdf).
NeurIPS 2021.
Shiqi Wang\*, Huan Zhang\*, Kaidi Xu\*, Suman Jana, Xue Lin, Cho-Jui Hsieh and Zico Kolter (\* Equal contribution).

* [GCP-CROWN: General Cutting Planes for Bound-Propagation-Based Neural Network Verification](https://arxiv.org/abs/2208.05740).
Huan Zhang\*, Shiqi Wang\*, Kaidi Xu\*, Linyi Li, Bo Li, Suman Jana, Cho-Jui Hsieh and Zico Kolter (\* Equal contribution).

* [Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation](https://arxiv.org/abs/2210.07394).
NeurIPS 2022.
Zhouxing Shi, Yihan Wang, Huan Zhang, Zico Kolter, Cho-Jui Hsieh.

Certified robust training using `auto_LiRPA` is improved to allow much shorter warmup and faster training:
* [Fast Certified Robust Training with Short Warmup](https://arxiv.org/pdf/2103.17268.pdf).
NeurIPS 2021.
Zhouxing Shi\*, Yihan Wang\*, Huan Zhang, Jinfeng Yi and Cho-Jui Hsieh (\* Equal contribution).

Branch and bound for non-ReLU and general activation functions:
* [Formal Verification for Neural Networks with General Nonlinearities via Branch-and-Bound](https://files.sri.inf.ethz.ch/wfvml23/papers/paper_24.pdf).
Zhouxing Shi\*, Qirui Jin\*, Zico Kolter, Suman Jana, Cho-Jui Hsieh, Huan Zhang (\* Equal contribution).

Tightening of bounds and preimage computation using the INVPROP algorithm:
* [Provably Bounding Neural Network Preimages](https://arxiv.org/pdf/2302.01404.pdf).
Suhas Kotha\*, Christopher Brix\*, Zico Kolter, Krishnamurthy (Dj) Dvijotham\*\*, Huan Zhang\*\* (\* Equal contribution; \*\* Equal advising).

## Developers and Copyright

Team lead:
* Huan Zhang (huan@huan-zhang.com), UIUC

Current developers:
* Zhouxing Shi (zshi@cs.ucla.edu), UCLA
* Christopher Brix (brix@cs.rwth-aachen.de), RWTH Aachen University
* Kaidi Xu (kx46@drexel.edu), Drexel University
* Xiangru Zhong (xiangruzh0915@gmail.com), Sun Yat-sen University
* Qirui Jin (qiruijin@umich.edu), University of Michigan
* Hao Chen (haoc8@illinois.edu), UIUC
* Hongji Xu (hx84@duke.edu), Duke University


Past developers:
* Linyi Li (linyi2@illinois.edu), UIUC
* Zhuolin Yang (zhuolin5@illinois.edu), UIUC
* Zhuowen Yuan (realzhuowen@gmail.com), UIUC
* Shiqi Wang (sw3215@columbia.edu), Columbia University
* Yihan Wang (yihanwang@ucla.edu), UCLA
* Jinqi (Kathryn) Chen (jinqic@cs.cmu.edu), CMU

We thank the [commits](https://github.com/Verified-Intelligence/auto_LiRPA/commits) and [pull requests](https://github.com/Verified-Intelligence/auto_LiRPA/pulls) from community contributors.

Our library is released under the BSD 3-Clause license.
