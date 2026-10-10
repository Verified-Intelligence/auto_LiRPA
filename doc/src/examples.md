# Examples

We provide many [examples](../../examples) of using our `auto_LiRPA` library,
including robustness verification and certified robust training for fairly
complicated networks and specifications. Please first install required libraries
to run the examples:

```bash
cd examples
pip install -r requirements.txt
```

## Basic Bound Computation and Robustness Verification of Neural Networks

We provide a very simple tutorial for `auto_LiRPA` at
[examples/vision/simple_verification.py](../../examples/vision/simple_verification.py).
This script is self-contained. It loads a simple CNN model and compute the
guaranteed lower and upper bounds using LiRPA for each output neuron under a L
infinity perturbation.

```bash
cd examples/vision
python simple_verification.py
```

In this example, we compute lower and upper bounds of neural network outputs
under input perturbations with a few different methods: IBP, CROWN-IBP,
CROWN (backward) and optimized-CROWN (α-CROWN). For the adversarially trained
network in this demonstration, IBP usually provides a very loose bound. CROWN
can provide a reasonably tight bound almost instantly, and the tightest bounds
can be obtained using α-CROWN within a few seconds.


## Optimizing and Reading a Shared Affine Relaxation

[examples/simple/dynamic_forward.py](../../examples/simple/dynamic_forward.py)
constructs a small ReLU network with explicit weights and one input box. It
runs on CPU, with no model or dataset downloads. From the repository root:

```bash
python examples/simple/dynamic_forward.py
```

### The relaxation and the optimization objective

Dynamic forward represents a relaxed network over shared bounded variables:

```text
y_relaxed = centre + G e,    e in [-1, 1]^n.
```

The input box contributes normalized input variables. Each unstable ReLU
contributes an error variable: on its fixed pre-activation interval `[l, u]`,

```text
relu(z) = s z + h (1 + e_k),
h = max(-s l, (1 - s) u) / 2,    0 <= s <= 1.
```

For every concrete network input in the box, some assignment of the bounded
variables reproduces the network's values. The relaxation also includes
assignments that do not correspond to concrete inputs. Stable ReLUs are exact.

`method='dynamic-forward'` uses the default slope `u / (u - l)` on unstable
ReLUs. `method='dynamic-forward-optimized'` uses the native alpha-CROWN
optimizer to choose one slope per ReLU neuron, shared across specifications,
backward start nodes, and lower and upper bound sides. With specification
matrix `C` shaped `[1, queries, output_elements]`, it maximizes the smallest
query lower bound:

```text
maximize over shared slopes: min_q (C centre - row_sum(abs(C G)))[q].
```

Omitting `C` uses the identity specification, one query per output element.
The returned upper bounds use the same chosen slopes; they are not optimized
in a separate upper-bound search. Ordinary alpha-CROWN can use different
parameters for different specifications, start nodes, and bound sides. Sharing
the slopes instead provides one consistent affine relaxation whose output and
intermediate rows can be consumed together. It is a different optimization
constraint, not a guarantee of tighter bounds than ordinary alpha-CROWN.
Improving the worst lower bound can loosen an upper bound or another query's
lower bound.

### Calling the method and reading generators

The example first computes pre-activation bounds with CROWN and passes them
as `interm_bounds` to both methods, so the comparison holds those intervals
fixed. Providing `interm_bounds` is optional: the optimized method computes
missing intervals before searching, then holds them fixed. Search settings
come from the existing `optimize_bound_args`, for example:

```python
bm.set_bound_opts({'optimize_bound_args': {
    'iteration': 40, 'lr_alpha': 0.05, 'early_stop_patience': 40}})
lb, ub, forms = bm.compute_bounds(
    x=(bounded_x,), C=C, method='dynamic-forward-optimized',
    interm_bounds=interm_bounds, return_generators=True,
    needed_generators_dict={bm.final_name: None, pre.name: unstable})
G, centre = forms[bm.final_name]
```

The search uses the configured optimizer, iteration count, learning rate,
learning-rate decay, patience, and stopping criterion. The method sets
network-wide sharing, the worst-query objective, fixed intermediate bounds,
and coherent best-iteration selection internally. In particular,
`loss_reduction_func`, `keep_best`, and `reproduce_alpha_bounds` follow that
objective rather than their caller-provided values.

The dictionary maps each requested node name to flattened row indices, or
`None` for every row. The example requests all output rows and the first
ReLU's pre-activation rows at its unstable neurons. Requests must name the
bound target or its ancestors. Omitting `needed_generators_dict` requests all
rows of the target; `{}` requests no entries. With `return_generators=False`,
the method returns only `(lb, ub)` and ignores the generator dictionary.
Generator requests use `return_generators`, rather than `return_A`.

Each entry is `(G, centre)` with shapes `[1, selected_rows, n]` and
`[1, selected_rows]`. `C` shapes the returned bounds, while the generator rows
describe the requested node itself. The example checks this relation:

```python
query_G = C @ G
query_centre = (C @ centre.unsqueeze(-1)).squeeze(-1)
radius = query_G.abs().sum(-1)
torch.testing.assert_close(lb, query_centre - radius)
torch.testing.assert_close(ub, query_centre + radius)
```

Column `j` denotes the same variable in every returned entry. An earlier
node's generators can have fewer columns because later ReLU variables do not
affect it; missing trailing columns are zero. When generators are requested,
`bm[bm.input_name[0]].linear.dynamic_variable_ranges` maps node names to their
`(start, count)` column blocks. Input variables precede ReLU error variables;
sparse initialization can retain zero padding columns. The selected slopes
are available as `bm.dynamic_forward_slopes` after the call; another
`compute_bounds` call resets that attribute.

### Supported scope and state

The optimized method supports one input box (`batch_size=1`) with an explicit
`PerturbationLpNorm(norm=float('inf'), x_L=..., x_U=...)` and ReLU relaxations.
Generator extraction also requires the graph's perturbed operators to support
dynamic forward. Reverse extraction uses a zero-column dynamic-forward pass
for centres and variable numbering, followed by CROWN traversals for rows;
it does not extend the supported operator set. Unsupported cases raise
`NotImplementedError`; a generator request outside the target's ancestor
subgraph raises `ValueError`.

`bound_opts['dynamic_forward']['generators']` selects `'forward'`, `'reverse'`,
or `'auto'`. The default `'auto'` chooses a direction based on the numbers of
requested rows and variables. Both directions read the same relaxation.
Requesting only one bound side returns `None` for the other side and does not
change which generator entries are returned. An omitted `x` reuses the input
from the previous call.

The search uses fixed intermediate bounds and excludes beta, cuts,
output-constraint refinement, and mixed-precision optimization. Existing
alpha/beta parameters and optimization options are preserved; inputs and
propagation caches follow the normal `compute_bounds` lifecycle. Results
follow the caller's grad mode. Their gradients hold the chosen slopes and
intermediate bounds fixed, while gradients through the input box,
specification matrix, and model parameters remain available. Gradients do not
differentiate through the slope search. The example uses `torch.no_grad()`
because it only reads bounds and generators.


## Basic Certified Adversarial Defense Training

We provide a [simple example of certified training](../../examples/vision/simple_training.py).
By default it uses [CROWN-IBP](https://arxiv.org/pdf/1906.06316.pdf) to train a certifiably robust
model:

```bash
cd examples/vision
python simple_training.py
```

The default model is a small ResNet model for MNIST, used in [Wong et al. 2018](https://arxiv.org/pdf/1805.12514.pdf). You should get
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
e.g., [cnn_7layer_bn](../../examples/vision/models/feedforward.py),
[DenseNet](../../examples/vision/models/densenet.py),
[ResNet18](../../examples/vision/models/resnet18.py),
[ResNeXt](../../examples/vision/models/resnext.py). For example, to train a ResNeXt model on CIFAR,
use:

```bash
python cifar_training.py --batch_size 256 --model ResNeXt_cifar
```

See a list of supported models [here](../../examples/vision/models/__init__.py).
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
in [doc/paper.md](paper.md).


## Certified Adversarial Defense on Downscaled ImageNet and TinyImageNet with Loss Fusion

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
[Image-Net.org](http://image-net.org/download-images), under the "Download downsampled image data (32x32, 64x64)" section, to `example/vision/data/ImageNet64/raw_data`,
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

See more details in [paper.md](paper.md) for these examples.


## Certified Adversarial Defense Training for LSTM on MNIST

In [examples/sequence](../../examples/sequence), we have an example of training a
certifiably robust LSTM on MNIST, where an input image is perturbed within an
Lp-ball and sliced to several pieces each regarded as an input frame. To run
the example:

```bash
cd examples/sequence
python train.py
```

## Certifiably Robust Language Classifier with Transformer and LSTM

In [examples/language](../../examples/language),  we show that our framework can
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

## Certified Robustness against Model Weight Perturbations and Certified Defense

In our paper ([Xu et al. 2020](https://arxiv.org/pdf/2002.12920)), we provide
an example for training a robust network under **weight perturbations** by
applying LiRPA bounds on network weights rather than data inputs.  Importantly,
because our algorithm considers general computational graphs, and model weights
are also inputs of a computational graph, LiRPA bounds can be naturally applied
on model weights, immediately enabling **robustness certification and certified
defense against model weight perturbations**. This also allows us to obtain a
network that has a "flat" optimization landscape (a small change in weight
parameters does not change the loss too much).

The run robustness verification and certified defense for model weight
perturbations, run the following code example:

```bash
cd examples/vision
python weight_perturbation_training.py --norm 2 --bound_type CROWN-IBP
```

By default it uses the [CROWN-IBP](https://arxiv.org/pdf/1906.06316.pdf) bound
to efficiently bound model outputs under model weight perturbations, and the
training process is reasonably fast. Other bound types such as the full
backward (CROWN) bounds are also supported for weight perturbations.

## Contribute Additional Examples

If you have an example based on `auto_LiRPA` that can be potentially helpful
for other users, you are encouraged to create a pull request so that we can
include your example here.  Any contributions from the community will be
greatly appreciated.

## BibTeX Entries

If you find our library useful, please kindly cite [our papers](../../README.md#publications):

The original auto_LiRPA paper:
```bibtex
@article{xu2020automatic,
  title={Automatic perturbation analysis for scalable certified robustness and beyond},
  author={Xu, Kaidi and Shi, Zhouxing and Zhang, Huan and Wang, Yihan and Chang, Kai-Wei and Huang, Minlie and Kailkhura, Bhavya and Lin, Xue and Hsieh, Cho-Jui},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

Extensions:
```bibtex
@inproceedings{xu2021fast,
  title={{Fast and Complete}: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers},
  author={Kaidi Xu and Huan Zhang and Shiqi Wang and Yihan Wang and Suman Jana and Xue Lin and Cho-Jui Hsieh},
  booktitle={International Conference on Learning Representations},
  year={2021},
}

@article{wang2021beta,
  title={{Beta-CROWN}: Efficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={arXiv preprint arXiv:2103.06624},
  year={2021}
}

@article{zhang2022general,
  title={General Cutting Planes for Bound-Propagation-Based Neural Network Verification},
  author={Zhang, Huan and Wang, Shiqi and Xu, Kaidi and Li, Linyi and Li, Bo and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{shi2022efficiently,
  title={Efficiently computing local lipschitz constants of neural networks via bound propagation},
  author={Shi, Zhouxing and Wang, Yihan and Zhang, Huan and Kolter, J Zico and Hsieh, Cho-Jui},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={2350--2364},
  year={2022}
}

@inproceedings{shi2024genbab,
  title={Neural Network Verification with Branch-and-Bound for General Nonlinearities},
  author={Shi, Zhouxing and Jin, Qirui and Kolter, Zico and Jana, Suman and Hsieh, Cho-Jui and Zhang, Huan},
  booktitle={International Conference on Tools and Algorithms for the Construction and Analysis of Systems},
  year={2025}
}

@inproceedings{kotha2023provably,
  author={Kotha, Suhas and Brix, Christopher and Kolter, J. Zico and Dvijotham, Krishnamurthy and Zhang, Huan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={80270--80290},
  title={Provably Bounding Neural Network Preimages},
  volume={36},
  year={2023}
}
```
