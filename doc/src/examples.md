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


## Basic Certified Adversarial Defense Training

We provide a [simple example of certified
training](../../examples/vision/simple_training.py). By default it uses
[CROWN-IBP](https://arxiv.org/pdf/1906.06316.pdf) to train a certifiably robust
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

If you find our library useful, please kindly cite our papers:

```
@article{xu2020automatic,
  title={Automatic perturbation analysis for scalable certified robustness and beyond},
  author={Xu, Kaidi and Shi, Zhouxing and Zhang, Huan and Wang, Yihan and Chang, Kai-Wei and Huang, Minlie and Kailkhura, Bhavya and Lin, Xue and Hsieh, Cho-Jui},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{xu2021fast,
    title={{Fast and Complete}: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers},
    author={Kaidi Xu and Huan Zhang and Shiqi Wang and Yihan Wang and Suman Jana and Xue Lin and Cho-Jui Hsieh},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=nVZtXBI6LNn}
}

@article{wang2021beta,
  title={{Beta-CROWN}: Efficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={arXiv preprint arXiv:2103.06624},
  year={2021}
}
```
