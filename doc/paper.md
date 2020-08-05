# Reproducing the Results on Our Paper

In our paper, we demonstrated the applications of our framework on robustness verification and certified training. Please follow this guide to reproduce the results.

## Requirements

Install libraries are needed for running the examples:

```bash
cd examples/language
pip install -r requirements.txt
```

## Environment Variables

Here is a list of enrironment variables that need to be set whenever involved:

- `DIR`: the path of the directory to save or load the trained model.
- `BUDGET`: the budget for synonym-based word substitution (for testing certifiably trained language models only, set to 1~6 in our paper).

## Language Models

First, [download data](https://drive.google.com/file/d/12DlaHm1rG0g7M2ITHghb_tZvDg7oSQE5/view?usp=sharing) and extract them to `examples/language/data`.

We can then do training or verification for models. We have experiments for Transformer and LSTM in our paper.

### LSTM

Regular training:

```bash
python train.py --dir=$DIR --num_epochs=10 --model=lstm --lr=1e-3 --dropout=0.5 --train
python train.py --dir=$DIR --model=lstm --load=$DIR/ckpt_10 --robust --method=IBP|IBP+backward|forward|forward+backward # for verification
```

IBP training:

```bash
python train.py --dir=$DIR --model=lstm --lr=1e-3 --robust --method=IBP --dropout=0.5 --train
python train.py --dir=$DIR --model=lstm --load=$DIR/ckpt_25 --robust --method=IBP # for verification
```

LiRPA training:

```bash
python train.py --dir=$DIR --model=lstm --lr=1e-3 --robust --method=IBP+backward_train --dropout=0.5 --train
python train.py --dir=$DIR --model=lstm --load=$DIR/ckpt_25 --robust --method=IBP+backward # for verification
```

### Transformer

Regular training:

```bash
python train.py --dir=$DIR --num_epochs=2 --train
python train.py --dir=$DIR --robust --method=IBP|IBP+backward|forward|forward+backward # for verification
```

IBP training:

```bash
python train.py --dir=$DIR --robust --method=IBP --train
python train.py --dir=$DIR --robust --method=IBP # for verification
```

LiRPA training:

```bash
python train.py --dir=$DIR --robust --method=IBP+backward_train --train
python train.py --dir=$DIR --robust --method=IBP+backward # for verification
```

### Other options

You may add `—budget OTHER_BUDGET` to set a different budget for word substitution.





## Vision Models

### CIFAR-10
For CIFAR-10, we provided some sample models in `examples/vision/models`:

[cnn_7layer_bn](../examples/vision/models/feedforward.py), 
[DenseNet](../examples/vision/models/densenet.py), 
[ResNet18](../examples/vision/models/resnet18.py), 
[ResNeXt](../examples/vision/models/resnext.py).



To reproduce our state-of-the-art results CNN-7+BN model, just run:

```bash
cd examples/vision
python cifar_training.py --batch_size 256 --lr_decay_milestones 1400 1700 --model cnn_7layer_bn
```

Or you can change the model to ResNeXt like:

```bash
python cifar_training.py --batch_size 256 --lr_decay_milestones 1400 1700 --model ResNeXt_cifar
```

To evaluate the clean error and verified error of our CNN-7+BN model:
```bash
python cifar_training.py --verify  --model cnn_7layer_bn --load saved_models/cnn_7layer_bn_cifar --eps 0.03137254901961
```

Or you can evaluate your models by specific $DIR  in --load

In case there is a need to train the model without loss fusion (would be slower noticebly),  please add --no_loss_fusion flag:
```bash
python cifar_training.py --batch_size 256 --lr_decay_milestones 1400 1700 --model cnn_7layer_bn --no_loss_fusion
```

### Tiny-ImageNet

First, we need to prepare the data:
```bash
cd examples/vision/data/tinyImageNet
bash tinyimagenet_download.sh
```

To reproduce our results on WideResNet model, just run:
```bash
cd examples/vision
python tinyimagenet_training.py --batch_size 100 --lr_decay_milestones 600 700 --model wide_resnet_imagenet64
```
To evaluate the clean error and verified error:
```bash
python tinyimagenet_training.py --verify  --model wide_resnet_imagenet64 --load $DIR --eps 0.003921568627451
```

### MNIST

Certified training with backward mode perturbation analysis for L2 perturbation on weights:

```bash
cd examples/vision
python weights_training.py --norm 2 --bound_type CROWN-IBP --lr_decay_milestones 120 140
```

To reproduce the model with  "flat" optimization landscape, we only 10% data on MNIST, we can set:

```bash
python weights_training.py --norm 2 --ratio 0.1 --bound_type CROWN-IBP --batch_size 500 --lr_decay_milestones 3700 4000 --scheduler_opts start=200,length=3200 --opt SGD --lr 0.1
```
Evaluate the certified cross entropy and test accuracy:

```bash
python weights_training.py --load $DIR --norm 2  --bound_type CROWN-IBP --batch_size 500 --verify
```

### FashionMNIST
Similarly for FashionMNIST with a different dataset argument:
```bash
cd examples/vision
python weights_training.py --data FashionMNIST --norm 2 --ratio 0.1 --bound_type CROWN-IBP --batch_size 500 --lr_decay_milestones 3700 4000 --scheduler_opts start=200,length=3200 --opt SGD --lr 0.1 --eps 0.05
```

### Scalability

We provide multi-GPU training and **Loss Fusion** in our framework to improve scalability.

All experiments on Transformer/LSTM and other vision models on MNIST dataset
can be conducted on a single Nvidia GTX 1080Ti GPU.
 
Certified training on CIFAR-10 dataset with **Loss Fusion** can be conducted
on two Nvidia GTX 1080Ti GPUs with batch size = 256.  By contrast, the batch
size can only be set to 64 (or lower) without **Loss Fusion**.
 
Certified training on Tiny-ImageNet dataset can be conducted on four Nvidia GTX
1080TI GPUs only with **Loss Fusion**. The batch size can be  set as 100~256
depends on the model.

