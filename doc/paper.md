# Reproducing the Results

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

First, [download data](https://drive.google.com/file/d/12DlaHm1rG0g7M2ITHghb_tZvDg7oSQE5/view?usp=sharing) and extract them to `examples/language/data'.

We can then do training or verification for models. We have experiments for Transformer and LSTM in our paper.

### Transformer

Normal training:

```bash
cd examples/language
python train.py --dir=$DIR --num_epochs=1 --train
```

Robustness verification:

```bash
cd examples/language
python verify.py $DIR
```

Certified robust training with IBP:

```bash
cd examples/language
python train.py --dir=$DIR --num_epochs_warmup=5 --robust --ibp --train # train
python train.py --dir=$DIR --robust --ibp --budget=$BUDGET # test
```

Certified robust training with IBP+backward:

```bash
cd examples/language
python train.py --dir=$DIR --robust --ibp --method=backward --train # training
python train.py --dir=$DIR --robust --ibp --method=backward --budget=$BUDGET # test
```

### LSTM

Normal training:

```bash
cd examples/language
python train.py --dir=$DIR --model=lstm --num_epochs_all_nodes=10 --num_epochs=50 --grad_clip=5.0 --lr=0.001 --train 
```

Certified robust training with IBP:

```bash
cd examples/language
python train.py --dir=$DIR --model=lstm --num_epochs_warmup=10 --num_epochs_all_nodes=10 --num_epochs=50 --grad_clip=5.0 --lr=0.001 --robust --ibp --train # training
python train.py --dir=$DIR --model=lstm --robust --ibp --budget=$BUDGET # test
```

Certified robust training with IBP+backward:

```bash
cd examples/language
python train.py --dir=$DIR --model=lstm --grad_clip=5.0 --lr=0.001 --robust --ibp --method=backward --train # training
python train.py --dir=$DIR --model=lstm --robust --ibp --budget=$BUDGET # test
```

## Vision Models

### MNIST

Certified training with backward mode perturbation analysis for L2 perturbation on weights:

```bash
cd examples/vision
python simple_training_weights_perturbation.py --norm 2 --bound_type CROWN --batch_size 64
```

You can also change the hyperparameters in arguments, like to speed up the training process we can use CROWN-IBP by:
```bash
python simple_training_weights_perturbation.py --norm 2 --bound_type CROWN-IBP --batch_size 256
```

For example, when training with only 10% data on MNIST, we can set:

```bash
python simple_training_weights_perturbation.py --norm 2 --ratio 0.1 --bound_type CROWN --batch_size 64
```
Evaluate the certified cross entropy and test accuracy:

```bash
python simple_training_weights_perturbation.py --load $DIR --norm 2  --bound_type CROWN --batch_size 64 --verify
```

### FashionMNIST
Similarly for FashionMNIST with a different dataset argument:
```bash
cd examples/vision
python simple_training_weights_perturbation.py --data FashionMNIST --norm 2 --bound_type CROWN --batch_size 64 --eps 0.01
```

## References

Jia, R., Raghunathan, A., Göksel, K., and Liang, P. Certified Robustness to Adversarial Word Substitutions. In *Empirical Methods in Natural Language Processing (EMNLP)*, 2019.

Holtzman, A., Buys, J., Forbes, M., Bosselut, A., Golub, D., and Choi, Y. Learning to Write with Cooperative Discriminator. In *Proceedings of the Association for Computational Linguistics*, 2018.