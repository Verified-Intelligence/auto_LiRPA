# Reproducing the Results

In our paper, we demonstrated the applications of our framework on robustness verification and certified training. Please follow this guide to reproduce the results.

## Requirements

Install libraries are needed for running the examples:

```bash
pip install -r examples/requirements.txt
```

## Environment Variables

Here is a list of enrironment variables that need to be set whenever involved:

- `DIR`: the path of the directory to save or load the trained model.
- `BUDGET`: the budget for synonym-based word substitution (for testing certifiably trained language models only, set to 1~6 in our paper).

## Language Models

For experiments on language models, we use a perburbation specification of synonym-based word substitution, where the synonym list of each word in the examples is constructed following Jia et. al, 2019 and their [code](https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689).

Please download the dependencies first:

- `tmp/synonyms.json`: Neighbors of all the words in the counterfitted GloVE space [[Download]](https://worksheets.codalab.org/rest/bundles/0x6ba96d4232c046bc9944965919959d93/contents/blob/)
- `lm/windweller-l2w`: Language model with parameters to screen the neighbor words, by Holtzman et al., 2018 [[Download]](https://worksheets.codalab.org/rest/bundles/0xb4e6f8f93ef04cfebbfa6a466f4ff578/contents/blob/)
- `lm/pytorch-torchfile`: Library to load language model parameters [[Download]](https://worksheets.codalab.org/rest/bundles/0x6ba96d4232c046bc9944965919959d93/contents/blob/)

Next, pre-compute language model scores for neighbor words:

```bash
python -m examples.language.pre_compute_lm_scores.py
```

We can then do training or verification for models. We have experiments for Transformer and LSTM in our paper.

### Transformer

Normal training:

```bash
python -m examples.language.train --dir=$DIR --num_epochs=1 --train
```

Robustness verification:

```bash
python -m examples.language.verify $DIR
```

Certified robust training with IBP:

```bash
python -m examples.language.train --dir=$DIR --num_epochs_warmup=5 --robust --ibp --train # train
python -m examples.language.train --dir=$DIR --robust --ibp --budget=$BUDGET # test
```

Certified robust training with IBP+backward:

```bash
python -m examples.language.train --dir=$DIR --robust --ibp --method=backward --train # training
python -m examples.language.train --dir=$DIR --robust --ibp --method=backward --budget=$BUDGET # test
```

### LSTM

Normal training:

```bash
python -m examples.language.train --dir=$DIR --model=lstm --num_epochs_all_nodes=10 --num_epochs=50 --grad_clip=5.0 --lr=0.001 --train 
```

Certified robust training with IBP:

```bash
python -m examples.language.train --dir=$DIR --model=lstm --num_epochs_warmup=10 --num_epochs_all_nodes=10 --num_epochs=50 --grad_clip=5.0 --lr=0.001 --robust --ibp --train # training
python -m examples.language.train --dir=$DIR --model=lstm --robust --ibp --budget=$BUDGET # test
```

Certified robust training with IBP+backward:

```bash
python -m examples.language.train --dir=$DIR --model=lstm --grad_clip=5.0 --lr=0.001 --robust --ibp --method=backward --train # training
python -m examples.language.train --dir=$DIR --model=lstm --robust --ibp --budget=$BUDGET # test
```

## Vision Models

### MNIST

Certified training with backward mode perturbation analysis for L2 perturbation on weights:

```bash
cd examples/vision
python train_general.py --config config/mnist_crown_L2.json --path_prefix $DIR  --model_subset 2
```

You can also change the hyperparameters in `config/mnist_crown_L2.json`.

For example, when training with only 10% data on MNIST, we can set `"training_params:loader_params:ratio=0.1"`in the .json file or just run:

```bash
python train_general.py "training_params:loader_params:ratio=0.1" --config config/fashion-mnist_crown.json --path_prefix $DIR  --model_subset 2
```

Evaluate the certified cross entropy and test accuracy:

```bash
python eval_general.py --config config/mnist_crown_L2.json --path_prefix $DIR  --model_subset 2
```

### FashionMNIST

Similarly for FashionMNIST with a different .json file:

```bash
cd examples/vision
python train_general.py --config config/fashion-mnist_crown.json --path_prefix $DIR  --model_subset 2
```

## References

Jia, R., Raghunathan, A., Göksel, K., and Liang, P. Certified Robustness to Adversarial Word Substitutions. In *Empirical Methods in Natural Language Processing (EMNLP)*, 2019.

Holtzman, A., Buys, J., Forbes, M., Bosselut, A., Golub, D., and Choi, Y. Learning to Write with Cooperative Discriminator. In *Proceedings of the Association for Computational Linguistics*, 2018.