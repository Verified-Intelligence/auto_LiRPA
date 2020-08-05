# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights   rved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import random
import sys
import shutil
import scipy
import pickle
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from Transformer.modeling import BertForSequenceClassification, BertConfig
from Transformer.utils import convert_examples_to_features
from language_utils import build_vocab
from auto_LiRPA.utils import logger

class Transformer(nn.Module):
    def __init__(self, args, data_train):
        super().__init__()
        self.args = args
        self.max_seq_length = args.max_sent_length
        self.drop_unk = args.drop_unk        
        self.num_labels = args.num_classes
        self.label_list = range(args.num_classes) 
        self.device = args.device
        self.lr = args.lr

        self.dir = args.dir
        self.vocab = build_vocab(data_train, args.min_word_freq)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint = 0
        config = BertConfig(len(self.vocab))
        config.num_hidden_layers = args.num_layers
        config.embedding_size = args.embedding_size
        config.hidden_size = args.hidden_size
        config.intermediate_size = args.intermediate_size
        config.hidden_act = args.hidden_act
        config.num_attention_heads = args.num_attention_heads
        config.layer_norm = args.layer_norm
        config.hidden_dropout_prob = args.dropout
        self.model = BertForSequenceClassification(
            config, self.num_labels, vocab=self.vocab).to(self.device)
        logger.info("Model initialized")
        if args.load:
            checkpoint = torch.load(args.load, map_location=torch.device(self.device))
            epoch = checkpoint['epoch']
            self.model.embeddings.load_state_dict(checkpoint['state_dict_embeddings'])
            self.model.model_from_embeddings.load_state_dict(checkpoint['state_dict_model_from_embeddings'])
            logger.info('Checkpoint loaded: {}'.format(args.load))

        self.model_from_embeddings = self.model.model_from_embeddings
        self.word_embeddings = self.model.embeddings.word_embeddings
        self.model_from_embeddings.device = self.device

    def save(self, epoch):
        self.model.model_from_embeddings = self.model_from_embeddings
        path = os.path.join(self.dir, "ckpt_{}".format(epoch))
        torch.save({ 
            'state_dict_embeddings': self.model.embeddings.state_dict(), 
            'state_dict_model_from_embeddings': self.model.model_from_embeddings.state_dict(), 
            'epoch': epoch
        }, path)
        logger.info("Model saved to {}".format(path))
        
    def build_optimizer(self):
        # update the original model with the converted model
        self.model.model_from_embeddings = self.model_from_embeddings
        param_group = [
            {"params": [p[1] for p in self.model.named_parameters()], "weight_decay": 0.},
        ]    
        return torch.optim.Adam(param_group, lr=self.lr)

    def train(self):
        self.model.train()
        self.model_from_embeddings.train()

    def eval(self):
        self.model.eval() 
        self.model_from_embeddings.eval()

    def get_input(self, batch):
        features = convert_examples_to_features(
            batch, self.label_list, self.max_seq_length, self.vocab, drop_unk=self.drop_unk)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)       
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(self.device)
        tokens = [f.tokens for f in features]

        embeddings, extended_attention_mask = \
            self.model(input_ids, segment_ids, input_mask, embed_only=True)

        return embeddings, extended_attention_mask, tokens, label_ids

    def forward(self, batch):
        embeddings, extended_attention_mask, tokens, label_ids = self.get_input(batch)
        logits = self.model_from_embeddings(embeddings, extended_attention_mask)        
        preds = torch.argmax(logits, dim=1)
        return preds