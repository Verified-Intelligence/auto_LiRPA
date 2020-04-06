# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

from pytorch_pretrained_bert.modeling import ACT2FN, BertConfig, BertIntermediate, \
    BertSelfAttention, BertPreTrainedModel

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertLayerNormNoVar(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNormNoVar, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = x - u
        return self.weight * x + self.bias       

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, glove=None, vocab=None):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.config = config
        
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # position/token_type embedding disabled
        # embeddings = words_embeddings + position_embeddings + token_type_embeddings
        
        embeddings = words_embeddings
        return embeddings

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if hasattr(config, "layer_norm") and config.layer_norm == "no_var":
            self.LayerNorm = BertLayerNormNoVar(config.hidden_size, eps=1e-12)    
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if hidden_states.shape[-1] == input_tensor.shape[-1]:
            hidden_states = hidden_states + input_tensor    
        if hasattr(self.config, "layer_norm") and self.config.layer_norm == "no":
            pass
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, input_size):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)

        return attention_output

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        if hasattr(config, "layer_norm") and config.layer_norm == "no_var":
            self.LayerNorm = BertLayerNormNoVar(config.hidden_size, eps=1e-12)    
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if hasattr(self.config, "layer_norm") and self.config.layer_norm == "no":
            pass
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config, layer_id):
        super(BertLayer, self).__init__()
        self.input_size = config.hidden_size
        self.attention = BertAttention(config, self.input_size)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModelFromEmbeddings(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelFromEmbeddings, self).__init__(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, embeddings, extended_attention_mask):
        encoded_layers  = self.encoder(embeddings, extended_attention_mask)    
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return pooled_output

class BertForSequenceClassificationFromEmbeddings(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassificationFromEmbeddings, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModelFromEmbeddings(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.linear_in = nn.Linear(config.embedding_size, config.hidden_size)

        self.layer_norm = config.layer_norm
        if hasattr(config, "layer_norm") and config.layer_norm == "no_var":
            self.LayerNorm = BertLayerNormNoVar(config.embedding_size, eps=1e-12)    
        else:
            self.LayerNorm = BertLayerNorm(config.embedding_size, eps=1e-12)

        self.apply(self.init_bert_weights)

    def forward(self, embeddings, extended_attention_mask):
        embeddings = self.linear_in(embeddings)

        if self.layer_norm == "no":
            pass
        else:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
            
        pooled_output = self.bert(embeddings, extended_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, glove=None, vocab=None):
        super(BertForSequenceClassification, self).__init__(config)
        self.model_from_embeddings = BertForSequenceClassificationFromEmbeddings(
            config, num_labels
        )
        self.num_labels = num_labels
        self.embeddings = BertEmbeddings(config, glove=glove, vocab=vocab)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, embed_only=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embeddings = self.embeddings(input_ids, token_type_ids)
        if embed_only:
            return embeddings, extended_attention_mask
        logits = self.model_from_embeddings(embeddings, extended_attention_mask)
        return logits
