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

from pytorch_pretrained_bert.modeling import BertConfig, logger,\
    ACT2FN, PRETRAINED_MODEL_ARCHIVE_MAP, BERT_CONFIG_NAME, TF_WEIGHTS_NAME, BertPreTrainedModel

# try:
#     raise ImportError # DEBUG
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# except ImportError:
#     logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
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
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNormNoVar, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = x - u
        # s = (x - u).pow(2).mean(-1, keepdim=True)
        # x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias       

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        if not hasattr(config, "embedding_size"):
            config.embedding_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.config = config

        if hasattr(config, "layer_norm") and config.layer_norm == "no_var":
            self.LayerNorm = BertLayerNormNoVar(config.embedding_size, eps=1e-12)    
        else:
            self.LayerNorm = BertLayerNorm(config.embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if hasattr(self.config, "layer_norm") and self.config.layer_norm == "no":
            pass
        else:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config, input_size):
        super(BertSelfAttention, self).__init__()
        if input_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (input_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(input_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # DEBUG
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask 
 
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config, input_size):
        super(BertSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(input_size, config.hidden_size)
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
        self.self = BertSelfAttention(config, input_size)
        self.output = BertSelfOutput(config, input_size)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_scores, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)

        return attention_output, self_output, attention_scores, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


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
        if not hasattr(config, "embedding_size"):
            config.embedding_size = config.hidden_size
        self.input_size = config.embedding_size if layer_id == 0 else config.hidden_size
        self.attention = BertAttention(config, self.input_size)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, self_output, attention_scores, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, self_output, attention_scores, attention_probs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_self_output = [] # right after summation weighted by softmax probs
        all_attention_scores = []
        all_attention_probs = []
        all_attention_output = []
        for layer_module in self.layer:
            hidden_states, self_output, attention_scores, attention_probs = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_self_output.append(self_output)
                all_attention_scores.append(attention_scores)
                all_attention_probs.append(attention_probs)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_self_output.append(self_output)
            all_attention_scores.append(attention_scores)
            all_attention_probs.append(attention_probs)
        return all_encoder_layers, all_self_output, all_attention_scores, all_attention_probs

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
        encoded_layers, self_output, attention_scores, attention_probs = self.encoder(
            embeddings, extended_attention_mask)    
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
        self.apply(self.init_bert_weights)

    def forward(self, embeddings, extended_attention_mask):
        pooled_output = self.bert(embeddings, extended_attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.model_from_embeddings = BertForSequenceClassificationFromEmbeddings(
            config, num_labels
        )
        self.num_labels = num_labels
        self.embeddings = BertEmbeddings(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, embed_only=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embeddings = self.embeddings(input_ids, token_type_ids)
        if embed_only:
            return embeddings, extended_attention_mask
        logits = self.model_from_embeddings(embeddings, extended_attention_mask)
        return logits

class BertClassifierFromEmbeddings(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertClassifierFromEmbeddings, self).__init__(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, embeddings):
        extended_attention_mask = torch.ones(embeddings.shape[0], 1, 1, embeddings.shape[1])\
            .to(embeddings.device)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers, self_output, attention_scores, attention_probs = self.encoder(embeddings,
                                    extended_attention_mask,
                                    output_all_encoded_layers=True)            
            
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        logits = self.classifier(pooled_output)
        
        return logits