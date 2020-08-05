import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA.utils import logger
from language_utils import build_vocab

class LSTMFromEmbeddings(nn.Module):
    def __init__(self, args, vocab_size):
        super(LSTMFromEmbeddings, self).__init__()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.device = args.device

        self.cell_f = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.cell_b = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.num_classes)
        if args.dropout is not None:
            self.dropout = nn.Dropout(p=args.dropout)
            logger.info('LSTM dropout: {}'.format(args.dropout))
        else:
            self.dropout = None

    def forward(self, embeddings, mask):
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        embeddings = embeddings * mask.unsqueeze(-1)
        batch_size = embeddings.shape[0]
        length = embeddings.shape[1]
        h_f = torch.zeros(batch_size, self.hidden_size).to(embeddings.device)
        c_f = h_f.clone()
        h_b, c_b = h_f.clone(), c_f.clone()
        h_f_sum, h_b_sum = h_f.clone(), h_b.clone()
        for i in range(length):
            h_f, c_f = self.cell_f(embeddings[:, i], (h_f, c_f))
            h_b, c_b = self.cell_b(embeddings[:, length - i - 1], (h_b, c_b))
            h_f_sum = h_f_sum + h_f
            h_b_sum = h_b_sum + h_b
        states = torch.cat([h_f_sum / float(length), h_b_sum / float(length)], dim=-1)
        logits = self.linear(states)
        return logits

class LSTM(nn.Module):
    def __init__(self, args, data_train):
        super(LSTM, self).__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.max_seq_length = args.max_sent_length
        self.min_word_freq = args.min_word_freq
        self.device = args.device
        self.lr = args.lr

        self.dir = args.dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.vocab = self.vocab_actual = build_vocab(data_train, args.min_word_freq)
        self.checkpoint = 0

        if args.load:
            ckpt = torch.load(args.load, map_location=torch.device(self.device))  
            self.embedding = torch.nn.Embedding(len(self.vocab), self.embedding_size)
            self.model_from_embeddings = LSTMFromEmbeddings(args, len(self.vocab))
            self.model = self.embedding, LSTMFromEmbeddings(args, len(self.vocab))
            self.embedding.load_state_dict(ckpt['state_dict_embedding'])
            self.model_from_embeddings.load_state_dict(ckpt['state_dict_model_from_embeddings'])
            self.checkpoint = ckpt['epoch']                
        else:
            self.embedding = torch.nn.Embedding(len(self.vocab), self.embedding_size)
            self.model_from_embeddings = LSTMFromEmbeddings(args, len(self.vocab))
            self.model = self.embedding, LSTMFromEmbeddings(args, len(self.vocab))
            logger.info("Model initialized")
        self.embedding = self.embedding.to(self.device)
        self.model_from_embeddings = self.model_from_embeddings.to(self.device)
        self.word_embeddings = self.embedding

    def save(self, epoch):
        path = os.path.join(self.dir, 'ckpt_{}'.format(epoch))
        torch.save({
            'state_dict_embedding': self.embedding.state_dict(), 
            'state_dict_model_from_embeddings': self.model_from_embeddings.state_dict(),
            'epoch': epoch
        }, path)
        logger.info('LSTM saved: {}'.format(path))

    def build_optimizer(self):
        self.model = (self.model[0], self.model_from_embeddings)
        param_group = []
        for m in self.model:
            for p in m.named_parameters():
                param_group.append(p)
        param_group = [{"params": [p[1] for p in param_group], "weight_decay": 0.}]    
        return torch.optim.Adam(param_group, lr=self.lr)

    def get_input(self, batch):
        mask, tokens = [], []
        for example in batch:
            _tokens = []
            for token in example["sentence"].strip().lower().split(' ')[:self.max_seq_length]:
                if token in self.vocab:
                    _tokens.append(token)
                else:
                    _tokens.append("[UNK]")
            tokens.append(_tokens)
        max_seq_length = max([len(t) for t in tokens])
        token_ids = []
        for t in tokens:
            ids = [self.vocab[w] for w in t]
            mask.append(torch.cat([
                torch.ones(1, len(ids)),
                torch.zeros(1, self.max_seq_length - len(ids))
            ], dim=-1).to(self.device))
            ids += [self.vocab["[PAD]"]] * (self.max_seq_length - len(ids))
            token_ids.append(ids)
        embeddings = self.embedding(torch.tensor(token_ids, dtype=torch.long).to(self.device))
        mask = torch.cat(mask, dim=0)
        label_ids = torch.tensor([example["label"] for example in batch]).to(self.device)
        return embeddings, mask, tokens, label_ids

    def train(self):
        self.model_from_embeddings.train()

    def eval(self):
        self.model_from_embeddings.eval()    