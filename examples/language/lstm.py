import os, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA.utils import logger
from examples.language.language_utils import build_vocab

class LSTMFromEmbeddings(nn.Module):
    def __init__(self, args, vocab_size):
        super(LSTMFromEmbeddings, self).__init__()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.num_labels = args.num_labels
        self.device = args.device

        self.linear_in = nn.Linear(self.embedding_size, self.embedding_size)
        self.cell_f = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.cell_b = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.num_labels)

    def forward(self, embeddings, mask):
        embeddings = self.linear_in(embeddings)
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
        if os.path.exists(os.path.join(self.dir, "checkpoint")):
            with open(os.path.join(self.dir, "checkpoint")) as file:
                self.checkpoint = int(file.readline())
            dir_ckpt = os.path.join(self.dir, "ckpt-{}".format(self.checkpoint))
            path = os.path.join(dir_ckpt, "model")
            self.model = torch.load(path)
            logger.info("Model loaded: {}".format(dir_ckpt))
        else:
            self.embedding = torch.nn.Embedding(len(self.vocab), self.embedding_size)
            self.model = self.embedding, LSTMFromEmbeddings(args, len(self.vocab))
            logger.info("Model initialized")
        self.embedding, self.model_from_embeddings = self.model
        self.embedding = self.embedding.to(self.device)
        self.model_from_embeddings = self.model_from_embeddings.to(self.device)
        self.word_embeddings = self.embedding

    def save(self, epoch):
        self.model = (self.model[0], self.model_from_embeddings)        

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

        output_dir = os.path.join(self.dir, "ckpt-%d" % epoch)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        path = os.path.join(output_dir, "model")
        torch.save(self.model, path)

        with open(os.path.join(self.dir, "checkpoint"), "w") as file: 
            file.write(str(epoch))

        logger.info("LSTM saved: %s" % output_dir)

    def _build_actual_vocab(self, args, vocab, data_train):
        vocab_actual = {}
        for example in data_train:
            for token in example["sentence"].strip().lower().split():
                if token in vocab:
                    if not token in vocab_actual:
                        vocab_actual[token] = 1
                    else:
                        vocab_actual[token] += 1
        for w in list(vocab_actual.keys()):
            if vocab_actual[w] < self.min_word_freq:
                del(vocab_actual[w])
        logger.info("Size of the vocabulary for perturbation: {}".format(len(vocab_actual)))
        return vocab_actual

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