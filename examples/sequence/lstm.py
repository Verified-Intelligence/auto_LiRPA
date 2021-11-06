import os
import shutil
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA.utils import logger

class LSTMCore(nn.Module):
    def __init__(self, args):
        super(LSTMCore, self).__init__()

        self.input_size = args.input_size // args.num_slices
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.device = args.device

        self.cell_f = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, X):
        batch_size, length = X.shape[0], X.shape[1]
        h_f = torch.zeros(batch_size, self.hidden_size).to(X.device)
        c_f = h_f.clone()
        h_f_sum = h_f.clone()
        for i in range(length):
            h_f, c_f = self.cell_f(X[:, i], (h_f, c_f))
            h_f_sum = h_f_sum + h_f
        states = h_f_sum / float(length)
        logits = self.linear(states)
        return logits

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device
        self.lr = args.lr
        self.num_slices = args.num_slices

        self.dir = args.dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint = 0
        self.model = LSTMCore(args)
        if args.load:
            self.model.load_state_dict(args.load)
            logger.info(f"Model loaded: {args.load}")
        else:
            logger.info("Model initialized")
        self.model = self.model.to(self.device)
        self.core = self.model

    def save(self, epoch):
        output_dir = os.path.join(self.dir, "ckpt-%d" % epoch)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        path = os.path.join(output_dir, "model")
        torch.save(self.core.state_dict(), path)
        with open(os.path.join(self.dir, "checkpoint"), "w") as file: 
            file.write(str(epoch))
        logger.info("LSTM saved: %s" % output_dir)

    def build_optimizer(self):
        param_group = []
        for p in self.core.named_parameters():
            param_group.append(p)
        param_group = [{"params": [p[1] for p in param_group], "weight_decay": 0.}]    
        return torch.optim.Adam(param_group, lr=self.lr)

    def get_input(self, batch):
        X = torch.cat([example[0].reshape(1, self.num_slices, -1) for example in batch])
        y = torch.tensor([example[1] for example in batch], dtype=torch.long)
        return X.to(self.device), y.to(self.device)

    def train(self):
        self.core.train()

    def eval(self):
        self.core.eval()    