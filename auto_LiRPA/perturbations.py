import os, json, pdb
import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA.utils import logger
from auto_LiRPA.bound_ops import LinearBound, Hull

class Perturbation:
    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps
    
    def concretize(self, x, A, sign=-1, aux=None):
        raise NotImplementedError

    def init_linear(self, x, aux=None, forward=False):
        raise NotImplementedError

    def init_interval(self, x, aux=None):
        raise NotImplementedError

# Perturbation constrained by the L_p norm
class PerturbationLpNorm(Perturbation):
    def __init__(self, norm, eps, x_L=None, x_U=None):
        self.norm = norm
        self.eps = eps
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None
        A = A.reshape(A.shape[0], A.shape[1], -1)
        x_L = x - self.eps if self.x_L is None else self.x_L
        x_U = x + self.eps if self.x_U is None else self.x_U
        if self.norm == np.inf:
            x_ub = x_U.reshape(x_U.shape[0], -1, 1)
            x_lb = x_L.reshape(x_L.shape[0], -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
        else:
            x = x.reshape(x.shape[0], -1, 1)
            deviation = A.norm(self.dual_norm, -1) * self.eps
            bound = A.bmm(x) + sign * deviation.unsqueeze(-1)
        bound = bound.squeeze(-1)
        return bound

    def init_linear(self, x, aux=None, forward=False):
        if not forward:
            return None, x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x).to(x.device)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, None, None), x, None
    
    def init_interval(self, x, aux=None):
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            x_L = x - x * self.eps / torch.norm(x, p=self.dual_norm)
            x_U = x + x * self.eps / torch.norm(x, p=self.dual_norm)
        return x_L, x_U

    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return 'PerturbationLpNorm(norm=inf, eps={})'.format(self.eps)
            else:
                return 'PerturbationLpNorm(norm=inf, eps={}, x_L={}, x_U={})'.format(self.eps, self.x_L, self.x_U)
        else:
            return 'PerturbationLpNorm(norm={}, eps={})'.format(self.norm, self.eps)


class PerturbationSynonym(Perturbation):
    def __init__(self, budget, eps=1.0, use_simple=False):
        super(PerturbationSynonym, self).__init__()
        self._load_synonyms()
        self.budget = budget
        self.eps = eps
        self.use_simple = use_simple
        self.model = None
        self.train = False

    def __repr__(self):
        return 'perturbation(Synonym-based word substitution budget={}, eps={})'.format(
            self.budget, self.eps)

    def _load_synonyms(self, path='data/synonyms.json'):
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info('Synonym list loaded for {} words'.format(len(self.synonym)))

    def set_train(self, train):
        self.train = train

    def concretize(self, x, A, sign, aux):
        assert(self.model is not None)     
        x_rep, cnt_rep, x_full, can_be_replaced = aux
        batch_size, length, length_full = x.shape[0], x.shape[1], x_full.shape[1]
        dim_out = A.shape[1]

        bias = torch.zeros(batch_size, dim_out).to(x.device)
        if A.shape[-1] != x.reshape(batch_size, -1).shape[1]:
            A = A.reshape(batch_size, dim_out, length_full, -1)
            mask = torch.zeros(batch_size, length_full, dtype=torch.float32, device=A.device)
            zeros = torch.zeros(dim_out, A.shape[-1], device=A.device)
            A_new = []
            for t in range(batch_size):
                cnt = 0
                for i in range(0, length_full):
                    if not can_be_replaced[t][i]:
                        mask[t][i] = 1
                    else:
                        A_new.append(A[t, :, i, :])
                        cnt += 1
                A_new += [zeros] * (length - cnt)
            A_new = torch.cat(A_new).reshape(batch_size, length, dim_out, A.shape[-1])\
                .transpose(1, 2)
            bias += torch.sum(
                mask.unsqueeze(-1) * torch.bmm(
                    A.permute(0, 2, 1, 3).reshape(-1, A.shape[1], A.shape[3]), 
                    x_full.reshape(batch_size * length_full, -1, 1)
                ).reshape(batch_size, length_full, -1), dim=1)
            A = A_new

        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2) 
        x = x.reshape(batch_size, length, -1, 1)

        if sign == 1:
            cmp, init = torch.max, -1e30
        else:
            cmp, init = torch.min, 1e30

        init_tensor = torch.ones(batch_size, dim_out).to(x.device) * init
        dp = [[init_tensor] * (self.budget + 1) for i in range(0, length + 1)]
        dp[0][0] = torch.zeros(batch_size, dim_out).to(x.device)     
 
        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(
            A,
            x.reshape(batch_size * length, x.shape[2], x.shape[3])
        ).reshape(batch_size, length, A.shape[1])

        max_num_cand = x_rep.shape[2]
        Ax_rep = torch.bmm(
            A,
            x_rep.reshape(batch_size * length, max_num_cand, x.shape[2]).transpose(-1, -2)
        ).reshape(batch_size, length, A.shape[1], max_num_cand)
        init_tensor = torch.ones(A.shape[1]).to(x.device) * init
        Ax_rep_bound = [[init_tensor] * length for t in range(batch_size)]
        for t in range(batch_size):
            for i in range(length):
                if cnt_rep[t][i] > 0:
                    Ax_rep_bound[t][i] = cmp(Ax_rep[t][i][:, :cnt_rep[t][i]], dim=1).values
            Ax_rep_bound[t] = torch.cat(Ax_rep_bound[t]).reshape(length, A.shape[1])
        Ax_rep_bound = torch.cat(Ax_rep_bound, dim=0).reshape(batch_size, length, A.shape[1])

        if self.use_simple and self.train:
            return torch.sum(cmp(Ax, Ax_rep_bound), dim=1) + bias

        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(
                    dp[i - 1][j] + Ax[:, i - 1], 
                    dp[i - 1][j - 1] + Ax_rep_bound[:, i - 1]
                )
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1, batch_size, dim_out)        

        return cmp(dp, dim=0).values + bias

    def _build_substitution(self, batch):
        for t, example in enumerate(batch):
            if not 'candidates' in example or example['candidates'] is None:
                candidates = []
                tokens = example['sentence'].strip().lower().split(' ')
                for i in range(len(tokens)):
                    _cand = []
                    if tokens[i] in self.synonym:
                        for w in self.synonym[tokens[i]]:
                            if w in self.model.vocab:
                                _cand.append(w)
                    if len(_cand) > 0:
                        _cand = [tokens[i]] + _cand
                    candidates.append(_cand)
                example['candidates'] = candidates

    def init_linear(self, x, aux=None, forward=False):
        tokens, batch = aux
        assert(len(x.shape) == 3)
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]

        max_pos = 1
        can_be_replaced = np.zeros((batch_size, length), dtype=np.bool)

        self._build_substitution(batch)

        for t in range(batch_size):
            cnt = 0
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                if tokens[t][i] == '[UNK]' or \
                        len(candidates[i]) == 0 or tokens[t][i] != candidates[i][0]:
                    continue
                for w in candidates[i][1:]:
                    if w in self.model.vocab:
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            max_pos = max(max_pos, cnt)

        dim = max_pos * dim_word
        if forward:
            eye = torch.eye(dim_word).to(x.device)
            lw = torch.zeros(batch_size, dim, length, dim_word).to(x.device)
            lb = torch.zeros_like(x).to(x.device)   
        x_new = []     
        word_embeddings = self.model.word_embeddings.weight
        vocab = self.model.vocab
        x_rep = [[[] for i in range(max_pos)] for t in range(batch_size)]
        max_num_cand = 0
        for t in range(batch_size):
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]      
            _x_new = []
            cnt = 0
            for i in range(length):
                if can_be_replaced[t][i]:
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    if forward:
                        lw[t, (len(_x_new) * dim_word):((len(_x_new) + 1) * dim_word), i, :] = eye
                        lb[t, i, :] = x[t, i, :] - word_embed
                    _x_new.append(word_embed)
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            x_rep[t][cnt].append(
                                word_embeddings[self.model.vocab[w]])
                    max_num_cand = max(max_num_cand, len(x_rep[t][cnt]))
                    cnt += 1
                else:
                    if forward:
                        lb[t, i, :] = x[t, i, :]
            if len(_x_new) == 0:
                x_new.append(torch.zeros(max_pos, dim_word).to(x.device))
            else:
                x_new.append(torch.cat([
                    torch.cat(_x_new).reshape(len(_x_new), dim_word),
                    torch.zeros(max_pos - len(_x_new), dim_word).to(x.device)
                ]))
        x_new = torch.cat(x_new).reshape(batch_size, max_pos, dim_word)
        zeros = torch.zeros(dim_word, device=x.device)
        _x_rep = x_rep
        x_rep = []
        cnt_rep = [[0] * max_pos for t in range(batch_size)]
        for t in range(batch_size):
            for i in range(max_pos):
                cnt_rep[t][i] = len(_x_rep[t][i])
                x_rep += _x_rep[t][i] + [zeros] * (max_num_cand - cnt_rep[t][i])
        x_rep = torch.cat(x_rep).reshape(batch_size, max_pos, max_num_cand, dim_word)
        x_rep = x_rep * self.eps + x_new.unsqueeze(2) * (1 - self.eps)

        if forward:
            uw, ub = lw, lb
        else:
            lw, lb, uw, ub = None, None, None, None

        return LinearBound(lw, lb, uw, ub, None, None), x_new, (x_rep, cnt_rep, x, can_be_replaced)

    def init_interval(self, x, aux):
        tokens, batch = aux
        assert(len(x.shape) == 3)
        word_embeddings = self.model.word_embeddings.weight        
        batch_size, length, dim_word, device = x.shape[0], x.shape[1], x.shape[2], x.device
        hull, mask = [], []
        self._build_substitution(batch)
        max_num_cand = 1
        for t in range(batch_size):
            for cand in batch[t]['candidates']:
                max_num_cand = max(max_num_cand, len(cand))
        zero_base = torch.zeros(dim_word, device=device)
        zeros = [torch.tensor([], device=device)] + \
            [zero_base.repeat(i) for i in range(1, max_num_cand)]
        for t in range(batch_size):
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]                   
            assert(len(tokens[t]) <= len(candidates))
            for i in range(length):
                hull.append(x[t][i])
                mask.append(1)   
                cnt = 1      
                if i >= len(tokens[t]) or tokens[t][i] == '[UNK]' or \
                        len(candidates[i]) > 0 and tokens[t][i] != candidates[i][0]:
                    pass
                else:
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            hull.append(word_embeddings[self.model.vocab[w]])                            
                            mask.append(1)
                            cnt += 1
                if cnt < max_num_cand:
                    pad = max_num_cand - cnt
                    hull.append(zeros[pad])
                    mask += [0] * pad
        hull = torch.cat(hull).reshape(batch_size, length, max_num_cand, -1) * self.budget + \
            x.unsqueeze(2) * (1 - self.budget)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)\
            .reshape(batch_size, length, max_num_cand)
        inf = 1e20
        lower = torch.min(mask.unsqueeze(-1) * hull + (1 - mask).unsqueeze(-1) * inf, dim=2).values
        upper = torch.max(mask.unsqueeze(-1) * hull + (1 - mask).unsqueeze(-1) * (-inf), dim=2).values
        return Hull(hull, mask, self.eps, lower, upper)
