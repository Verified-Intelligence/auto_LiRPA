import os, json, pdb
import numpy as np
import torch
from auto_LiRPA.utils import logger
from auto_LiRPA.bound_ops import LinearBound, Hull

class Perturbation:
    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps
    
    def concretize(self, x, A, sum_b, sign=-1):
        raise NotImplementedError

    def earear(self, x):
        raise NotImplementedError

    def init_interval(self, x):
        raise NotImplementedError

# Perturbation constrained by the L_p norm
class PerturbationLpNorm(Perturbation):
    def __init__(self, norm, eps, x_L=None, x_U=None):
        self.norm = norm
        self.eps = eps
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U


    def concretize(self, x, A, sum_b, sign=-1):
        if A is None:
            return None
        A = A.reshape(A.shape[0], A.shape[1], -1)
        sum_b = sum_b.reshape(A.shape[0], -1)
        x_L = x - self.eps if self.x_L is None else self.x_L
        x_U = x + self.eps if self.x_U is None else self.x_U
        # x_U, x_L = x + self.eps, x - self.eps
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
        bound = bound.squeeze(-1) + sum_b
        return bound

    def init_linear(self, x):
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x).to(x.device)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, None, None), x
    
    def init_interval(self, x):
        x_L = x - self.eps if self.x_L is None else self.x_L
        x_U = x + self.eps if self.x_U is None else self.x_U
        return x_L, x_U

class PerturbationLpNorm_2bounds(Perturbation):
    def __init__(self, norm, eps):
        self.norm = norm
        self.eps = eps # eps of input x
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        logger.debug('Using l{} norm to concretize'.format(self.dual_norm))

    def concretize_2bounds(self, x, Ax, sum_b, sign=-1, y=[]):
        # only support linear layer so far
        if Ax is None:
            return None
        batch = x.shape[0]
        _tmp_Ay = 0
        _tmp_center = 0
        if sign == -1:
            for i in range(len(y)):
                logger.debug(y[i].shape)
                logger.debug(y[i].lA_y.shape)
                Ay = y[i].lA_y
                Ay = Ay.reshape(*Ay.shape[:2], -1)
                _tmp_Ay -= torch.norm(Ay, self.dual_norm, -1) * y[i].eps
                _tmp_center += Ay.bmm(y[i].reshape(-1).unsqueeze(-1).unsqueeze(0).repeat(batch, 1, 1))
        elif sign == 1:
            for i in range(len(y)):
                Ay = y[i].uA_y
                Ay = Ay.reshape(*Ay.shape[:2], -1)
                _tmp_Ay += torch.norm(Ay, self.dual_norm, -1) * y[i].eps
                _tmp_center += Ay.bmm(y[i].reshape(-1).unsqueeze(-1).unsqueeze(0).repeat(batch, 1, 1))

        _tmp_center += Ax.bmm(x.reshape(batch, -1).unsqueeze(-1)) + sum_b.unsqueeze(-1)
        bound = _tmp_center.squeeze(-1) + sign * torch.norm(Ax, self.dual_norm, -1) * self.eps + _tmp_Ay

        return bound

    def init_linear(self, value):
        batch_size = value.shape[0]
        dim = value.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(value.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *value.shape[1:])
        lb = torch.zeros_like(value).to(value.device)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, None, None)

class PerturbationPositions(PerturbationLpNorm):
    def __init__(self, norm, eps, length, pos):
        super(PerturbationPositions, self).__init__(norm, eps)
        self.length = length
        self.pos = pos
    
    def concretize(self, x, A, sum_b, sign):
        x = x.reshape(x.shape[0], -1, 1)
        dim = x.shape[1] // self.length
        A = A.reshape(A.shape[0], A.shape[1], -1)
        sum_b = sum_b.reshape(A.shape[0], -1).to(torch.float)
        if A.shape[-1] == x.shape[1]:
            res = A.bmm(x).squeeze(-1) + sum_b
            for pos in self.pos:
                res = res + sign * self.eps * \
                    A[:, :, (dim * pos):(dim * (pos + 1))].norm(self.dual_norm, dim=-1)
        else:
            x = torch.cat([x[:, (dim * pos):(dim * (pos + 1)), :] for pos in self.pos], dim=1)
            res = A.bmm(x).squeeze(-1) + sum_b
            for i in range(len(self.pos)):
                res = res + sign * self.eps * \
                    A[:, :, (dim * i):(dim * (i + 1))].norm(self.dual_norm, dim=-1)
        return res

    def init_linear(self, x):
        assert(len(x.shape) == 3) # (batch_size, length, *)
        assert(self.length == x.shape[1])
        batch_size, length, dim = x.shape
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = torch.zeros(batch_size, dim * len(self.pos), length, dim).to(x.device)
        lb = x.clone()
        for i in range(len(self.pos)):
            lw[:, (i * dim): ((i + 1) * dim), self.pos[i]] = eye
            lb[:, self.pos[i], :] *= 0.
        uw, ub = lw.clone(), lb.clone()

        return LinearBound(lw, lb, uw, ub, None, None), x

    def init_interval(self, x):
        raise NotImplementedError

class PerturbationSynonym(Perturbation):
    def __init__(self, budget):
        super(PerturbationSynonym, self).__init__()
        self.substitution = self._load_substitution()
        self._load_synonyms()
        self.budget = budget
        self.model = None

    def _load_substitution(self):
        res = {}
        for set in ["train", "dev", "test"]:
            with open("./tmp/{}_lm_scores.txt".format(set)) as file:
                line = file.readline().strip().split('\t')
                while True:
                    if len(line) < 2: break
                    sentence = line[-1]
                    tokens = sentence.split(' ')
                    candidates = [[] for i in range(len(tokens))]
                    while True:
                        line = file.readline().strip().split('\t')
                        if len(line) != 4: break
                        if (len(candidates[int(line[1])]) == 0 or\
                                float(line[3]) >= candidates[int(line[1])][0][1] - 5.0) and\
                                float(line[3]) != float("-inf"):
                            candidates[int(line[1])].append((line[2], float(line[3])))
                    res[sentence] = candidates
        return res

    def _load_synonyms(self,
            url="https://worksheets.codalab.org/rest/bundles/0x6ba96d4232c046bc9944965919959d93/contents/blob/",
            path="tmp/synonyms.json"):
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        if not os.path.exists(path):
            logger.info("Downloading the synonym list from {}".format(url))
            os.system("curl {} -o {}".format(url, path))
            logger.info("Synonym list saved to {}".format(path))
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info("Synonym list loaded for {} words".format(len(self.synonym)))

    def concretize(self, x, A, sum_b, sign):
        assert(self.model is not None)     
        x, x_rep, can_be_replaced = x
        batch_size, length = x.shape[0], x.shape[1]

        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2) 
        x = x.reshape(batch_size, length, -1, 1)
        sum_b = sum_b.reshape(batch_size, -1).to(torch.float)

        if sign == 1:
            cmp, init = torch.max, -1e30
        else:
            cmp, init = torch.min, 1e30

        init_tensor = torch.ones(*sum_b.shape).to(x.device) * init
        dp = [[init_tensor] * (self.budget + 1) for i in range(0, length + 1)]
        dp[0][0] = torch.zeros(*sum_b.shape).to(x.device)     

        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(
            A,
            x.reshape(batch_size * length, x.shape[2], x.shape[3])
        ).reshape(batch_size, length, A.shape[1])

        max_num_cand = 0
        for t in range(batch_size):
            max_num_cand = max(max_num_cand, max([len(x_rep[t][i]) for i in range(len(x_rep[t]))]))
        x_rep_tmp = x_rep
        x_rep = torch.zeros(batch_size, length, max_num_cand, x.shape[2]).to(x.device)
        for t in range(batch_size):
            for i in range(0, length):
                if len(x_rep_tmp[t][i]) > 0:
                    x_rep[t][i][:len(x_rep_tmp[t][i]), :] += \
                        torch.cat(x_rep_tmp[t][i]).reshape(len(x_rep_tmp[t][i]), -1)              

        Ax_rep = torch.bmm(
            A,
            x_rep.reshape(batch_size * length, max_num_cand, x.shape[2]).transpose(-1, -2)
        ).reshape(batch_size, length, A.shape[1], max_num_cand)
        init_tensor = torch.ones(A.shape[1]).to(x.device) * init
        Ax_rep_bound = [[init_tensor] * length for t in range(batch_size)]
        for t in range(batch_size):
            for i in range(length):
                if len(x_rep_tmp[t][i]) > 0:
                    Ax_rep_bound[t][i] = cmp(Ax_rep[t][i][:, :len(x_rep_tmp[t][i])], dim=1).values
            Ax_rep_bound[t] = torch.cat(Ax_rep_bound[t]).reshape(length, A.shape[1])
        Ax_rep_bound = torch.cat(Ax_rep_bound, dim=0).reshape(batch_size, length, A.shape[1])

        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(
                    dp[i - 1][j] + Ax[:, i - 1], 
                    dp[i - 1][j - 1] + Ax_rep_bound[:, i - 1]
                )
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1, *sum_b.shape)        

        return sum_b + cmp(dp, dim=0).values

    def _build_substitution(self, batch, tokens):
        for t in range(len(batch)):
            if not batch[t]["sentence"] in self.substitution:
                candidates = []
                for i in range(len(tokens[t])):
                    _cand = []
                    if tokens[t][i] in self.synonym:
                        for w in self.synonym[tokens[t][i]]:
                            if w in self.model.vocab:
                                _cand.append((w, 0))
                    if len(_cand) > 0:
                        _cand = [(tokens[t][i], 0)] + _cand
                    candidates.append(_cand)
                self.substitution[batch[t]["sentence"]] = candidates

    def init_linear(self, x, eps=1.0):
        x, tokens, batch = x
        assert(len(x.shape) == 3)
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]

        max_pos = 1
        can_be_replaced = np.zeros((batch_size, length), dtype=np.bool)

        self._build_substitution(batch, tokens)

        for t in range(batch_size):
            cnt = 0
            candidates = self.substitution[batch[t]["sentence"]]
            # for transformers
            if tokens[t][0] == "[CLS]" and len(candidates) != len(tokens[t]):
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                if tokens[t][i] == "[UNK]":
                    continue
                try:
                    if len(candidates[i]) == 0 or tokens[t][i] != candidates[i][0][0]:
                        continue
                except:
                    pdb.set_trace()
                for w in candidates[i][1:]:
                    if w[0] in self.model.vocab:
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            max_pos = max(max_pos, cnt)

        dim = max_pos * dim_word
        eye = torch.eye(dim_word).to(x.device)
        lw = torch.zeros(batch_size, dim, length, dim_word).to(x.device)
        lb = torch.zeros_like(x).to(x.device)   
        x_new = []     
        word_embeddings = self.model.word_embeddings.weight
        vocab = self.model.vocab
        x_rep = [[[] for i in range(max_pos)] for t in range(batch_size)]
        # pdb.set_trace()
        for t in range(batch_size):
            candidates = self.substitution[batch[t]["sentence"]]
            # for transformers
            if tokens[t][0] == "[CLS]" and len(candidates) != len(tokens[t]):
                candidates = [[]] + candidates + [[]]      
            _x_new = []
            cnt = 0
            for i in range(length):
                if can_be_replaced[t][i]:
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    lw[t, (len(_x_new) * dim_word):((len(_x_new) + 1) * dim_word), i, :] = eye
                    lb[t, i, :] = x[t, i, :] - word_embed
                    _x_new.append(word_embed)

                    for w in candidates[i][1:]:
                        if w[0] in self.model.vocab:
                            x_rep[t][cnt].append(
                                word_embeddings[self.model.vocab[w[0]]])
                    cnt += 1
                else:
                    lb[t, i, :] = x[t, i, :]
            if len(_x_new) == 0:
                x_new.append(torch.zeros(max_pos, dim_word).to(x.device))
            else:
                x_new.append(torch.cat([
                    torch.cat(_x_new).reshape(len(_x_new), dim_word),
                    torch.zeros(max_pos - len(_x_new), dim_word).to(x.device)
                ]))
        x_new = torch.cat(x_new).reshape(batch_size, max_pos, dim_word)

        lw = lw * eps
        lb = lb * eps + x * (1 - eps)

        uw, ub = lw, lb
        return LinearBound(lw, lb, uw, ub, None, None), (x_new, x_rep, can_be_replaced)

    def init_interval(self, x):
        x, tokens, batch = x
        assert(len(x.shape) == 3)
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]        
        word_embeddings = self.model.word_embeddings.weight
        hull = []
        for t in range(batch_size):
            _hull = [x[t]]
            if batch[t]["sentence"] in self.substitution:
                candidates = self.substitution[batch[t]["sentence"]]
                # for transformers
                if tokens[t][0] == "[CLS]":
                    candidates = [[]] + candidates + [[]]                   
                try:
                    assert(len(tokens[t]) <= len(candidates))
                except:
                    pdb.set_trace()
                for i in range(len(tokens[t])):
                    if tokens[t][i] == "[UNK]":
                        continue
                    if len(candidates[i]) > 0:
                        # this word does not appear in the vocabulary of the
                        # language model
                        if tokens[t][i] != candidates[i][0][0]:
                            continue
                    for w in candidates[i][1:]:
                        if w[0] in self.model.vocab:
                            _hull.append(torch.cat([
                                x[t][:i],
                                (x[t][i] + (
                                    word_embeddings[self.model.vocab[w[0]]]\
                                    - word_embeddings[self.model.vocab[tokens[t][i]]]
                                ) * self.budget).unsqueeze(0),
                                x[t][(i+1):]
                            ]))
            else:
                for i in range(len(tokens[t])):
                    if tokens[t][i] in self.synonym:
                        for w in self.synonym[tokens[t][i]]:
                            if w in self.model.vocab:
                                _hull.append(torch.cat([
                                    x[t][:i],
                                    (x[t][i] + (
                                        word_embeddings[self.model.vocab[w]]\
                                        - word_embeddings[self.model.vocab[tokens[t][i]]]
                                    ) * self.budget).unsqueeze(0),
                                    x[t][(i+1):]
                                ]))
            hull.append(_hull)
        return Hull(hull, self.eps)