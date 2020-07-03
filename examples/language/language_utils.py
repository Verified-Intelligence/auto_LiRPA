from auto_LiRPA.utils import logger
import numpy as np

def build_vocab(data_train, min_word_freq, dump=False, include=[]):
    vocab = {
        '[PAD]': 0,
        '[UNK]': 1,
        '[CLS]': 2,
        '[SEP]': 3,
        '[MASK]': 4
    }
    cnt = {}
    for example in data_train:
        for token in example['sentence'].strip().lower().split():
            if token in cnt:
                cnt[token] += 1
            else:
                cnt[token] = 1
    for w in cnt:
        if cnt[w] >= min_word_freq or w in include:
            vocab[w] = len(vocab)
    logger.info('Vocabulary size: {}'.format(len(vocab)))

    if dump:
        with open('tmp/vocab.txt', 'w') as file:
            for w in vocab.keys():
                file.write('{}\n'.format(w))

    return vocab

def tokenize(batch, vocab, max_seq_length, drop_unk=False):
    res = []
    for example in batch:
        t = example['sentence'].strip().lower().split(' ')
        if drop_unk:
            tokens = [w for w in t if w in vocab][:max_seq_length]
        else:
            tokens = []
            for token in t[:max_seq_length]:
                if token in vocab:
                    tokens.append(token)
                else:
                    tokens.append('[UNK]')
        res.append(tokens)    
    return res

def token_to_id(tokens, vocab):
    ids = []
    for t in tokens:
        ids.append([vocab[w] for w in t])
    return ids