from examples.utils import logger

def build_vocab(data_train, min_word_freq, dump=False):
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4
    }
    cnt = {}
    for example in data_train:
        for token in example["sentence"].strip().lower().split():
            if token in cnt:
                cnt[token] += 1
            else:
                cnt[token] = 1
    for w in cnt:
        if cnt[w] >= min_word_freq:
            vocab[w] = len(vocab)
    logger.info("Vocabulary size: {}".format(len(vocab)))

    if dump:
        with open("tmp/vocab.txt", "w") as file:
            for w in vocab.keys():
                file.write("{}\n".format(w))

    return vocab

def tokenize(batch, vocab, max_seq_length):
    tokens = []
    for example in batch:
        _tokens = []
        for token in example["sentence"].strip().lower().split(' ')[:max_seq_length]:
            if token in vocab:
                _tokens.append(token)
            else:
                _tokens.append("[UNK]")
        tokens.append(_tokens)    
    return tokens

def token_to_id(tokens, vocab):
    ids = []
    for t in tokens:
        ids.append([vocab[w] for w in t])
    return ids