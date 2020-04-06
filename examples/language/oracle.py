import torch
from auto_LiRPA.utils import logger
from auto_LiRPA import PerturbationSynonym
from data_utils import get_batches

def oracle(args, model, ptb, data, type):
    logger.info('Running oracle for {}'.format(type))
    model.eval()
    assert(isinstance(ptb, PerturbationSynonym))
    cnt_cor = 0
    word_embeddings = model.word_embeddings.weight
    vocab = model.vocab    
    for t, example in enumerate(data):
        embeddings, mask, tokens, label_ids = model.get_input([example])
        candidates = example['candidates']
        if tokens[0][0] == '[CLS]':
            candidates = [[]] + candidates + [[]]   
        embeddings_all = []
        def dfs(tokens, embeddings, budget, index):
            if index == len(tokens):
                embeddings_all.append(embeddings.cpu())
                return
            dfs(tokens, embeddings, budget, index + 1)
            if budget > 0 and tokens[index] != '[UNK]' and len(candidates[index]) > 0\
                    and tokens[index] == candidates[index][0]:
                for w in candidates[index][1:]:
                    if w in vocab:
                        _embeddings = torch.cat([
                            embeddings[:index],
                            word_embeddings[vocab[w]].unsqueeze(0),
                            embeddings[index + 1:]
                        ], dim=0)
                        dfs(tokens, _embeddings, budget - 1, index + 1)
        dfs(tokens[0], embeddings[0], ptb.budget, 0)
        cor = True
        for embeddings in get_batches(embeddings_all, args.oracle_batch_size):
            embeddings_tensor = torch.cat(embeddings).cuda().reshape(len(embeddings), *embeddings[0].shape)
            logits = model.model_from_embeddings(embeddings_tensor, mask)        
            for pred in list(torch.argmax(logits, dim=1)):
                if pred != example['label']:
                    cor = False
            if not cor: break
        cnt_cor += cor

        if (t + 1) % args.log_interval == 0:
            logger.info('{} {}/{}: oracle robust acc {:.3f}'.format(type, t + 1, len(data), cnt_cor * 1. / (t + 1)))
    logger.info('{}: oracle robust acc {:.3f}'.format(type, cnt_cor * 1. / (t + 1)))
    