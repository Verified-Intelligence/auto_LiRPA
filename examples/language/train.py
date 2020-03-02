import argparse, random, pickle, os, pdb, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from examples.language.data_utils import load_data, get_batches
from examples.language.Transformer.BERT import BERT
from examples.language.lstm import LSTM
from auto_LiRPA.utils import AverageMeter, logger
from auto_LiRPA.perturbations import PerturbationLpNorm, PerturbationSynonym
from auto_LiRPA.bound_general import BoundGeneral
from pytorch_pretrained_bert.optimization import BertAdam

parser = argparse.ArgumentParser()

parser.add_argument("--train", action="store_true")
parser.add_argument("--robust", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--dir", type=str, default="model")
parser.add_argument("--data", type=str, default="sst", choices=["sst"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

parser.add_argument("--ptb", type=str, default="synonym", 
                    choices=["synonym"])
parser.add_argument("--eps", type=float, default=0.01)
parser.add_argument("--budget", type=int, default=3)
parser.add_argument("--verbose_bound", action="store_true")
parser.add_argument("--ibp", action="store_true")
parser.add_argument("--kappa", type=float, default=0.8)
parser.add_argument("--check", action="store_true")
parser.add_argument("--method", type=str, default="None",
                    choices=[None, "forward", "backward"])
parser.add_argument("--res_file", type=str, default=None)

parser.add_argument("--model", type=str, default="transformer",
                    choices=["transformer", "lstm"])
parser.add_argument("--num_epochs", type=int, default=20)  
parser.add_argument("--num_epochs_all_nodes", type=int, default=5)      
parser.add_argument("--num_epochs_warmup", type=int, default=1)      
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--min_word_freq", type=int, default=2)
parser.add_argument("--use_bert", action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--oracle_batch_size", type=int, default=256)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_sent_length", type=int, default=32)
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--grad_clip", type=float, default=None)
parser.add_argument("--num_labels", type=int, default=2) 
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--num_attention_heads", type=int, default=4)
parser.add_argument("--hidden_size", type=int, default=64) # 256
parser.add_argument("--embedding_size", type=int, default=64) # 256
parser.add_argument("--intermediate_size", type=int, default=128) # 512
parser.add_argument("--hidden_act", type=str, default="relu")
parser.add_argument("--layer_norm", type=str, default="no_var",
                    choices=["standard", "no", "no_var"])

args = parser.parse_args()   

def build_perturbation():
    if args.ptb == "lp_norm":
        return PerturbationLpNorm(norm=np.inf, eps=args.eps)    
    elif args.ptb == "synonym":
        return PerturbationSynonym(budget=args.budget)
    else:
        raise NotImplementedError

def scale_gradients(optimizer, gradient_accumulation_steps):    
    parameters = []
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            parameters.append(param)
            if param.grad is not None:
                param.grad.data /= gradient_accumulation_steps
    if args.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        
def convert(model, ptb, batch, verbose=False):
    model.train()
    embeddings, mask, _, _ = model.get_input(batch)
    converted_model = BoundGeneral(
        model.model_from_embeddings, (embeddings, mask), verbose=verbose)    
    converted_model.eval()
    model.model_from_embeddings.eval()
    model.model_from_embeddings = converted_model

def step(model, ptb, batch, eps=1.0, train=False):
    model_bound = model.model_from_embeddings
    if train:
        model.train()
        grad = torch.enable_grad()
    else:
        model.eval()
        grad = torch.no_grad()

    with grad:
        embeddings, mask, tokens, label_ids = model.get_input(batch)
        logits = model_bound(embeddings, mask)

        if args.robust and eps > 1e-9:
            C = torch.eye(args.num_labels).to(model.device).unsqueeze(0).repeat(len(batch), 1, 1)
            if args.ptb.find("lp_norm") != -1:
                x = embeddings
            else:
                x = (embeddings, tokens, batch)

            start_time = time.time()
            ptb.set_eps(eps)
            logits_l, logits_u = model_bound.compute_bounds(
                ptb=ptb, x=x, C=C, IBP=args.ibp, forward=True, method=args.method)
            if args.check:
                # possible when there is no available perturbation
                try:
                    assert(torch.min(logits - logits_l) > -1e-3)
                    assert(torch.min(logits_u - logits) > -1e-3)
                except:
                    pdb.set_trace()
            one_hot = F.one_hot(label_ids, num_classes=args.num_labels)\
                .to(torch.float32).to(model.device)
            logits_robust = logits_l * one_hot + logits_u * (1. - one_hot)
        else:
            logits_robust = logits
        
    loss_fct = nn.CrossEntropyLoss()

    preds = torch.argmax(logits, dim=1)
    acc = torch.sum((preds == label_ids).to(torch.float32)) / len(batch)
    loss = loss_fct(logits, label_ids)
    loss_all = loss

    if args.robust:
        preds_robust = torch.argmax(logits_robust, dim=1)
        acc_robust = torch.sum((preds_robust == label_ids).to(torch.float32)) / len(batch)
        loss_robust = loss_fct(logits_robust, label_ids)
        loss_all = args.kappa * loss_robust + (1. - args.kappa) * loss_all
    else:
        acc_robust, loss_robust = acc * 0., loss * 0.

    if train:
        loss_all.backward()

    acc, loss = acc.detach(), loss.detach()
    acc_robust, loss_robust = acc_robust.detach(), loss_robust.detach()

    return acc, loss, acc_robust, loss_robust

def oracle(model, ptb, data, type):
    logger.info("Running oracle for {}".format(type))
    model.eval()
    assert(isinstance(ptb, PerturbationSynonym))
    cnt_cor = 0
    word_embeddings = model.word_embeddings.weight
    vocab = model.vocab    
    for t, example in enumerate(data):
        embeddings, mask, tokens, label_ids = model.get_input([example])
        candidates = ptb.substitution[example["sentence"]]
        if tokens[0][0] == "[CLS]":
            candidates = [[]] + candidates + [[]]   
        embeddings_all = []
        def dfs(tokens, embeddings, budget, index):
            if index == len(tokens):
                embeddings_all.append(embeddings.cpu())
                return
            dfs(tokens, embeddings, budget, index + 1)
            if budget > 0 and tokens[index] != "[UNK]" and len(candidates[index]) > 0\
                    and tokens[index] == candidates[index][0][0]:
                for w in candidates[index][1:]:
                    if w[0] in vocab:
                        _embeddings = torch.cat([
                            embeddings[:index],
                            (embeddings[index] \
                                - word_embeddings[vocab[tokens[index]]]\
                                + word_embeddings[vocab[w[0]]]
                             ).unsqueeze(0),
                            embeddings[index + 1:]
                        ], dim=0)
                        dfs(tokens, _embeddings, budget - 1, index + 1)
        dfs(tokens[0], embeddings[0], ptb.budget, 0)
        cor = True
        for embeddings in get_batches(embeddings_all, args.oracle_batch_size):
            embeddings_tensor = torch.cat(embeddings).cuda().reshape(len(embeddings), *embeddings[0].shape)
            logits = model.model_from_embeddings(embeddings_tensor, mask)        
            for pred in list(torch.argmax(logits, dim=1)):
                if pred != example["label"]:
                    cor = False
            if not cor: break
        cnt_cor += cor

        if (t + 1) % args.log_interval == 0:
            logger.info("{} {}/{}: oracle robust acc {:.3f}".format(type, t + 1, len(data), cnt_cor * 1. / (t + 1)))
    logger.info("{}: oracle robust acc {:.3f}".format(type, cnt_cor * 1. / (t + 1)))
    
data_train_warmup, data_train, data_dev, data_test = load_data(args.data)
logger.info("Dataset sizes: {}/{}/{}/{}".format(
    len(data_train_warmup), len(data_train), len(data_dev), len(data_test)))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.model == "transformer":
    model = BERT(args, data_train)
elif args.model == "lstm":
    model = LSTM(args, data_train)

dev_batches = get_batches(data_dev, args.batch_size)
test_batches = get_batches(data_test, args.batch_size)        

ptb = build_perturbation()
convert(model, ptb, dev_batches[0], verbose=args.verbose_bound)
ptb.model = model
optimizer = model.build_optimizer()
logger.info("Model converted to support bounds")

avg_acc, avg_loss, avg_acc_robust, avg_loss_robust = [AverageMeter() for i in range(4)]

def train(epoch):
    assert(optimizer is not None)
    if epoch <= args.num_epochs_all_nodes:
        train_batches = get_batches(data_train_warmup, args.batch_size)
    else:
        train_batches = get_batches(data_train, args.batch_size)
    avg_acc.reset()
    avg_loss.reset()
    avg_acc_robust.reset()
    avg_loss_robust.reset()       
    if args.robust:
        eps_inc_per_step = 1.0 / (args.num_epochs_warmup * len(train_batches))
    for i, batch in enumerate(train_batches):
        if args.robust:
            eps = min(eps_inc_per_step * ((epoch -  - 1) * len(train_batches) + i + 1), 1.0)
        else:
            eps = 0.
        acc, loss, acc_robust, loss_robust = \
            step(model, ptb, batch, eps=eps, train=True)
        avg_acc.update(acc, len(batch))
        avg_loss.update(loss, len(batch))
        avg_acc_robust.update(acc_robust, len(batch))
        avg_loss_robust.update(loss_robust, len(batch))   
        if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_batches):
            scale_gradients(optimizer, i % args.gradient_accumulation_steps + 1)
            optimizer.step()
            optimizer.zero_grad()            
        if (i + 1) % args.log_interval == 0:
            logger.info("Epoch {}, training step {}/{}: acc {:.3f}, loss {:.3f}, acc_robust {:.3f}, loss_robust {:.3f}, eps {:.3f}".format(
                epoch, i + 1, len(train_batches),
                avg_acc.avg, avg_loss.avg, avg_acc_robust.avg, avg_loss_robust.avg,
                eps
            ))
    model.save(epoch)

def infer(epoch, batches, type):
    avg_acc.reset()
    avg_loss.reset()
    avg_acc_robust.reset()
    avg_loss_robust.reset()    
    for i, batch in enumerate(batches):
        acc, loss, acc_robust, loss_robust = step(model, ptb, batch)
        avg_acc.update(acc, len(batch))
        avg_loss.update(loss, len(batch)) 
        avg_acc_robust.update(acc_robust, len(batch))
        avg_loss_robust.update(loss_robust, len(batch))                   
        if (i + 1) % args.log_interval == 0:
            logger.info("Epoch {}, {} step {}/{}: acc {:.3f}, loss {:.5f}, acc_robust {:.3f}, loss_robust {:.5f}".format(
                epoch, type, i + 1, len(batches),
                avg_acc.avg, avg_loss.avg, avg_acc_robust.avg, avg_loss_robust.avg
            ))        
    logger.info("Epoch {}, {}: acc {:.3f}, loss {:.5f}, acc_robust {:.3f}, loss_robust {:.5f}".format(
        epoch, type,
        avg_acc.avg, avg_loss.avg, avg_acc_robust.avg, avg_loss_robust.avg
    ))
    if args.res_file is not None:
        with open(args.res_file, "wb") as file:
            pickle.dump((avg_acc, avg_loss, avg_acc_robust, avg_loss_robust), file)

def main():
    if args.train:
        for t in range(model.checkpoint, args.num_epochs):
            train(t + 1)
            infer(t + 1, dev_batches, "dev")
            infer(t + 1, test_batches, "test")
    elif args.oracle:
        oracle(model, ptb, data_test, "test")
    else:
        infer(None, test_batches, "test")

if __name__ == "__main__":
    main()