import argparse, random, pickle, os, pdb, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from auto_LiRPA.utils import AverageMeter, logger, scale_gradients
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, PerturbationSynonym
from Transformer.BERT import BERT
from lstm import LSTM
from data_utils import load_data, clean_data, get_batches
from oracle import oracle

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true')
parser.add_argument('--robust', action='store_true')
parser.add_argument('--oracle', action='store_true')
parser.add_argument('--dir', type=str, default='model')
parser.add_argument('--checkpoint', type=int, default=None)
parser.add_argument('--data', type=str, default='sst', choices=['sst', 'imdb'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

parser.add_argument('--ptb', type=str, default='synonym', 
                    choices=['synonym'])
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--budget', type=int, default=3)
parser.add_argument('--ibp', action='store_true')
parser.add_argument('--kappa', type=float, default=0.8)
parser.add_argument('--check', action='store_true')
parser.add_argument('--method', type=str, default=None,
                    choices=[None, 'forward', 'backward'])
parser.add_argument('--res_file', type=str, default=None)

parser.add_argument('--model', type=str, default='transformer',
                    choices=['transformer', 'lstm'])
parser.add_argument('--num_epochs', type=int, default=20)  
parser.add_argument('--num_epochs_all_nodes', type=int, default=5)      
parser.add_argument('--num_epochs_warmup', type=int, default=5) 
parser.add_argument('--num_epochs_pretrain', type=int, default=0) 
parser.add_argument('--use_crown_ibp', action='store_true')
parser.add_argument('--use_crown_for_logging', action='store_true')
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--min_word_freq', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--oracle_batch_size', type=int, default=1024)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--max_sent_length', type=int, default=32)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=1)
parser.add_argument('--grad_clip', type=float, default=None)
parser.add_argument('--num_labels', type=int, default=2) 
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=64) # 256
parser.add_argument('--embedding_size', type=int, default=64) # 256
parser.add_argument('--intermediate_size', type=int, default=128) # 512
parser.add_argument('--use_glove', action='store_true')
parser.add_argument('--glove', type=str, default='data/glove/glove.6B.100d.txt')
parser.add_argument('--drop_unk', action='store_true')
parser.add_argument('--hidden_act', type=str, default='relu')
parser.add_argument('--layer_norm', type=str, default='no_var',
                    choices=['standard', 'no', 'no_var'])
parser.add_argument('--use_simple_concretization', action='store_true')

args = parser.parse_args()   

def build_perturbation():
    if args.ptb == 'lp_norm':
        return PerturbationLpNorm(norm=np.inf, eps=args.eps)    
    elif args.ptb == 'synonym':
        return PerturbationSynonym(budget=args.budget, use_simple=args.use_simple_concretization)
    else:
        raise NotImplementedError
        
def convert(model, ptb, batch):
    model.train()
    embeddings, mask, _, _ = model.get_input(batch)
    embeddings = BoundedTensor(embeddings, ptb)
    converted_model = BoundedModule(
        model.model_from_embeddings, (embeddings, mask), device=args.device)    
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

    loss_fct = nn.CrossEntropyLoss()

    with grad:
        ptb.set_eps(eps)
        ptb.set_train(train)
        embeddings_unbounded, mask, tokens, label_ids = model.get_input(batch)
        if args.robust and eps > 1e-9:
            embeddings = BoundedTensor(embeddings_unbounded, ptb)
        else:
            embeddings = embeddings_unbounded.detach().requires_grad_(True)

        logits = model_bound(embeddings, mask)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum((preds == label_ids).to(torch.float32)) / len(batch)
        loss = loss_fct(logits, label_ids)
        loss_all = loss

        if args.robust and eps > 1e-6:
            if args.use_crown_ibp:
                if not train:
                    logits_robust = model_bound.compute_worst_logits(
                        y=label_ids, aux=(tokens, batch), IBP=True, method=args.method)
                elif 1 - eps < 1e-3:
                    logits_robust = model_bound.compute_worst_logits(
                        y=label_ids, aux=(tokens, batch), IBP=True, method=None)
                else:
                    logits_robust_linear = model_bound.compute_worst_logits(
                        y=label_ids, aux=(tokens, batch), IBP=True, method='backward')
                    logits_robust_ibp = model_bound.compute_worst_logits(
                        y=label_ids, aux=(tokens, batch), IBP=True, method=None, reuse_ibp=True)   
                    logits_robust =  eps * logits_robust_ibp + (1 - eps) * logits_robust_linear
                preds_robust = torch.argmax(logits_robust, dim=1)
                acc_robust = torch.sum((preds_robust == label_ids).to(torch.float32)) / len(batch)
                loss_robust = loss_fct(logits_robust, label_ids)
                loss_all = (1 - args.kappa) * loss_all + args.kappa * loss_robust    
            elif args.use_crown_for_logging:
                logits_robust_linear = model_bound.compute_worst_logits(
                    y=label_ids, aux=(tokens, batch), IBP=True, method='backward')
                logits_robust_ibp = model_bound.compute_worst_logits(
                    y=label_ids, aux=(tokens, batch), IBP=True, method=None, reuse_ibp=True) 
                preds_robust = torch.argmax(logits_robust_linear, dim=1)
                acc_robust = torch.sum((preds_robust == label_ids).to(torch.float32)) / len(batch)
                if args.method == 'backward':
                    logits_robust = logits_robust_linear
                elif args.method is None:
                    logits_robust = logits_robust_ibp
                else:
                    raise NotImplementedError
                loss_robust = loss_fct(logits_robust, label_ids)
                loss_all = (1 - args.kappa) * loss_all + args.kappa * loss_robust    
            else:
                logits_robust = model_bound.compute_worst_logits(
                    y=label_ids, aux=(tokens, batch), IBP=args.ibp, method=args.method)
                preds_robust = torch.argmax(logits_robust, dim=1)
                acc_robust = torch.sum((preds_robust == label_ids).to(torch.float32)) / len(batch)
                loss_robust = loss_fct(logits_robust, label_ids)
                loss_all = (1 - args.kappa) * loss_all + args.kappa * loss_robust    
            acc, loss = acc.detach(), loss.detach()
            acc_robust, loss_robust = acc_robust.detach(), loss_robust.detach()
        else:
            acc_robust, loss_robust =  0., 0.

    if train:
        loss_all.backward()
        grad_embed = torch.autograd.grad(
            embeddings_unbounded, model.word_embeddings.weight, 
            grad_outputs=embeddings.grad)[0]
        if model.word_embeddings.weight.grad is None:
            model.word_embeddings.weight.grad = grad_embed
        else:
            model.word_embeddings.weight.grad += grad_embed

    return acc, loss, acc_robust, loss_robust

data_train_all_nodes, data_train, data_dev, data_test = load_data(args.data)
if args.robust:
    data_dev, data_test = clean_data(data_dev), clean_data(data_test)
logger.info('Dataset sizes: {}/{}/{}/{}'.format(
    len(data_train_all_nodes), len(data_train), len(data_dev), len(data_test)))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.model == 'transformer':
    model = BERT(args, data_train)
elif args.model == 'lstm':
    model = LSTM(args, data_train)

dev_batches = get_batches(data_dev, args.batch_size)
test_batches = get_batches(data_test, args.batch_size)        

ptb = build_perturbation()
convert(model, ptb, dev_batches[0])
ptb.model = model
optimizer = model.build_optimizer()
if args.lr_decay < 1:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay)
else:
    lr_scheduler = None
logger.info('Model converted to support bounds')

avg_acc, avg_loss, avg_acc_robust, avg_loss_robust = [AverageMeter() for i in range(4)]
writer = SummaryWriter(os.path.join(args.dir, 'log'), flush_secs=10)

def train(epoch):
    assert(optimizer is not None)
    if epoch <= args.num_epochs_all_nodes:
        train_batches = get_batches(data_train_all_nodes, args.batch_size)
    else:
        train_batches = get_batches(data_train, args.batch_size)
    avg_acc.reset()
    avg_loss.reset()
    avg_acc_robust.reset()
    avg_loss_robust.reset()       
    if args.robust and args.num_epochs_warmup > 0:
        eps_inc_per_step = 1.0 / (args.num_epochs_warmup * len(train_batches))
    else:
        eps_inc_per_step = 1.0
    for i, batch in enumerate(train_batches):
        if not args.robust or epoch <= args.num_epochs_pretrain:
            eps = 0
        else:
            eps = min(eps_inc_per_step * \
                ((epoch - args.num_epochs_pretrain - 1) * len(train_batches) + i + 1), 1.0)         
        acc, loss, acc_robust, loss_robust = \
            step(model, ptb, batch, eps=eps, train=True)
        avg_acc.update(acc, len(batch))
        avg_loss.update(loss, len(batch))
        avg_acc_robust.update(acc_robust, len(batch))
        avg_loss_robust.update(loss_robust, len(batch))   
        if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_batches):
            scale_gradients(optimizer, i % args.gradient_accumulation_steps + 1, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()    
        if lr_scheduler is not None:
            lr_scheduler.step()                    
        if (i + 1) % args.log_interval == 0:
            logger.info('Epoch {}, training step {}/{}: acc {:.3f}, loss {:.3f}, acc_robust {:.3f}, loss_robust {:.3f}, eps {:.3f}'.format(
                epoch, i + 1, len(train_batches),
                avg_acc.avg, avg_loss.avg, avg_acc_robust.avg, avg_loss_robust.avg,
                eps
            ))
            if lr_scheduler is not None:
                logger.info('lr {}'.format(lr_scheduler.get_lr()))
            writer.add_scalar('loss_train_{}'.format(epoch), avg_loss.avg, i + 1)
            writer.add_scalar('loss_robust_train_{}'.format(epoch), avg_loss_robust.avg, i + 1)
            writer.add_scalar('acc_train_{}'.format(epoch), avg_acc.avg, i + 1)
            writer.add_scalar('acc_robust_train_{}'.format(epoch), avg_acc_robust.avg, i + 1)
    writer.add_scalar('loss/train', avg_loss.avg, epoch)
    writer.add_scalar('loss_robust/train', avg_loss_robust.avg, epoch)
    writer.add_scalar('acc/train', avg_acc.avg, epoch)
    writer.add_scalar('acc_robust/train', avg_acc_robust.avg, epoch)

    model.save(epoch)

def infer(epoch, batches, type):
    avg_acc.reset()
    avg_loss.reset()
    avg_acc_robust.reset()
    avg_loss_robust.reset()    
    for i, batch in enumerate(batches):
        acc, loss, acc_robust, loss_robust = step(model, ptb, batch, eps=args.eps)
        avg_acc.update(acc, len(batch))
        avg_loss.update(loss, len(batch)) 
        avg_acc_robust.update(acc_robust, len(batch))
        avg_loss_robust.update(loss_robust, len(batch))                   
        if (i + 1) % args.log_interval == 0:
            logger.info('Epoch {}, {} step {}/{}: acc {:.3f}, loss {:.5f}, acc_robust {:.3f}, loss_robust {:.5f}'.format(
                epoch, type, i + 1, len(batches),
                avg_acc.avg, avg_loss.avg, avg_acc_robust.avg, avg_loss_robust.avg
            ))        
    logger.info('Epoch {}, {}: acc {:.3f}, loss {:.5f}, acc_robust {:.3f}, loss_robust {:.5f}'.format(
        epoch, type,
        avg_acc.avg, avg_loss.avg, avg_acc_robust.avg, avg_loss_robust.avg
    ))
    if epoch is not None:
        writer.add_scalar('loss/{}'.format(type), avg_loss.avg, epoch)
        writer.add_scalar('loss_robust/{}'.format(type), avg_loss_robust.avg, epoch)
        writer.add_scalar('acc/{}'.format(type), avg_acc.avg, epoch)
        writer.add_scalar('acc_robust/{}'.format(type), avg_acc_robust.avg, epoch)    
    if args.res_file is not None:
        with open(args.res_file, 'wb') as file:
            pickle.dump((avg_acc, avg_loss, avg_acc_robust, avg_loss_robust), file)

def main():
    if args.train:
        for t in range(model.checkpoint, args.num_epochs):
            train(t + 1)
            infer(t + 1, dev_batches, 'dev')
            infer(t + 1, test_batches, 'test')
    elif args.oracle:
        oracle(args, model, ptb, data_test, 'test')
    else:
        infer(None, test_batches, 'test')

if __name__ == '__main__':
    main()
