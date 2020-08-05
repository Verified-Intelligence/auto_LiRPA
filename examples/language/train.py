import argparse
import random
import pickle
import os
import pdb
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationSynonym, CrossEntropyWrapperMultiInput
from auto_LiRPA.utils import MultiAverageMeter, AverageMeter, logger, scale_gradients
from auto_LiRPA.eps_scheduler import *
from Transformer.Transformer import Transformer
from lstm import LSTM
from data_utils import load_data, clean_data, get_batches
from oracle import oracle

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true')
parser.add_argument('--robust', action='store_true')
parser.add_argument('--oracle', action='store_true')
parser.add_argument('--dir', type=str, default='model')
parser.add_argument('--checkpoint', type=int, default=None)
parser.add_argument('--data', type=str, default='sst', choices=['sst'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--legacy_loading', action='store_true', help='use a deprecated way of loading checkpoints for previously saved models')
parser.add_argument('--auto_test', action='store_true')

parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--budget', type=int, default=6)
parser.add_argument('--method', type=str, default=None,
                    choices=['IBP', 'IBP+backward', 'IBP+backward_train', 'forward', 'forward+backward'])

parser.add_argument('--model', type=str, default='transformer',
                    choices=['transformer', 'lstm'])
parser.add_argument('--num_epochs', type=int, default=25)  
parser.add_argument('--num_epochs_all_nodes', type=int, default=20)      
parser.add_argument('--eps_start', type=int, default=1)
parser.add_argument('--eps_length', type=int, default=10)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--min_word_freq', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--oracle_batch_size', type=int, default=1024)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--max_sent_length', type=int, default=32)
parser.add_argument('--vocab_size', type=int, default=50000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=1)
parser.add_argument('--grad_clip', type=float, default=10.0)
parser.add_argument('--num_classes', type=int, default=2) 
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--embedding_size', type=int, default=64) 
parser.add_argument('--intermediate_size', type=int, default=128)
parser.add_argument('--drop_unk', action='store_true')
parser.add_argument('--hidden_act', type=str, default='relu')
parser.add_argument('--layer_norm', type=str, default='no_var',
                    choices=['standard', 'no', 'no_var'])
parser.add_argument('--loss_fusion', action='store_true')
parser.add_argument('--no_lm', action='store_true', help='no language model constraint')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--bound_opts_relu', type=str, default='zero-lb')

args = parser.parse_args()   

writer = SummaryWriter(os.path.join(args.dir, 'log'), flush_secs=10)
file_handler = logging.FileHandler(os.path.join(args.dir, 'log/train.log'))
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler)

data_train_all_nodes, data_train, data_dev, data_test = load_data(args.data)
if args.robust:
    data_dev, data_test = clean_data(data_dev), clean_data(data_test)
if args.auto_test:
    random.seed(args.seed)
    random.shuffle(data_test)
    data_test = data_test[:10]
    assert args.batch_size >= 10
logger.info('Dataset sizes: {}/{}/{}/{}'.format(
    len(data_train_all_nodes), len(data_train), len(data_dev), len(data_test)))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

dummy_embeddings = torch.zeros(1, args.max_sent_length, args.embedding_size, device=args.device)
dummy_labels = torch.zeros(1, dtype=torch.long, device=args.device)

if args.model == 'transformer':
    dummy_mask = torch.zeros(1, 1, 1, args.max_sent_length, device=args.device)
    model = Transformer(args, data_train)
elif args.model == 'lstm':
    dummy_mask = torch.zeros(1, args.max_sent_length, device=args.device)
    model = LSTM(args, data_train)

if args.no_lm:
    # no language model constraint for test data
    for i in range(len(data_test)):
        data_test[i]['candidates'] = None
    
dev_batches = get_batches(data_dev, args.batch_size)
test_batches = get_batches(data_test, args.batch_size)

ptb = PerturbationSynonym(budget=args.budget)
dummy_embeddings = BoundedTensor(dummy_embeddings, ptb)
model_ori = model.model_from_embeddings
bound_opts = { 'relu': args.bound_opts_relu, 'exp': 'no-max-input' }
if isinstance(model_ori, BoundedModule):
    model_bound = model_ori
else:
    model_bound = BoundedModule(
        model_ori, (dummy_embeddings, dummy_mask), bound_opts=bound_opts, device=args.device)
model.model_from_embeddings = model_bound
if args.loss_fusion:
    bound_opts['loss_fusion'] = True
    model_loss = BoundedModule(
        CrossEntropyWrapperMultiInput(model_ori), 
        (torch.zeros(1, dtype=torch.long), dummy_embeddings, dummy_mask), 
        bound_opts=bound_opts, device=args.device)

ptb.model = model
optimizer = model.build_optimizer()
if args.lr_decay < 1:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay)
else:
    lr_scheduler = None
if args.robust:
    eps_scheduler = LinearScheduler(args.eps, 'start={},length={}'.format(args.eps_start, args.eps_length))
    for i in range(model.checkpoint):
        eps_scheduler.step_epoch(verbose=False)
else:
    eps_scheduler = None
logger.info('Model converted to support bounds')

def step(model, ptb, batch, eps=1.0, train=False):
    model_bound = model.model_from_embeddings
    if train:
        model.train()
        model_bound.train()
        grad = torch.enable_grad()
        if args.loss_fusion:
            model_loss.train()
    else:
        model.eval()
        model_bound.eval()
        grad = torch.no_grad()
    if args.auto_test:
        grad = torch.enable_grad()

    with grad:
        ptb.set_eps(eps)
        ptb.set_train(train)
        embeddings_unbounded, mask, tokens, labels = model.get_input(batch)
        aux = (tokens, batch)
        if args.robust and eps > 1e-9:
            embeddings = BoundedTensor(embeddings_unbounded, ptb)
        else:
            embeddings = embeddings_unbounded.detach().requires_grad_(True)

        robust = args.robust and eps > 1e-6

        if train and robust and args.loss_fusion:
            # loss_fusion loss
            if args.method == 'IBP+backward_train':
                lb, ub = model_loss.compute_bounds(
                    x=(labels, embeddings, mask), aux=aux, 
                    IBP=True, C=None, method='backward', bound_lower=False)
            else:
                raise NotImplementedError
            loss_robust = torch.log(ub).mean()
            loss = acc = acc_robust = -1 # unknown
        else:
            # regular loss
            logits = model_bound(embeddings, mask)
            loss = CrossEntropyLoss()(logits, labels)
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

            if robust:
                num_class = args.num_classes
                c = torch.eye(num_class).type_as(embeddings)[labels].unsqueeze(1) - \
                    torch.eye(num_class).type_as(embeddings).unsqueeze(0)
                I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
                c = (c[I].view(embeddings.size(0), num_class - 1, num_class))
                if args.method == 'IBP':
                    lb, ub = model_bound.compute_bounds(aux=aux, IBP=True, C=c, method=None)
                elif args.method == 'IBP+backward':
                    lb, ub = model_bound.compute_bounds(aux=aux, IBP=True, C=c, method='backward', bound_upper=False)
                elif args.method == 'IBP+backward_train':
                    if 1 - eps > 1e-4:
                        lb, ub = model_bound.compute_bounds(aux=aux, IBP=True, C=c, method='backward', bound_upper=False)
                        ilb, iub = model_bound.compute_bounds(aux=aux, IBP=True, C=c, method=None, reuse_ibp=True)
                        lb = eps * ilb + (1 - eps) * lb
                    else:
                        lb, ub = model_bound.compute_bounds(aux=aux, IBP=True, C=c, method=None)
                elif args.method == 'forward':
                    lb, ub = model_bound.compute_bounds(aux=aux, IBP=False, C=c, method='forward', bound_upper=False)
                elif args.method == 'forward+backward':
                    lb, ub = model_bound.compute_bounds(aux=aux, IBP=False, forward=True, C=c, method='backward', bound_upper=False)
                else:
                    raise NotImplementedError
                lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
                fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
                loss_robust = robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
                acc_robust = 1 - torch.mean((lb < 0).any(dim=1).float())
            else:
                acc_robust, loss_robust = acc, loss
                
    if train or args.auto_test:
        loss_robust.backward()
        grad_embed = torch.autograd.grad(
            embeddings_unbounded, model.word_embeddings.weight, 
            grad_outputs=embeddings.grad)[0]
        if model.word_embeddings.weight.grad is None:
            model.word_embeddings.weight.grad = grad_embed
        else:
            model.word_embeddings.weight.grad += grad_embed

    if args.auto_test:
        with open('res_test.pkl', 'wb') as file:
            pickle.dump((
                float(acc), float(loss), float(acc_robust), float(loss_robust), 
                grad_embed.detach().numpy()), file)

    return acc, loss, acc_robust, loss_robust

def train(epoch, batches, type):
    meter = MultiAverageMeter()
    assert(optimizer is not None) 
    train = type == 'train'
    if args.robust:
        eps_scheduler.set_epoch_length(len(batches))    
        if train:
            eps_scheduler.train()
            eps_scheduler.step_epoch()
        else:
            eps_scheduler.eval()
    for i, batch in enumerate(batches):
        if args.robust:
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
        else:
            eps = 0
        acc, loss, acc_robust, loss_robust = \
            step(model, ptb, batch, eps=eps, train=train)
        meter.update('acc', acc, len(batch))
        meter.update('loss', loss, len(batch))
        meter.update('acc_rob', acc_robust, len(batch))
        meter.update('loss_rob', loss_robust, len(batch))
        if train:
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(batches):
                scale_gradients(optimizer, i % args.gradient_accumulation_steps + 1, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()    
            if lr_scheduler is not None:
                lr_scheduler.step()                    
            writer.add_scalar('loss_train_{}'.format(epoch), meter.avg('loss'), i + 1)
            writer.add_scalar('loss_robust_train_{}'.format(epoch), meter.avg('loss_rob'), i + 1)
            writer.add_scalar('acc_train_{}'.format(epoch), meter.avg('acc'), i + 1)
            writer.add_scalar('acc_robust_train_{}'.format(epoch), meter.avg('acc_rob'), i + 1)
        if (i + 1) % args.log_interval == 0 or (i + 1) == len(batches):
            logger.info('Epoch {}, {} step {}/{}: eps {:.5f}, {}'.format(
                epoch, type, i + 1, len(batches), eps, meter))
            if lr_scheduler is not None:
                logger.info('lr {}'.format(lr_scheduler.get_lr()))
    writer.add_scalar('loss/{}'.format(type), meter.avg('loss'), epoch)
    writer.add_scalar('loss_robust/{}'.format(type), meter.avg('loss_rob'), epoch)
    writer.add_scalar('acc/{}'.format(type), meter.avg('acc'), epoch)
    writer.add_scalar('acc_robust/{}'.format(type), meter.avg('acc_rob'), epoch)

    if train:
        if args.loss_fusion:
            state_dict_loss = model_loss.state_dict() 
            state_dict = {}
            for name in state_dict_loss:
                assert(name.startswith('model.'))
                state_dict[name[6:]] = state_dict_loss[name]
            model_ori.load_state_dict(state_dict)
            model_bound = BoundedModule(
                model_ori, (dummy_embeddings, dummy_mask), bound_opts=bound_opts, device=args.device)
            model.model_from_embeddings = model_bound        
        model.save(epoch)

    return meter.avg('acc_rob')

def main():
    if args.train:
        for t in range(model.checkpoint, args.num_epochs):
            if t + 1 <= args.num_epochs_all_nodes:
                train(t + 1, get_batches(data_train_all_nodes, args.batch_size), 'train')
            else:
                train(t + 1, get_batches(data_train, args.batch_size), 'train')
            train(t + 1, dev_batches, 'dev')
            train(t + 1, test_batches, 'test')
    elif args.oracle:
        oracle(args, model, ptb, data_test, 'test')
    else:
        if args.robust:
            for i in range(args.num_epochs):
                eps_scheduler.step_epoch(verbose=False)
            res = []
            for i in range(1, args.budget + 1):
                logger.info('budget {}'.format(i))
                ptb.budget = i
                acc_rob = train(None, test_batches, 'test')
                res.append(acc_rob)
            logger.info('Verification results:')
            for i in range(len(res)):
                logger.info('budget {} acc_rob {:.3f}'.format(i + 1, res[i]))                
            logger.info(res)
        else:
            train(None, test_batches, 'test')


if __name__ == '__main__':
    main()
