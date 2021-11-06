import random
import time
import argparse
import multiprocessing
import logging
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.bound_ops import BoundExp
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter, logger, get_spec_matrix
import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from auto_LiRPA.eps_scheduler import *

def get_exp_module(bounded_module):
    for _, node in bounded_module.named_modules():
        # Find the Exp neuron in computational graph
        if isinstance(node, BoundExp):
            return node
    return None

parser = argparse.ArgumentParser()

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data_dir", type=str, default="data/ImageNet64",
                    help='dir of dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=1. / 255, help='Target training epsilon')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="wide_resnet_imagenet64_1000class",
                    help='model name (mlp_3layer, cnn_4layer, cnn_6layer, cnn_7layer, resnet)')
parser.add_argument("--num_epochs", type=int, default=240, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=125, help='batch size')
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
parser.add_argument("--lr_decay_milestones", nargs='+', type=int, default=[200, 220], help='learning rate dacay milestones')
parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler"], help='epsilon scheduler')
parser.add_argument("--scheduler_opts", type=str, default="start=100,length=80", help='options for epsilon scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options')
parser.add_argument('--clip_grad_norm', type=float, default=8.0)
parser.add_argument('--in_planes', type=int, default=16)
parser.add_argument('--widen_factor', type=int, default=10)

args = parser.parse_args()

exp_name = args.model + '_b' + str(args.batch_size) + '_' + str(args.bound_type) + '_epoch' + str(
    args.num_epochs) + '_' + args.scheduler_opts + '_' + str(args.eps)[:6]
log_file = f'saved_models/{exp_name}{"_test" if args.verify else ""}.log'
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler) 

def Train(model, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust', loss_fusion=True,
          final_node_name=None):
    num_class = 1000
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    exp_module = get_exp_module(model)

    def get_bound_loss(x=None, c=None):
        if loss_fusion:
            bound_lower, bound_upper = False, True
        else:
            bound_lower, bound_upper = True, False

        if bound_type == 'IBP':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                           final_node_name=final_node_name, no_replicas=True)
        elif bound_type == 'CROWN':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=False, C=c, method='backward',
                           bound_lower=bound_lower, bound_upper=bound_upper)
        elif bound_type == 'CROWN-IBP':
            # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method='backward')  # pure IBP bound
            # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
            factor = (eps_scheduler.get_max_eps() - eps_scheduler.get_eps()) / eps_scheduler.get_max_eps()
            ilb, iub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                             final_node_name=final_node_name, no_replicas=True)
            if factor < 1e-50:
                lb, ub = ilb, iub
            else:
                clb, cub = model(method_opt="compute_bounds", IBP=False, C=c, method='backward',
                                 bound_lower=bound_lower, bound_upper=bound_upper, final_node_name=final_node_name,
                                 no_replicas=True)
                if loss_fusion:
                    ub = cub * factor + iub * (1 - factor)
                else:
                    lb = clb * factor + ilb * (1 - factor)

        if loss_fusion:
            if isinstance(model, BoundDataParallel):
                max_input = model(get_property=True, node_class=BoundExp, att_name='max_input')
            else:
                max_input = exp_module.max_input
            return None, torch.mean(torch.log(ub) + max_input)
        else:
            # Pad zero at the beginning for each example, and use fake label '0' for all examples
            lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
            return lb, robust_ce

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-50:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        # bound input for Linf norm used only
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1, -1, 1, 1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1, -1, 1, 1), data_min)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels = data.cuda(), labels.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb)
        if loss_fusion:
            if batch_method == 'natural' or not train:
                output = model(x, labels)
                regular_ce = torch.mean(torch.log(output))
            else:
                model(x, labels)
                regular_ce = torch.tensor(0., device=data.device)
            meter.update('CE', regular_ce.item(), x.size(0))
            x = (x, labels)
            c = None
        else:
            c = get_spec_matrix(data, labels, num_class)
            x = (x, labels)
            output = model(x, final_node_name=final_node_name)
            regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
            meter.update('CE', regular_ce.item(), x[0].size(0))
            meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).item() / x[0].size(0), x[0].size(0))

        if batch_method == 'robust':
            # print(data.sum())
            lb, robust_ce = get_bound_loss(x=x, c=c)
            loss = robust_ce
        elif batch_method == 'natural':
            loss = regular_ce

        if train:
            loss.backward()

            if args.clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                meter.update('grad_norm', grad_norm)

            if isinstance(eps_scheduler, AdaptiveScheduler):
                eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))

        if batch_method != 'natural':
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            if not loss_fusion:
                # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
                # If any margin is < 0 this example is counted as an error
                meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)

        if (i + 1) % 500 == 0 and train:
            logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))

    logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))
    return meter


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Step 1: Initial original model as usual, see model details in models/example_feedforward.py and models/example_resnet.py
    model_ori = models.Models[args.model](in_planes=args.in_planes, widen_factor=args.widen_factor)
    epoch = 0
    if args.load:
        checkpoint = torch.load(args.load)
        epoch, state_dict, opt_state = checkpoint['epoch'], checkpoint['state_dict'], checkpoint.get('optimizer')
        for k, v in state_dict.items():
            assert torch.isnan(v).any().cpu().numpy() == 0 and torch.isinf(v).any().cpu().numpy() == 0
        model_ori.load_state_dict(state_dict)
        logger.info('Checkpoint loaded: {}'.format(args.load))

    ## Step 2: Prepare dataset as usual
    dummy_input = torch.randn(2, 3, 56, 56)
    normalize = transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2153, 0.2111, 0.2121])
    train_data = datasets.ImageFolder(args.data_dir + '/train',
                                      transform=transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(56, padding_mode='edge'),
                                          transforms.ToTensor(),
                                          normalize,
                                      ]))
    test_data = datasets.ImageFolder(args.data_dir + '/test',
                                     transform=transforms.Compose([
                                         # transforms.RandomResizedCrop(64, scale=(0.875, 0.875), ratio=(1., 1.)),
                                         transforms.CenterCrop(56),
                                         transforms.ToTensor(),
                                         normalize]))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                             num_workers=min(multiprocessing.cpu_count(), 4))
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size // 4, pin_memory=True,
                                            num_workers=min(multiprocessing.cpu_count(), 4))
    train_data.mean = test_data.mean = torch.tensor([0.4815, 0.4578, 0.4082])
    train_data.std = test_data.std = torch.tensor([0.2153, 0.2111, 0.2121])

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    model = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
    model_loss = BoundedModule(CrossEntropyWrapper(model_ori), (dummy_input, torch.zeros(1, dtype=torch.long)),
                               bound_opts= { 'relu': args.bound_opts, 'loss_fusion': True }, device=args.device)
    model_loss = BoundDataParallel(model_loss)

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    opt = optim.Adam(model_loss.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_decay_milestones, gamma=0.1)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)
    logger.info(str(model_ori))

    if args.load:
        if opt_state:
            opt.load_state_dict(opt_state)
            logger.info('resume opt_state')

    # skip epochs
    if epoch > 0:
        epoch_length = int((len(train_data.dataset) + train_data.batch_size - 1) / train_data.batch_size)
        eps_scheduler.set_epoch_length(epoch_length)
        eps_scheduler.train()
        for i in range(epoch):
            lr_scheduler.step()
            eps_scheduler.step_epoch(verbose=True)
            for j in range(epoch_length):
                eps_scheduler.step_batch()
        logger.info('resume from eps={:.12f}'.format(eps_scheduler.get_eps()))

    ## Step 5: start training
    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            Train(model, 1, test_data, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False, final_node_name=None)
    else:
        timer = 0.0
        best_err = 1e10
        # with torch.autograd.detect_anomaly():
        for t in range(epoch + 1, args.num_epochs + 1):
            logger.info("Epoch {}, learning rate {}".format(t, lr_scheduler.get_last_lr()))
            start_time = time.time()
            Train(model_loss, t, train_data, eps_scheduler, norm, True, opt, args.bound_type, loss_fusion=True)
            lr_scheduler.step()
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.info('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))

            logger.info("Evaluating...")
            torch.cuda.empty_cache()

            # remove 'model.' in state_dict
            state_dict_loss = model_loss.state_dict()
            state_dict = {}
            for name in state_dict_loss:
                assert (name.startswith('model.'))
                state_dict[name[6:]] = state_dict_loss[name]

            with torch.no_grad():
                if int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']) > t >= int(
                        eps_scheduler.params['start']):
                    m = Train(model_loss, t, test_data, eps_scheduler, norm, False, None, args.bound_type, loss_fusion=True)
                else:
                    model_ori.load_state_dict(state_dict)
                    model = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)
                    model = BoundDataParallel(model)
                    m = Train(model, t, test_data, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False)
                    del model

            save_dict = {'state_dict': state_dict, 'epoch': t, 'optimizer': opt.state_dict()}
            if t < int(eps_scheduler.params['start']):
                torch.save(save_dict, 'saved_models/natural_' + exp_name)
            elif t > int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']):
                current_err = m.avg('Verified_Err')
                if current_err < best_err:
                    best_err = current_err
                    torch.save(save_dict, 'saved_models/' + exp_name + '_best_' + str(best_err)[:6])
            else:
                torch.save(save_dict, 'saved_models/' + exp_name)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    logger.info(args)
    main(args)
