import copy
import random
import sys
import os

import time

import torch.optim as optim
from torch.nn import CrossEntropyLoss

from examples.vision.argparser import argparser
from auto_LiRPA import BoundGeneral
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import AverageMeter
from examples.vision.config import load_config, get_path, config_modelloader, config_dataloader, update_dict
# from convex_adversarial import DualNetwork

Grad_accum = False
Grad_accum_step = 10


# sys.settrace(gpu_profile)

class Logger(object):
    def __init__(self, log_file=None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file=self.log_file)
            self.log_file.flush()


def Train(model, t, loader, start_eps, end_eps, max_eps, weights_eps_start, weights_eps_end, norm, logger, verbose,
          train, opt, method, **kwargs):
    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    num_class = 10
    losses = AverageMeter()
    l1_losses = AverageMeter()
    errors = AverageMeter()
    robust_errors = AverageMeter()
    regular_ce_losses = AverageMeter()
    robust_ce_losses = AverageMeter()
    batch_time = AverageMeter()
    # initial
    if train:
        model.train()
    else:
        model.eval()
    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 1), dtype=np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa)
    total = len(loader.dataset)
    batch_size = loader.batch_size
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    batch_eps = np.linspace(start_eps, end_eps, (total // batch_size) + 1)
    batch_weights_eps = np.zeros(((total // batch_size) + 1, len(weights_eps_start)))
    for _i in range(len(weights_eps_start)):
        batch_weights_eps[:, _i] = np.linspace(weights_eps_start[_i], weights_eps_end[_i], (total // batch_size) + 1)
    model_range = 0.0
    if batch_weights_eps[-1, 0] == 0:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"
    if train:
        opt.zero_grad()
    for i, (data, labels) in enumerate(loader):
        torch.cuda.empty_cache()
        if kwargs["bound_type"] == "weights-crown":
            data = data.reshape(data.shape[0], -1)
        start = time.time()
        eps = batch_eps[i]
        weights_eps = batch_weights_eps[i]
        # print(i, weights_eps, batch_eps)
        # if train:
        #     if not Grad_accum or i % Grad_accum_step == 0:
        #         # print('normal training without grad accsum')
        #         opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_class - 1, num_class))
        # scatter matrix to avoid compute margin to self
        sa_labels = sa[labels]
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(data.size(0), num_class)
        ub_s = torch.zeros(data.size(0), num_class)

        # FIXME: Assume data is from range 0 - 1
        if kwargs["bounded_input"]:
            assert loader.std == [1, 1, 1] or loader.std == [1]
            if norm != np.inf:
                raise ValueError("bounded input only makes sense for Linf perturbation. "
                                 "Please set the bounded_input option to false.")
            data_ub = (data + eps).clamp(max=1.0)
            data_lb = (data - eps).clamp(min=0.0)
        else:
            if norm == np.inf:
                data_ub = data + (eps / std)
                data_lb = data - (eps / std)
            else:
                data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data = data.cuda()
            data_ub = data_ub.cuda()
            data_lb = data_lb.cuda()
            labels = labels.cuda()
            c = c.cuda()
            sa_labels = sa_labels.cuda()
            lb_s = lb_s.cuda()
            ub_s = ub_s.cuda()
        # convert epsilon to a tensor
        eps_tensor = data.new(1)
        eps_tensor[0] = eps

        # omit the regular cross entropy, since we use robust error
        output = model(data)
        regular_ce = CrossEntropyLoss()(output, labels)
        regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0))
        errors.update(torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / data.size(0),
                      data.size(0))
        # get range statistic
        model_range = output.max().detach().cpu().item() - output.min().detach().cpu().item()

        if kwargs["bound_type"] == "weights-crown":
            ptb = PerturbationLpNorm_2bounds(norm=norm, eps=eps)
        else:
            ptb = PerturbationLpNorm(norm=norm, eps=eps)

        if verbose or method != "natural":
            if kwargs["bound_type"] == "interval":
                lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method=None)
            elif kwargs["bound_type"] == "crown-full":
                lb, ub = model.compute_bounds(ptb=ptb, IBP=False, x=data, C=c, method="backward")
            elif kwargs["bound_type"] == "weights-crown":
                lb, ub = model.weights_full_backward_range(ptb=ptb, norm=norm, x=data, C=c,
                                                           eps=eps, w_eps=weights_eps)
            elif kwargs["bound_type"] == "crown-interval":
                lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")
            else:
                raise RuntimeError("Unknown bound_type " + kwargs["bound_type"])

            lb = lb_s.scatter(1, sa_labels, lb)
            robust_ce = CrossEntropyLoss()(-lb, labels)

        if method == "robust":
            loss = robust_ce
        elif method == "natural":
            loss = regular_ce
        else:
            raise ValueError("Unknown method " + method)

        if "l1_reg" in kwargs:
            reg = kwargs["l1_reg"]
            l1_loss = 0.0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_loss = l1_loss + (reg * torch.sum(torch.abs(param)))
            loss = loss + l1_loss
            l1_losses.update(l1_loss.cpu().detach().numpy(), data.size(0))
        if train:
            loss.backward()
            if not Grad_accum or (i + 1) % Grad_accum_step == 0 or i == len(loader) - 1:
                opt.step()
                opt.zero_grad()

        losses.update(loss.cpu().detach().numpy(), data.size(0))

        if verbose or method != "natural":
            robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0))
            # robust_ce_losses.update(robust_ce, data.size(0))
            robust_errors.update(torch.sum((lb < 0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))

        avg_weights = model.choices[0].weight.data.cpu().numpy()

        batch_time.update(time.time() - start)
        if i % 50 == 0 and train:
            logger.log('[{:2d}:{:4d}]: eps {:4f}  '
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                       'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
                       'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
                       'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
                       'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
                       'Err {errors.val:.4f} ({errors.avg:.4f})  '
                       'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
                       'R {model_range:.3f}  '
                       'layer1 {avg_weights:.3f} ({range_weights:.3f})  '.format(
                t, i, eps, batch_time=batch_time,
                loss=losses, errors=errors, robust_errors=robust_errors, l1_loss=l1_losses,
                regular_ce_loss=regular_ce_losses, robust_ce_loss=robust_ce_losses,
                model_range=model_range, avg_weights=np.abs(avg_weights).mean(),
                range_weights=np.ptp(avg_weights)))

    # if Grad_accum and train:
    #     opt.step()

    logger.log('[FINAL RESULT epoch:{:2d} eps:{:.4f}]: '
               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
               'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
               'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
               'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
               'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
               'Err {errors.val:.4f} ({errors.avg:.4f})  '
               'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
               'R {model_range:.3f}  '
               'layer1 {avg_weights:.3f} ({range_weights:.3f})  \n'.format(
        t, eps, batch_time=batch_time,
        loss=losses, errors=errors, robust_errors=robust_errors, l1_loss=l1_losses,
        regular_ce_loss=regular_ce_losses, robust_ce_loss=robust_ce_losses,
        model_range=model_range, avg_weights=np.abs(avg_weights).mean(),
        range_weights=np.ptp(avg_weights)))
    # for i, l in enumerate(model.module()):
    #     if isinstance(l, BoundLinear) or isinstance(l, BoundConv2d):
    #         norm = l.weight.data.detach().view(l.weight.size(0), -1).abs().sum(1).max().cpu()
    #         logger.log('layer {} norm {}'.format(i, norm))
    if method == "natural":
        return errors.avg, errors.avg
    else:
        return robust_errors.avg, errors.avg


def main(args):
    config = load_config(args)
    global_train_config = config["training_params"]
    models, model_names = config_modelloader(config)

    # used for bound generally
    dummy_input = torch.randn(10, 784)
    # converted_models_ori = [BoundSequential.convert(model) for model in models]
    converted_models = [BoundGeneral(model, dummy_input) for model in models]

    # out_converted_model = converted_models_ori[0](dummy_input)
    # out_model = converted_models[0](dummy_input)
    # same = torch.sum(torch.abs(out_converted_model - out_model)) < 1e-5
    # print('Check output identity: {}\n'.format(same))
    # assert same

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)

    for model, model_id, model_config in zip(converted_models, model_names, config["models"]):
        # print(model.state_dict())
        # make a copy of global training config, and update per-model config
        train_config = copy.deepcopy(global_train_config)
        if "training_params" in model_config:
            train_config = update_dict(train_config, model_config["training_params"])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = DataParallel(model)
        model.to(device)

        # read training parameters from config file
        epochs = train_config["epochs"]
        lr = train_config["lr"]
        weight_decay = train_config["weight_decay"]
        starting_epsilon = train_config["starting_epsilon"]
        end_epsilon = train_config["epsilon"]
        schedule_length = train_config["schedule_length"]
        schedule_start = train_config["schedule_start"]
        optimizer = train_config["optimizer"]
        method = train_config["method"]
        verbose = train_config["verbose"]
        lr_decay_step = train_config["lr_decay_step"]
        lr_decay_factor = train_config["lr_decay_factor"]
        # parameters specific to a training method
        method_param = train_config["method_params"]
        norm = float(train_config["norm"])
        train_data, test_data = config_dataloader(config, **train_config["loader_params"])

        if optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=False, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer")

        eps_schedule = [0] * schedule_start + list(np.linspace(starting_epsilon, end_epsilon, schedule_length))
        weights_eps_schedule = np.zeros((schedule_length + schedule_start, len(train_config['epsilon_weights'])))
        # numpy 1.16+
        # weights_eps_schedule[schedule_start:] = np.linspace(starting_epsilon, train_config['epsilon_weights'], schedule_length)
        for _i in range(len(train_config['epsilon_weights'])):
            weights_eps_schedule[schedule_start:, _i] = np.linspace(starting_epsilon,
                                                                    train_config['epsilon_weights'][_i],
                                                                    schedule_length)
        max_eps = end_epsilon
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
        model_name = get_path(config, model_id, "model", load=False)
        best_model_name = get_path(config, model_id, "best_model", load=False)
        print(model_name)
        model_log = get_path(config, model_id, "train_log")
        logger = Logger(open(model_log, "w"))
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", train_config)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", train_data.std)
        logger.log("norm:", norm)

        best_err = np.inf
        recorded_clean_err = np.inf
        timer = 0.0
        for t in range(epochs):
            lr_scheduler.step(epoch=max(t - len(eps_schedule), 0))
            if t >= len(eps_schedule):
                eps = end_epsilon
                weights_eps = weights_eps_schedule[-1]
            else:
                weights_eps_start = weights_eps_schedule[t]
                epoch_start_eps = eps_schedule[t]
                if t + 1 >= len(eps_schedule):
                    epoch_end_eps = epoch_start_eps
                    weights_eps_end = weights_eps_start
                else:
                    epoch_end_eps = eps_schedule[t + 1]
                    weights_eps_end = weights_eps_schedule[t + 1]

            logger.log(
                "Epoch {}, learning rate {}, epsilon {:.6f} - {:.6f}".format(t, lr_scheduler.get_lr(),
                                                                             weights_eps_start[0],
                                                                             weights_eps_end[0]))
            # with torch.autograd.detect_anomaly():
            start_time = time.time()
            Train(model, t, train_data, epoch_start_eps, epoch_end_eps, max_eps, weights_eps_start, weights_eps_end,
                  norm, logger, verbose, True, opt,
                  method, **method_param)
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            torch.cuda.empty_cache()
            if t % 20 == 0 or t == epochs - 1 or t == schedule_start - 1:
                logger.log("Evaluating...")
                with torch.no_grad():
                    # evaluate
                    err, clean_err = Train(model, t, test_data, epoch_end_eps, epoch_end_eps, max_eps, weights_eps_end,
                                           weights_eps_end, norm, logger,
                                           verbose, False, None, method, **method_param)

            logger.log('saving to', model_name)
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': t,
            }, model_name)

            # save the best model after we reached the schedule
            if t >= len(eps_schedule):
                if err <= best_err:
                    best_err = err
                    recorded_clean_err = clean_err
                    logger.log('Saving best model {} with error {}'.format(best_model_name, best_err))
                    torch.save({
                        'state_dict': model.state_dict(),
                        'robust_err': err,
                        'clean_err': clean_err,
                        'epoch': t,
                    }, best_model_name)

        logger.log('Total Time: {:.4f}'.format(timer))
        logger.log('Model {} best err {}, clean err {}'.format(model_id, best_err, recorded_clean_err))


if __name__ == "__main__":
    args = argparser()
    main(args)
