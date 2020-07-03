import random
import sys
import time
import argparse
import multiprocessing
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from torch.utils.tensorboard import SummaryWriter
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from autoLiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

torch.cuda.set_device(2)

# class ensemble_MNIST(nn.Module):
#     def __init__(self):
#         super(ensemble_MNIST, self).__init__()
#         self.model_1 = models.cnn_6layer(1, 28)
#         self.model_2 = models.cnn_4layer(1, 28)
#         # self.fc1 = nn.Linear(784, 64, bias=True)
#         # self.fc2 = nn.Linear(64, 64, bias=True)
#         # self.fc3 = nn.Linear(64, 10, bias=True)

#         # # Add perturbation on the weights/bias of fc1/fc2
#         # self.ptb_w = PerturbationLpNorm(norm=np.inf, eps=0.)
#         # self.fc1.weight = BoundedParameter(self.fc1.weight.data, self.ptb_w)
#         # # self.fc1.bias = BoundedParameter(self.fc1.bias.data, ptb)
#         # self.fc2.weight = BoundedParameter(self.fc2.weight.data, self.ptb_w)
#         # self.fc3.weight = BoundedParameter(self.fc3.weight.data, self.ptb_w)
    
#     def load(self, path1, path2):
#         state_dict = torch.load(path1)['state_dict']
#         self.model_1.load_state_dict(state_dict)

#         state_dict = torch.load(path2)['state_dict']
#         self.model_2.load_state_dict(state_dict)

#     def forward(self, x):
#         # x_flatten = x.view(-1, 784)

#         x1 = self.model_1(x)
#         x2 = self.model_2(x)
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         multi = torch.full_like(x1, 0.5)
#         return (x1 + x2) * multi

parser = argparse.ArgumentParser()

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="MNIST", choices=["MNIST", "CIFAR"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=0.3, help='Target training epsilon')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="cnn_4layer", help='model name (mlp_3layer, cnn_4layer, cnn_6layer, cnn_7layer, resnet)')
parser.add_argument("--num_epochs", type=int, default=100, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--eps_scheduler_name", type=str, default="LinearScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler"], help='epsilon scheduler')
parser.add_argument("--crown_scheduler_name", type=str, default="LinearScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler"], help='crown-ibp scheduler')
parser.add_argument("--eps_scheduler_opts", type=str, default="start=2,length=60", help='options for epsilon scheduler')
parser.add_argument("--crown_scheduler_opts", type=str, default="start=2,length=20", help='options for crown-ibp scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options')
parser.add_argument("--dir", type=str, default="exp_inv/")
parser.add_argument("--eval_bound_type", type=str, default="same",
                    choices=["IBP", "CROWN-IBP", "CROWN", "same"])

args = parser.parse_args()

# path of tensorboard statistics: exp_inv/cnn_6layer-CROWN/log
writer = SummaryWriter(os.path.join(args.dir, os.path.join(args.model + "-" + args.bound_type + "/log")), flush_secs=10)

# path of checkpoints
ckp_path = os.path.join(args.dir, os.path.join(args.model + "-" + args.bound_type + "/ckp"))

def Train(model, t, loader, eps_scheduler, crown_scheduler, norm, train, opt, bound_type, epoch, method='robust'):
    num_class = 10
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))

        if crown_scheduler is not None:
            crown_scheduler.train()
            crown_scheduler.step_epoch()
            crown_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size)) # the number of batches in an epoch

    else:
        model.eval()
        eps_scheduler.eval()
        
        if crown_scheduler is not None:
            crown_scheduler.eval()

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-12:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_class - 1, num_class))
        # bound input for Linf norm used only
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels, c = data.cuda(), labels.cuda(), c.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        # Specify Lp norm perturbation.
        # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb)

        output = model(x)
        regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
        meter.update('CE', regular_ce.item(), x.size(0))
        meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0), x.size(0))

        if batch_method == "robust":
            if bound_type == "IBP":
                lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
            elif bound_type == "CROWN":
                ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
                lb, ub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
            elif bound_type == "CROWN-IBP":
                # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
                # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
                # factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                if crown_scheduler is not None:
                    crown_scheduler.step_batch()
                    factor = 1 - crown_scheduler.get_eps()
                else:
                    factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
                if factor < 1e-5:
                    lb = ilb
                else:
                    clb, cub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                    lb = clb * factor + ilb * (1 - factor)

            # Pad zero at the beginning for each example, and use fake label "0" for all examples
            lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
        if batch_method == "robust":
            loss = robust_ce
        elif batch_method == "natural":
            loss = regular_ce
        if train:
            loss.backward()
            eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))
        if batch_method != "natural":
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
            # If any margin is < 0 this example is counted as an error
            meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)

        if train:
            writer.add_scalar('loss_train_{}'.format(epoch), meter.avg("CE"), i + 1)
            writer.add_scalar('acc_train_{}'.format(epoch), 1.0 - meter.avg("Err"), i + 1)

            if batch_method != "natural":
                writer.add_scalar('loss_robust_train_{}'.format(epoch), meter.avg("Robust_CE"), i + 1)
                writer.add_scalar('acc_robust_train_{}'.format(epoch), 1.0 - meter.avg("Verified_Err"), i + 1)
        if i % 50 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
    print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))

    if train:
        writer.add_scalar('loss/train', meter.avg("CE"), epoch)
        writer.add_scalar('acc/train', 1.0 - meter.avg("Err"), epoch)

        if batch_method != "natural":
            writer.add_scalar('loss/robust_train', meter.avg("Robust_CE"), epoch)
            writer.add_scalar('acc/robust_train', 1.0 - meter.avg("Verified_Err"), epoch)
    else:
        writer.add_scalar('loss/test', meter.avg("CE"), epoch)
        writer.add_scalar('acc/test', 1.0 - meter.avg("Err"), epoch)

        if batch_method != "natural":
            writer.add_scalar('loss/robust_test', meter.avg("Robust_CE"), epoch)
            writer.add_scalar('acc/robust_test', 1.0 - meter.avg("Verified_Err"), epoch)

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Step 1: Initial original model as usual, see model details in models/example_feedforward.py and models/example_resnet.py
    if args.data == 'MNIST':
        model_ori = models.Models[args.model](in_ch=1, in_dim=28)
    else:
        model_ori = models.Models[args.model](in_ch=3, in_dim=32)
    if args.load:
        state_dict = torch.load(args.load)['state_dict']
        model_ori.load_state_dict(state_dict)
    # model_ori = ensemble_MNIST()
    # model_ori.load("./cnn_6layer", "./cnn_4layer")

    ## Step 2: Prepare dataset as usual
    if args.data == 'MNIST':
        dummy_input = torch.randn(1, 1, 28, 28)
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    elif args.data == 'CIFAR':
        dummy_input = torch.randn(1, 3, 32, 32)
        normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
        train_data = datasets.CIFAR10("./data", train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize])) 
        test_data = datasets.CIFAR10("./data", train=False, download=True, 
                transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    if args.data == 'MNIST':
        train_data.mean = test_data.mean = torch.tensor([0.0])
        train_data.std = test_data.std = torch.tensor([1.0])
    elif args.data == 'CIFAR':
        train_data.mean = test_data.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        train_data.std = test_data.std = torch.tensor([0.2023, 0.1994, 0.2010])

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    model = BoundedModule(model_ori, dummy_input, bound_opts={'relu':args.bound_opts}, device=args.device)

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    opt = optim.Adam(model.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_scheduler = eval(args.eps_scheduler_name)(args.eps, args.eps_scheduler_opts)

    if args.crown_scheduler_name is not None:
        crown_scheduler = eval(args.crown_scheduler_name)(1.0, args.crown_scheduler_opts)
    print("Model structure: \n", str(model_ori))

    ## Step 5: start training
    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            Train(model, 1, test_data, eps_scheduler, None, norm, False, None, args.bound_type, 0)
    else:
        timer = 0.0
        for t in range(1, args.num_epochs+1):
            if eps_scheduler.reached_max_eps():
                # Only decay learning rate after reaching the maximum eps
                lr_scheduler.step()
            print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
            start_time = time.time()
            Train(model, t, train_data, eps_scheduler, crown_scheduler, norm, True, opt, args.bound_type, t)
            epoch_time = time.time() - start_time
            timer += epoch_time
            print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            print("Evaluating...")

            eval_bound_type = args.bound_type if args.eval_bound_type == "same" else args.eval_bound_type
            with torch.no_grad():
                Train(model, t, test_data, eps_scheduler, crown_scheduler, norm, False, None, eval_bound_type, t)
            
            if not os.path.exists(ckp_path):
                os.mkdir(ckp_path)

            torch.save({'state_dict': model.state_dict(), 'epoch': t}, os.path.join(ckp_path, "epoch_{}".format(t)))


if __name__ == "__main__":
    main(args)
