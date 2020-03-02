import random, sys, time
import argparse
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundGeneral
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from models.sample_models import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument("--train", action="store_true", help='training/verification mode')
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help='use cpu or gpu')
parser.add_argument("--data", type=str, default="MNIST", choices=["MNIST", "CIFAR"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--eps", type=float, default=0.3, help='epsilon for concretize')
parser.add_argument("--norm", type=float, default='inf', help='lp norm to concretize')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="cnn_2layer_MNIST",
                    choices=["cnn_2layer_MNIST", "mlp_3layer_MNIST"], help='see in sample_models.py')
parser.add_argument("--num_epochs", type=int, default=100, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--schedule_length", type=int, default=61, help='number of epochs for robust training')
parser.add_argument("--schedule_start", type=int, default=1, help='number of epochs for pre natural training')

args = parser.parse_args()


def Train(model, t, loader, start_eps, end_eps, max_eps, norm, train, opt, bound_type, method='robust'):
    num_class = 10
    meter = MultiAverageMeter()
    if train:
        model.train()
    else:
        model.eval()
    # Pre-generate the array for specifications, will be used latter for scatter
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

    # Increase epsilon batch by batch
    batch_eps = np.linspace(start_eps, end_eps, (total // batch_size) + 1)
    # For small eps just use natural training, no need to compute LiRPA bounds
    if end_eps < 1e-6: method = "natural"

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps = batch_eps[i]
        if train:
            opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_class - 1, num_class))
        # scatter matrix to avoid compute margin to self
        sa_labels = sa[labels]
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(data.size(0), num_class)
        # bound input for Linf norm used only
        if norm == np.inf:
            data_ub = (data + eps).clamp(max=1.0)
            data_lb = (data - eps).clamp(min=0.0)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels, sa_labels, c, lb_s = data.cuda(), labels.cuda(), sa_labels.cuda(), c.cuda(), lb_s.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        output = model(data)
        regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
        meter.update('CE', regular_ce.cpu().detach().numpy(), data.size(0))
        meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / data.size(0), data.size(0))

        # Specify Lp norm perturbation.
        # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        if method == "robust":
            if bound_type == "IBP":
                lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method=None)
            elif bound_type == "CROWN":
                lb, ub = model.compute_bounds(ptb=ptb, IBP=False, x=data, C=c, method="backward")
            elif bound_type == "CROWN-IBP":
                # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
                # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
                factor = (max_eps - eps) / max_eps
                ilb, iub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method=None)
                if factor < 1e-5:
                    lb = ilb
                else:
                    clb, cub = model.compute_bounds(ptb=ptb, IBP=False, x=data, C=c, method="backward")
                    lb = clb * factor + ilb * (1 - factor)

            # Filling a missing 0 in lb. The margin from class j to itself is always 0 and not computed.
            lb = lb_s.scatter(1, sa_labels, lb)
            # Use the robust cross entropy loss objective (Wong & Kolter, 2018)
            robust_ce = CrossEntropyLoss()(-lb, labels)
        if method == "robust":
            loss = robust_ce
        elif method == "natural":
            loss = regular_ce
        if train:
            loss.backward()
            opt.step()
        meter.update('Loss', loss.cpu().detach().numpy(), data.size(0))
        if method != "natural":
            meter.update('Robust_CE', robust_ce.cpu().detach().numpy(), data.size(0))
            # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
            # If any margin is < 0 this example is counted as an error
            meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)
        if i % 50 == 0 and train:
            print('[{:2d}:{:4d}]: eps={:4f} {}'.format(t, i, eps, meter))

    print('[FINAL RESULT] epoch={:2d} eps={:.4f} {}'.format(t, eps, meter))

def main(args):
    ## Step 1: Initial original model as usual, see model details in models/sample_models.py
    model_ori = Models[args.model]()

    ## Step 2: Prepare dataset as usual
    if args.data == 'MNIST':
        dummy_input = torch.randn(1, 1, 28, 28)
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    elif args.data == 'CIFAR':
        dummy_input = torch.randn(1, 3, 32, 32)
        train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())

    train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    model = BoundGeneral(model_ori, dummy_input)

    ## Step 4: preprocessing as usual
    if args.device == "gpu": model.cuda()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    ## Step 4 prepare epsilon scheduler and learning rate scheduler
    norm = float(args.norm)
    # increasing epsilon gradually leads better convergence
    eps_schedule = [0] * args.schedule_start + list(np.linspace(0., args.eps, args.schedule_length))
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    print("Model structure: \n", str(model))

    ## Step 5: start training
    timer = 0.0
    for t in range(args.num_epochs):
        lr_scheduler.step(epoch=max(t - len(eps_schedule), 0))
        if t >= len(eps_schedule):
            epoch_start_eps = epoch_end_eps = args.eps
        else:
            epoch_start_eps = eps_schedule[t]
            if t + 1 >= len(eps_schedule):
                epoch_end_eps = epoch_start_eps
            else:
                epoch_end_eps = eps_schedule[t + 1]

        print("Epoch {}, learning rate {}, epsilon {:.6f} - {:.6f}".format(t, lr_scheduler.get_lr(), epoch_start_eps,
                                                                           epoch_end_eps))
        start_time = time.time()
        Train(model, t, train_data, epoch_start_eps, epoch_end_eps, args.eps, norm, True, opt, args.bound_type)
        epoch_time = time.time() - start_time
        timer += epoch_time
        print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
        print("Evaluating...")
        with torch.no_grad():
            Train(model, t, test_data, epoch_end_eps, epoch_end_eps, args.eps, norm, False, None, args.bound_type)
        torch.save({'state_dict': model.state_dict(), 'epoch': t}, args.model)


if __name__ == "__main__":
    main(args)
