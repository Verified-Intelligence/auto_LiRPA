import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lstm import LSTM
from data_utils import load_data, get_batches
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.utils import MultiAverageMeter, logger, get_spec_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--norm", type=int, default=np.inf)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--num_slices", type=int, default=8)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--input_size", type=int, default=784)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--dir", type=str, default="model", help="directory to load or save the model")
parser.add_argument("--num_epochs_warmup", type=int, default=10, help="number of epochs for the warmup stage when eps is linearly increased from 0 to the full value")
parser.add_argument("--log_interval", type=int, default=10, help="interval of printing the log during training")
args = parser.parse_args()


## Train or test one batch.
def step(model, ptb, batch, eps=args.eps, train=False):
    # We increase the perturbation each batch.
    ptb.set_eps(eps)
    # We create a BoundedTensor object with current batch of data.
    X, y = model.get_input(batch)
    X = BoundedTensor(X, ptb)
    logits = model.core(X)

    # Form the linear speicifications, which are margins of ground truth class and other classes.
    num_class = args.num_classes
    c = get_spec_matrix(X, y, num_class)

    # Compute CROWN-IBP (IBP+backward) bounds for training. We only need the lower bound.
    # Here we can omit the x=(X,) argument because we have just used X for forward propagation.
    lb, ub = model.core.compute_bounds(C=c, method='CROWN-IBP', bound_upper=False)

    # Compute robust cross entropy loss.
    lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
    fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
    loss = nn.CrossEntropyLoss()(-lb_padded, fake_labels)

    # Report accuracy and robust accuracy.
    acc = (torch.argmax(logits, dim=-1) == y).float().mean()
    acc_robust = 1 - torch.mean((lb < 0).any(dim=1).float())

    if train:
        loss.backward()

    return acc.detach(), acc_robust.detach(), loss.detach()


## Train one epoch.
def train(epoch):
    meter = MultiAverageMeter()
    model.train()
    # Load data for a epoch.
    train_batches = get_batches(data_train, args.batch_size)

    eps_inc_per_step = 1.0 / (args.num_epochs_warmup * len(train_batches))

    for i, batch in enumerate(train_batches):
        # We increase eps linearly every batch.
        eps = args.eps * min(eps_inc_per_step * ((epoch - 1) * len(train_batches) + i + 1), 1.0)
        # Call the main training loop.
        acc, acc_robust, loss = step(model, ptb, batch, eps=eps, train=True)
        # Optimize the loss.
        torch.nn.utils.clip_grad_norm_(model.core.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        meter.set_batch_size(len(batch))
        meter.update('acc', acc)
        meter.update('acc_rob', acc_robust)
        meter.update('loss', loss)
        if (i + 1) % args.log_interval == 0:
            logger.info("Epoch {}, training step {}/{}: {}, eps {:.3f}".format(
                epoch, i + 1, len(train_batches), meter, eps))
    model.save(epoch)


## Test accuracy and robust accuracy.
def test(epoch, batches):
    meter = MultiAverageMeter()
    model.eval()
    for batch in batches:
        acc, acc_robust, loss = step(model, ptb, batch)
        meter.set_batch_size(len(batch))
        meter.update('acc', acc)
        meter.update('acc_rob', acc_robust)
        meter.update('loss', loss)
    logger.info("Epoch {} test: {}".format(epoch, meter))

# Load MNIST dataset
logger.info("Loading data...")
data_train, data_test = load_data()
logger.info("Dataset sizes: {}/{}".format(len(data_train), len(data_test)))
test_batches = get_batches(data_test, args.batch_size)

# Set all random seeds.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create a LSTM sequence classifier.
logger.info("Creating LSTM model...")
model = LSTM(args).to(args.device)
X, y = model.get_input(test_batches[0])
# Create the perturbation object once here, and we can reuse it.
ptb = PerturbationLpNorm(norm=args.norm, eps=args.eps)
# Convert the LSTM to BoundedModule
X = BoundedTensor(X, ptb)
model.core = BoundedModule(model.core, (X,), device=args.device)
optimizer = model.build_optimizer()

# Main training loop.
for t in range(model.checkpoint, args.num_epochs):
    train(t + 1)
    test(t + 1, test_batches)

# If the loaded model has already reached the last epoch, test it directly.
if model.checkpoint == args.num_epochs:
    test(args.num_epochs, test_batches)

