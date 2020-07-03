import multiprocessing
import torch
from torch.utils import data
from functools import partial
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# compute image statistics (by Andreas https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/4)
def get_stats(loader):
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0) 
        reshaped_img = images.view(batch_samples, images.size(1), -1)
        mean += reshaped_img.mean(2).sum(0)
    w = images.size(2)
    h = images.size(3)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*w*h))
    return mean, std

# load MNIST of Fashion-MNIST
def mnist_loaders(dataset, batch_size, shuffle_train = True, shuffle_test = False, ratio=None, test_batch_size=None):
    mnist_train = dataset("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = dataset("./data", train=False, download=True, transform=transforms.ToTensor())

    if ratio is not None:
        # only sample in training data
        num_of_each_class_train = int(len(mnist_train) // 10 * ratio)
        # num_of_each_class_test = int(len(mnist_test)//10*ratio)

        class_idx_train = [(mnist_train.targets == _).nonzero().numpy().squeeze() for _ in range(10)]
        # class_idx_test = [(mnist_test.targets==_).nonzero().numpy().squeeze() for _ in range(10)]

        for i in range(len(class_idx_train)):
            class_idx_train[i] = class_idx_train[i][:num_of_each_class_train]
            # class_idx_test[i] = class_idx_test[i][:num_of_each_class_test]

        mnist_train = data.Subset(mnist_train, [y for z in class_idx_train for y in z])

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=shuffle_train, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),2))
    if test_batch_size:
        batch_size = test_batch_size
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),2))
    std = [1.0]
    train_loader.std = std
    test_loader.std = std
    return train_loader, test_loader


def cifar_loaders(batch_size, shuffle_train = True, shuffle_test = False, train_random_transform = False, normalize_input = False, num_examples = None, test_batch_size=None): 
    if normalize_input:
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                          std = std)
    else:
        std = [1.0, 1.0, 1.0]
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=std)
    if train_random_transform:
        if normalize_input:
            train = datasets.CIFAR10('./data', train=True, download=True, 
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train = datasets.CIFAR10('./data', train=True, download=True, 
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                ]))
    else:
        train = datasets.CIFAR10('./data', train=True, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),normalize]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    if num_examples:
        indices = list(range(num_examples))
        train = data.Subset(train, indices)
        test = data.Subset(test, indices)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=shuffle_train, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    if test_batch_size:
        batch_size = test_batch_size
    test_loader = torch.utils.data.DataLoader(test, batch_size=max(batch_size, 1),
        shuffle=shuffle_test, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    train_loader.std = std
    test_loader.std = std
    return train_loader, test_loader

def svhn_loaders(batch_size, shuffle_train = True, shuffle_test = False, train_random_transform = False, normalize_input = False, num_examples = None, test_batch_size=None): 
    if normalize_input:
        mean = [0.43768206, 0.44376972, 0.47280434] 
        std = [0.19803014, 0.20101564, 0.19703615]
        normalize = transforms.Normalize(mean = mean,
                                          std = std)
    else:
        std = [1.0, 1.0, 1.0]
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=std)
    if train_random_transform:
        if normalize_input:
            train = datasets.SVHN('./data', split='train', download=True, 
                transform=transforms.Compose([
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train = datasets.SVHN('./data', split='train', download=True, 
                transform=transforms.Compose([
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                ]))
    else:
        train = datasets.SVHN('./data', split='train', download=True, 
            transform=transforms.Compose([transforms.ToTensor(),normalize]))
    test = datasets.SVHN('./data', split='test', download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    if num_examples:
        indices = list(range(num_examples))
        train = data.Subset(train, indices)
        test = data.Subset(test, indices)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=shuffle_train, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    if test_batch_size:
        batch_size = test_batch_size
    test_loader = torch.utils.data.DataLoader(test, batch_size=max(batch_size, 1),
        shuffle=shuffle_test, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    train_loader.std = std
    test_loader.std = std
    mean, std = get_stats(train_loader)
    print('dataset mean = ', mean.numpy(), 'std = ', std.numpy())
    return train_loader, test_loader

def load_data(data, batch_size):
    if data == 'MNIST':
        dummy_input = torch.randn(1, 1, 28, 28)
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    elif data == 'CIFAR':
        dummy_input = torch.randn(1, 3, 32, 32)
        normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4, padding_mode='edge'),
                    transforms.ToTensor(),
                    normalize]))
        test_data = datasets.CIFAR10('./data', train=False, download=True, 
                transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    test_data = torch.utils.data.DataLoader(test_data, batch_size=batch_size, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
    if data == 'MNIST':
        train_data.mean = test_data.mean = torch.tensor([0.0])
        train_data.std = test_data.std = torch.tensor([1.0])
    elif data == 'CIFAR':
        train_data.mean = test_data.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        train_data.std = test_data.std = torch.tensor([0.2023, 0.1994, 0.2010])

    return dummy_input, train_data, test_data

# when new loaders is added, they must be registered here
loaders = {
        "MNIST": partial(mnist_loaders, datasets.MNIST),
        "FashionMNIST": partial(mnist_loaders, datasets.FashionMNIST),
        "CIFAR": cifar_loaders,
        "svhn": svhn_loaders,
        }

