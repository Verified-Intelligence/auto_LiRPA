import random
from torchvision import transforms
from torchvision.datasets.mnist import MNIST as mnist

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_train = mnist("data", train=True, download=True, transform=transform)
    data_test = mnist("data", train=False, download=True, transform=transform)
    data_train = [data_train[i] for i in range(len(data_train))]
    data_test = [data_test[i] for i in range(len(data_test))]
    return data_train, data_test

def get_batches(data, batch_size):
    batches = []
    random.shuffle(data)
    for i in range((len(data) + batch_size - 1) // batch_size):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches