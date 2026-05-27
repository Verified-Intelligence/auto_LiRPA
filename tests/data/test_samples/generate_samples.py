"""Download datasets and save minimal test samples as .npy files.

Run this script to regenerate the test sample files:
    python tests/data/test_samples/generate_samples.py
"""
import os
import numpy as np
import torchvision

out_dir = os.path.dirname(__file__)

# CIFAR10: 1 test image, shape (1, 32, 32, 3), uint8
cifar = torchvision.datasets.CIFAR10(
    os.path.join(out_dir, '_download_cache'), train=False, download=True)
np.save(os.path.join(out_dir, 'cifar10_test_1.npy'), cifar.data[:1])
print(f'Saved cifar10_test_1.npy: shape={cifar.data[:1].shape}')

# MNIST: 2 test images, shape (2, 28, 28), uint8
mnist = torchvision.datasets.MNIST(
    os.path.join(out_dir, '_download_cache'), train=False, download=True)
np.save(os.path.join(out_dir, 'mnist_test_2.npy'), mnist.data[:2].numpy())
print(f'Saved mnist_test_2.npy: shape={mnist.data[:2].shape}')
