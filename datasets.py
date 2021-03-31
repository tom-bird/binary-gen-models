import os

import numpy as np
from observations import maybe_download_and_extract
from torchvision import datasets, transforms
import torch

CUDA_KWARGS = {'num_workers': 0, 'pin_memory': False}  # get some memory issues with True
ROOT = os.environ.get("DATASETS_PATH", "../data")


def _image_dataset(dataset, batch_size, test_batch_size, use_cuda, tensor_transforms=None):
    kwargs = CUDA_KWARGS if use_cuda else {}
    transform = [transforms.ToTensor()]
    if tensor_transforms:
        transform += tensor_transforms
    transform = transforms.Compose(transform)
    train_loader = torch.utils.data.DataLoader(
        dataset(ROOT, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset(ROOT, train=False, transform=transform),
        batch_size=test_batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader, test_loader


def binarised_mnist(batch_size, test_batch_size, use_cuda):
    return _image_dataset(datasets.MNIST, batch_size, test_batch_size, use_cuda, [lambda x: (x > 0.5).float()])


def mnist(batch_size, test_batch_size, use_cuda):
    return _image_dataset(datasets.MNIST, batch_size, test_batch_size, use_cuda, [lambda x: 255. * x])


def cifar(batch_size, test_batch_size, use_cuda):
    return _image_dataset(datasets.CIFAR10, batch_size, test_batch_size, use_cuda, [lambda x: 255. * x])


def imagenet32(batch_size, test_batch_size, use_cuda):
    def load(path):
        arr = np.load(path)
        arr = np.swapaxes(arr, 1, 3)
        return torch.from_numpy(arr).float()
    train = load(os.path.join(ROOT, 'imagenet32/train/train.npy'))
    test = load(os.path.join(ROOT, 'imagenet32/test/test.npy'))
    train_dataset = torch.utils.data.TensorDataset(train)
    test_dataset = torch.utils.data.TensorDataset(test)

    kwargs = CUDA_KWARGS if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader, test_loader


dataset_registry = {
    'binary_mnist': binarised_mnist,
    'mnist': mnist,
    'cifar': cifar,
    'imagenet32': imagenet32,
}
