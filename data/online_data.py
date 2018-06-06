"""
author: Caner Mercan
"""


import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_path = os.path.join(os.path.expanduser('~'), 
                             'learning',
                             'datasets')

def __load_data__(data_dir, batch_size):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # no transformation needed for capsnet contrary to convnets.
    tforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)) ])
    phases = ['train', 'test']
    data, loader = {}, {}
    for p in phases:
        train_flag = True if p == 'train' else False
        data[p] = datasets.MNIST(
                        root = data_dir, 
                        train = train_flag,
                        transform = tforms, 
                        download = True )
        loader[p] = DataLoader(
                        dataset = data[p],
                        batch_size = batch_size,
                        shuffle = True )    
    return loader

def load_MNIST(batch_size):
    mnist_dir = os.path.join(data_path, 'mnist')
    return __load_data__(mnist_dir, batch_size)

def load_CIFAR10(batch_size):
    cifar10_dir = os.path.join(data_path, 'cifar10')    
    return __load_data__(cifar10_dir, batch_size)

        