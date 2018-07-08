"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.

[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import scipy
from helper import LOG



def get_train_valid_loader(data, data_dir, batch_size, logFile, augment, random_seed, target_size,
                           valid_size=0.1, shuffle=True, show_sample=False, num_workers=4, pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    if target_size == (229,229,3):
        print("=====> Train dataset, resize CIFAR image to 229*229*3")
        LOG("=====> Train dataset, resize CIFAR image to 229*229*3", logFile)
        target_resize = (229, 229)
    else:
        LOG("=====> Train dataset, resize CIFAR image to 224*224*3", logFile)
        target_resize = (224, 224)

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    # valid_transform = transforms.Compose([
    #         # transforms.Pad(padding=96, padding_mode='reflect'),
    #         transforms.Resize(target_resize),
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(target_resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(target_resize),
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    if data == "CIFAR10" or data == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=train_transform
        )        

        # valid_dataset = datasets.CIFAR10(
        #     root=data_dir, train=True,
        #     download=True, transform=valid_transform,
        # )
        print("===========================load CIFAR10 dataset===========================")
    elif data == "cifar100" or data == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform
        )

        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=train_transform
        )        

        # valid_dataset = datasets.CIFAR100(
        #     root=data_dir, train=True,
        #     download=True, transform=valid_transform,
        # )
        print("===========================use CIFAR100 dataset===========================")
    else:
        print("ERROR =============================dataset should be CIFAR10 or CIFAR100")
        NotImplementedError


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory
    )    
    # valid_loader = None

    return train_loader, valid_loader, test_loader


def get_test_loader(data,
                    data_dir,
                    batch_size,
                    target_size,
                    logFile,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    if target_size == (229,229,3):
        print("==> Test dataset, resize CIFAR image to 229*229*3")
        LOG("==> Test dataset, resize CIFAR image to 229*229*3", logFile)
        target_resize = (229, 229)
    else:
        print("==> Test dataset, resize CIFAR image to 224*224*3")
        LOG("==> Test dataset, resize CIFAR image to 224*224*3", logFile)
        target_resize = (224, 224)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        # transforms.Pad(padding=96, padding_mode='reflect'),
        transforms.Resize(target_resize),
        transforms.ToTensor(),
        normalize,
    ])

    if data == "CIFAR10" or "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform
        )
    elif data == "CIFAR100" or "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform
        )
    else:
        print("ERROR =============================dataset should be CIFAR10 or CIFAR100")
        LOG("ERROR =============================dataset should be CIFAR10 or CIFAR100", logFile)
        NotImplementedError        

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )        

    return data_loader
