import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_train_dataloader(train_dir, batch_size, num_workers):
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=data_transform,
                                      target_transform=None)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)

    return train_dataloader


def get_test_dataloader(test_dir, batch_size, num_workers):
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)

    return test_dataloader


def create_data_loaders(train_dir, test_dir, batch_size):
    train_loader = get_train_dataloader(train_dir, batch_size, num_workers=1)
    test_loader = get_test_dataloader(test_dir, batch_size, num_workers=1)

    return train_loader, test_loader
