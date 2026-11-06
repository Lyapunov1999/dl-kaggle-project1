"""CIFAR-10 transforms and reproducible train/validation split."""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
TRAIN_SIZE = 45_000
VAL_SIZE = 5_000


def make_loaders(data_dir, batch_size, workers, seed, device):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_full = datasets.CIFAR10(data_dir, train=True, download=False, transform=train_transform)
    eval_full = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_transform)
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(len(train_full)), [TRAIN_SIZE, VAL_SIZE], generator=generator)
    train_data = Subset(train_full, train_idx.indices)
    val_data = Subset(eval_full, val_idx.indices)
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers,
                         pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
