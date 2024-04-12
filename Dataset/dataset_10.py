import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# Train Phase transformations
def get_train_transforms():
    train_transforms = A.Compose([
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465)),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), always_apply=True),
        ToTensorV2(),
    ])
    return train_transforms

def get_test_transforms():
    test_transforms = A.Compose([
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ])
    return test_transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        
        self.data = datasets.CIFAR10(root=self.root, train=train, download=download)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, target

def train():
    train_transforms = get_train_transforms()
    train = CIFAR10Dataset(root='./data', train=True, download=True, transform= train_transforms)
    return train

def test():
    test_transforms = get_test_transforms()
    test = CIFAR10Dataset(root='./data', train=False, download=True, transform= test_transforms)
    return test

def get_train_loader(batch_size, shuffle, num_workers, pin_memory, train):
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    return train_loader

def get_test_loader(batch_size, shuffle, num_workers, pin_memory, test):
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return test_loader

