"""LightningDataModule for CIFAR-10."""

from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 4,
        noise_level: float = 0.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_level = noise_level

        noise_transform = transforms.Lambda(
            lambda x: (x + self.noise_level * torch.randn_like(x)).clamp(0, 1)
        )
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                noise_transform,
                normalize,
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                noise_transform,
                normalize,
            ]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        del stage
        self.cifar10_train = CIFAR10(
            self.data_dir, train=True, transform=self.train_transform
        )
        self.cifar10_val = CIFAR10(
            self.data_dir, train=False, transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
