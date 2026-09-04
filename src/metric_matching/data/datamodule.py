"""LightningDataModule for MNIST"""

from typing import Optional
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
import torch


class MNISTDataModule(LightningDataModule):
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

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: (x + self.noise_level * torch.randn_like(x)).clamp(0, 1)
                ),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
