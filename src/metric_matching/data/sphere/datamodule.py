from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from metric_matching.data.synth.generator import generate_sphere_uniform


class SphereDataset(Dataset):
    """Dataset of uniformly sampled sphere points."""

    def __init__(
        self,
        num_points: int = 100,
        radius: float = 1.0,
        d: int = 2,
        D: int = 3,
        seed: Optional[int] = None,
        noise_level: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.num_points = num_points
        self.radius = radius
        self.d = d
        self.D = D
        self.seed = seed
        self.noise_level = noise_level
        self.device = device or torch.device("cpu")

        if seed is not None:
            torch.manual_seed(seed)

        self.points = generate_sphere_uniform(
            num_points=self.num_points,
            radius=self.radius,
            d=self.d,
            D=self.D,
            noise_level=self.noise_level,
            device=self.device,
        )

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        sample = self.points[idx]
        return sample, 0


class SphereDataModule(LightningDataModule):
    def __init__(
        self,
        num_points: int = 100,
        batch_size: int = 64,
        num_workers: int = 0,
        radius: float = 1.0,
        d: int = 2,
        D: int = 3,
        noise_level: float = 0.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.radius = radius
        self.d = d
        self.D = D
        self.noise_level = noise_level
        self.device = torch.device("cpu")

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        self.train = SphereDataset(
            num_points=self.num_points,
            radius=self.radius,
            d=self.d,
            D=self.D,
            seed=0,
            noise_level=self.noise_level,
            device=self.device,
        )
        self.val = SphereDataset(
            num_points=self.num_points,
            radius=self.radius,
            d=self.d,
            D=self.D,
            seed=1234,
            noise_level=self.noise_level,
            device=self.device,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
