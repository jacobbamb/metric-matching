"""FFHQ dataset and datamodule for denoiser/score model training.

FFHQ (Flickr-Faces-HQ) contains 70,000 high-quality 1024x1024 face images.
Data structure: images1024x1024/{folder}/{image_id}.png
  where folder is zero-padded to 5 digits, grouped in 1000s.

Standard split: 60k train / 10k val (last 10k images).
"""

from pathlib import Path
from typing import Callable, List, Optional

from lightning.pytorch import LightningDataModule
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FFHQDataset(Dataset):
    """FFHQ dataset loading images from the standard directory structure."""

    def __init__(
        self,
        root: str,
        image_paths: List[Path],
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def _build_transform(image_size: int, augment: bool) -> transforms.Compose:
    """Standard diffusion-model preprocessing for FFHQ.

    FFHQ images are already square (1024x1024), so no cropping is needed.
    We resize, optionally apply horizontal flip augmentation, and normalize
    pixel values from [0, 1] to [-1, 1].
    """
    ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS)]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


class FFHQDataModule(LightningDataModule):
    """LightningDataModule for FFHQ 256x256 denoiser training.

    Args:
        data_root: Path to the ffhq-dataset directory containing images1024x1024/.
        batch_size: Batch size for training and validation.
        num_workers: Number of DataLoader worker processes.
        image_size: Target spatial resolution (both H and W).
        val_size: Number of images reserved for validation (taken from the end).
        pin_memory: Pin memory in DataLoader for faster GPU transfers.
        drop_last: Drop the last incomplete batch during training.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 8,
        image_size: int = 256,
        val_size: int = 10_000,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_size = val_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_dataset: Optional[FFHQDataset] = None
        self.val_dataset: Optional[FFHQDataset] = None

    def _collect_image_paths(self) -> List[Path]:
        img_root = self.data_root / "images1024x1024"
        if not img_root.exists():
            raise FileNotFoundError(f"FFHQ image directory not found: {img_root}")
        paths = sorted(img_root.rglob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG images found under {img_root}")
        return paths

    def prepare_data(self):
        # Verify the dataset directory is accessible.
        self._collect_image_paths()

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            all_paths = self._collect_image_paths()
            train_paths = all_paths[: len(all_paths) - self.val_size]
            val_paths = all_paths[len(all_paths) - self.val_size :]

            self.train_dataset = FFHQDataset(
                root=self.data_root,
                image_paths=train_paths,
                transform=_build_transform(self.image_size, augment=True),
            )
            self.val_dataset = FFHQDataset(
                root=self.data_root,
                image_paths=val_paths,
                transform=_build_transform(self.image_size, augment=False),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
