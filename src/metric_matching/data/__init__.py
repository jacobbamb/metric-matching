"""data package"""

from .celeba_datamodule import CelebADataModule
from .cifar10_datamodule import CIFAR10DataModule
from .datamodule import MNISTDataModule
from .ffhq_datamodule import FFHQDataModule
from .sphere.datamodule import SphereDataModule

__all__ = [
    "CelebADataModule",
    "CIFAR10DataModule",
    "MNISTDataModule",
    "FFHQDataModule",
    "SphereDataModule",
]
