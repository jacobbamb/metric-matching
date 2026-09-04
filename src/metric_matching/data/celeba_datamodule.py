"""CelebA dataset/datamodule (64x64 aligned faces).

CelebA is not auto-downloaded. ``data_root`` must contain
``list_eval_partition.csv`` and the ``img_align_celeba`` image folder,
e.g. obtained via kagglehub::

    pip install kagglehub
    python -c "
    import kagglehub, shutil
    path = kagglehub.dataset_download('jessicali9530/celeba-dataset')
    shutil.copytree(path, 'data/celeba', dirs_exist_ok=True)"
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 64,
        return_attributes: bool = False,
    ):
        self.root = Path(root)
        self.return_attributes = return_attributes

        split_map = {"train": 0, "val": 1, "test": 2}
        if split not in split_map:
            raise ValueError(f"Unsupported CelebA split: {split}")

        partition_path = self.root / "list_eval_partition.csv"
        if not partition_path.exists():
            raise FileNotFoundError(f"Missing CelebA partition file: {partition_path}")

        partition_df = pd.read_csv(partition_path)
        split_id = split_map[split]
        self.filenames = partition_df.loc[
            partition_df["partition"] == split_id, "image_id"
        ].tolist()
        if not self.filenames:
            raise ValueError(f"No CelebA filenames found for split '{split}'")

        self.img_dir = self._resolve_image_dir()

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.attr_df = None
        if return_attributes:
            attr_path = self.root / "list_attr_celeba.csv"
            if not attr_path.exists():
                raise FileNotFoundError(f"Missing CelebA attribute file: {attr_path}")
            self.attr_df = pd.read_csv(attr_path).set_index("image_id")

    def _resolve_image_dir(self) -> Path:
        candidates = [
            self.root / "img_align_celeba",
            self.root / "img_align_celeba" / "img_align_celeba",
            self.root,
        ]
        sample_filename = self.filenames[0]
        for candidate in candidates:
            if candidate.exists() and (candidate / sample_filename).exists():
                return candidate

        candidate_text = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not locate CelebA images. Checked: "
            f"{candidate_text}. Expected to find {sample_filename}."
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_path = self.img_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Missing CelebA image file: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if not self.return_attributes:
            return image, 0

        attrs = self.attr_df.loc[filename].to_numpy()
        attrs = torch.tensor((attrs > 0).astype("float32"))
        return image, attrs


class CelebADataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 64,
        pin_memory: bool = True,
        drop_last: bool = True,
        return_attributes: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.return_attributes = return_attributes

        self.train_dataset: Optional[CelebADataset] = None
        self.val_dataset: Optional[CelebADataset] = None

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = CelebADataset(
                root=self.data_root,
                split="train",
                image_size=self.image_size,
                return_attributes=self.return_attributes,
            )
            self.val_dataset = CelebADataset(
                root=self.data_root,
                split="val",
                image_size=self.image_size,
                return_attributes=self.return_attributes,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
