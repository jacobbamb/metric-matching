from metric_matching.data.celeba_datamodule import CelebADataModule
from metric_matching.data.cifar10_datamodule import CIFAR10DataModule
from metric_matching.data.datamodule import MNISTDataModule
from metric_matching.data.ffhq_datamodule import FFHQDataModule
from metric_matching.data.sphere.datamodule import SphereDataModule


def fetch_data(data_name: str, data_cfg: dict):
    """Build the datamodule selected by ``cfg.data.name``.

    Supported names: ``sphere``, ``mnist``, ``cifar10``, ``celeba``, ``ffhq``.
    """
    if data_name == "sphere":
        dm = SphereDataModule(
            num_points=int(data_cfg.get("num_points", 100)),
            batch_size=int(data_cfg.get("batch_size", 64)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            radius=float(data_cfg.get("radius", 1.0)),
            d=int(data_cfg.get("d", 8)),
            D=int(data_cfg.get("D", 512)),
            noise_level=float(data_cfg.get("noise_level", 0.0)),
        )
    elif data_name in ("", "mnist"):
        dm = MNISTDataModule(
            data_dir=data_cfg.get("data_dir"),
            batch_size=int(data_cfg.get("batch_size", 64)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            noise_level=float(data_cfg.get("noise_level", 0.0)),
        )
    elif data_name == "cifar10":
        dm = CIFAR10DataModule(
            data_dir=data_cfg.get("data_dir"),
            batch_size=int(data_cfg.get("batch_size", 64)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            noise_level=float(data_cfg.get("noise_level", 0.0)),
        )
    elif data_name == "celeba":
        dm = CelebADataModule(
            data_root=data_cfg.get("data_root"),
            batch_size=int(data_cfg.get("batch_size", 32)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            image_size=int(data_cfg.get("image_size", 64)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=bool(data_cfg.get("drop_last", True)),
            return_attributes=bool(data_cfg.get("return_attributes", False)),
        )
    elif data_name == "ffhq":
        dm = FFHQDataModule(
            data_root=data_cfg.get("data_root"),
            batch_size=int(data_cfg.get("batch_size", 32)),
            num_workers=int(data_cfg.get("num_workers", 8)),
            image_size=int(data_cfg.get("image_size", 256)),
            val_size=int(data_cfg.get("val_size", 10_000)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=bool(data_cfg.get("drop_last", True)),
        )
    else:
        raise ValueError(
            f"Unsupported data.name: {data_name!r} "
            "(expected one of: sphere, mnist, cifar10, celeba, ffhq)"
        )

    return dm
