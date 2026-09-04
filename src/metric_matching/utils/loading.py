"""Checkpoint and config discovery/loading.

Training runs are laid out as::

    outputs/{score,metric}/<dataset>/<timestamp>/
        .hydra/config.yaml
        checkpoints/epoch=NNNN.ckpt, last.ckpt

``select_checkpoint`` also resolves legacy run directories
(``.../outputs/checkpoints/epochepoch=N.ckpt``): any filename carrying an
``epoch=<number>`` fragment is understood, and epochs are matched exactly
(epoch 99 never matches ``epoch=999``). When several runs live under the
same root, the most recently modified matching checkpoint wins.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
from omegaconf import OmegaConf
from omegaconf.errors import UnsupportedInterpolationType

_EPOCH_RE = re.compile(r"epoch=0*(\d+)")


def _all_checkpoints(root: Path) -> list:
    """All .ckpt files under root, oldest first by mtime."""
    return sorted(root.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)


def find_latest_ckpt(root: str = "outputs") -> Optional[Path]:
    """Return the most recently modified .ckpt under ``root``, or None."""
    cands = _all_checkpoints(Path(root))
    return cands[-1] if cands else None


def select_checkpoint(
    epoch: Optional[int] = None, preferred_root: Optional[str] = None
) -> Path:
    """Select a checkpoint under ``preferred_root`` (default: ``outputs``).

    epoch=None returns the newest ``last.ckpt`` (falling back to the newest
    checkpoint of any name); an explicit epoch is matched exactly against the
    ``epoch=<N>`` fragment of the filename. Raises FileNotFoundError with
    guidance when nothing matches.
    """
    root = Path(preferred_root) if preferred_root is not None else Path("outputs")
    candidates = _all_checkpoints(root) if root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under '{root}'. Train a model first, or point "
            "at an existing run directory."
        )

    if epoch is None:
        lasts = [p for p in candidates if p.name == "last.ckpt"]
        return (lasts or candidates)[-1]

    matches = []
    available = set()
    for p in candidates:
        m = _EPOCH_RE.search(p.name)
        if m is None:
            continue
        e = int(m.group(1))
        available.add(e)
        if e == int(epoch):
            matches.append(p)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint for epoch {epoch} under '{root}' "
            f"(available epochs: {sorted(available)}, plus any last.ckpt)."
        )
    return matches[-1]


def find_config_near(path: Optional[Path]) -> Optional[Path]:
    """Search upward from ``path`` for a Hydra ``.hydra/config.yaml`` (or a
    plain ``config.yaml``). Returns the config path or None."""
    if path is None:
        return None
    p = Path(path).resolve().parent
    for _ in range(6):
        c = p / ".hydra" / "config.yaml"
        if c.exists():
            return c
        c2 = p / "config.yaml"
        if c2.exists():
            return c2
        p = p.parent
    return None


def load_cfg_container(cfg_path: Optional[Path]) -> Dict[str, Any]:
    """Load a YAML config as a plain dict. Unresolvable interpolations (e.g.
    ``${hydra:runtime.cwd}`` in a saved run config) are left as-is."""
    if cfg_path is None:
        return {}
    cfg = OmegaConf.load(str(cfg_path))
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except UnsupportedInterpolationType:
        return OmegaConf.to_container(cfg, resolve=False)


def load_checkpoint_and_cfg(
    epoch: Optional[int] = None, preferred_root: Optional[str] = None
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    """Select a checkpoint, find its run config, and return
    ``(ckpt_path, cfg_path, cfg_container)``."""
    ckpt = select_checkpoint(epoch=epoch, preferred_root=preferred_root)
    cfg_path = find_config_near(ckpt)
    cfg_container = load_cfg_container(cfg_path)
    return ckpt, cfg_path, cfg_container


def load_system_from_checkpoint(
    system_cls: Type,
    epoch: Optional[int] = None,
    preferred_root: Optional[str] = None,
    map_location: str = "cpu",
):
    """Instantiate ``system_cls`` from the run config next to the selected
    checkpoint and load its weights. Prints the resolved checkpoint path so it
    is always visible which weights a run actually used."""
    ckpt, cfg_path, cfg_container = load_checkpoint_and_cfg(
        epoch=epoch, preferred_root=preferred_root
    )
    print(f"[metric-matching] loading {system_cls.__name__} weights from: {ckpt}")
    model = system_cls(cfg_container)
    state = torch.load(str(ckpt), map_location=map_location, weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    return model


def prepare_mnist_datamodule(
    cfg_container: Dict[str, Any], batch_size: int = 1, num_workers: int = 0
):
    """Instantiate MNISTDataModule from a run config and ensure the dataset is
    available. Returns ``(dm, val_ds)``."""
    from metric_matching.data.datamodule import MNISTDataModule

    data_cfg = cfg_container.get("data", {}) if isinstance(cfg_container, dict) else {}
    data_dir = data_cfg.get("data_dir", "data")

    dm = MNISTDataModule(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )
    try:
        dm.setup()
    except RuntimeError as e:
        if "Dataset not found" in str(e) or "download=True" in str(e):
            dm.prepare_data()
            dm.setup()
        else:
            raise
    return dm, dm.mnist_val
