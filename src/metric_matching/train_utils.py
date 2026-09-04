"""Shared utilities for training scripts.

Provides common setup functions for reproducibility, logging, and callbacks.
"""

import random
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


def setup_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)


def setup_precision() -> None:
    """Enable Tensor Core-friendly matmul precision if available."""
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass


def flatten_config(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict for W&B hyperparameter logging."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key, sep))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            items[new_key] = v
        elif isinstance(v, (list, tuple)):
            try:
                items[new_key] = ",".join(map(str, v))
            except Exception:
                items[new_key] = str(v)
        else:
            items[new_key] = str(v)
    return items


def create_wandb_logger(cfg: DictConfig) -> Optional[WandbLogger]:
    """Create WandB logger from config if configured."""
    log_cfg = cfg.get("logger", None)
    if not log_cfg or not log_cfg.get("project"):
        return None

    try:
        logger = WandbLogger(
            project=log_cfg.get("project"),
            entity=log_cfg.get("entity", None),
            log_model=log_cfg.get("log_model", False),
        )
        # Create the run now so a failing wandb.init is caught here rather
        # than later inside Trainer.fit (after _log_hyperparams swallowed it).
        logger.experiment
        _log_hyperparams(logger, cfg)
        return logger
    except Exception as e:
        warnings.warn(
            f"Could not create the W&B logger ({e!r}); training continues "
            "without logging. Set WANDB_MODE=offline, or drop the logger "
            "block with the `~logger` override, to silence this."
        )
        return None


def _log_hyperparams(logger: WandbLogger, cfg: DictConfig) -> None:
    """Log flattened config as hyperparameters to W&B."""
    try:
        hparams = OmegaConf.to_container(cfg, resolve=True)
        flat = {f"cfg.{k}": v for k, v in flatten_config(hparams).items()}
        logger.log_hyperparams(flat)
    except Exception:
        try:
            logger.log_hyperparams({"cfg": OmegaConf.to_yaml(cfg)})
        except Exception:
            pass


def create_checkpoint_callback(cfg: DictConfig) -> ModelCheckpoint:
    """Create checkpoint callback from config."""
    ckpt_cfg = cfg.get("checkpoint", {}) or {}
    return ModelCheckpoint(
        dirpath=ckpt_cfg.get("dirpath", "outputs/checkpoints"),
        filename=ckpt_cfg.get("filename", "{epoch:04d}"),
        every_n_epochs=ckpt_cfg.get("every_n_epochs", 1),
        save_top_k=ckpt_cfg.get("save_top_k", -1),
        save_last=ckpt_cfg.get("save_last", True),
    )


def create_trainer(
    cfg: DictConfig, logger: Optional[WandbLogger], callbacks: List
) -> Trainer:
    """Create PyTorch Lightning Trainer from config.

    Every key under ``cfg.trainer`` is forwarded to the Trainer verbatim, so
    command-line overrides such as ``+trainer.limit_train_batches=5`` work.
    """
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    return Trainer(
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.optimizer.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=cfg.optimizer.get("gradient_clip_algorithm", "norm"),
        **trainer_kwargs,
    )


def finalize_wandb(trainer: Trainer) -> None:
    """Clean up W&B logging if used."""
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.finish()
        wandb.finish()
