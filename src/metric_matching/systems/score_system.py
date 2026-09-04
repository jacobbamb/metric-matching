"""Minimal denoiser/score training LightningModule."""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from metric_matching.losses.matcher import ConditionalScoreMatching

from metric_matching.models.utils import build_encoder
from metric_matching.utils.loading import load_system_from_checkpoint
from metric_matching.utils.optim import build_optimizer


def load_score_model_from_checkpoint(
    epoch=None, preferred_root=None
) -> torch.nn.Module:
    """Load a trained ScoreSystem from the newest checkpoint under
    preferred_root (or the exact epoch when given)."""
    return load_system_from_checkpoint(
        ScoreSystem, epoch=epoch, preferred_root=preferred_root
    )


class ScoreSystem(LightningModule):
    """Simple denoiser model trained with Gaussian noise prediction."""

    def __init__(self, cfg: Any):
        super().__init__()
        try:
            self.save_hyperparameters(cfg)
        except Exception:
            pass

        self.cfg = cfg or {}
        model_cfg: Dict = self.cfg.get("model", {}) or {}
        self.out_dim = model_cfg.get("params", {}).get("output_dim", 28 * 28)

        loss_cfg = self.cfg.get("loss", {}) or {}
        self.matcher = ConditionalScoreMatching(
            h_max=loss_cfg.get("h_max", 1.0),
            h_min=loss_cfg.get("h_min", 0.001),
            sampling_method=loss_cfg.get("sampling_method", "uniform"),
        )

        params = model_cfg.get("params", {}) or {}
        rank = params.get("rank")
        if rank != 1:
            raise ValueError(
                "ScoreSystem requires model.params.rank to be set to 1 so it can "
                "reuse the same encoder builder as metric matching."
            )
        self.encoder = build_encoder(model_cfg, default_image_size=(28, 28))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(h, x)  # shape [bs, out_dim]
        out = out.reshape(out.shape[0], -1, self.out_dim)  # [bs, 1, out_dim]
        return out

    def _step(self, batch: Any) -> torch.Tensor:
        "Shared step for training and validation."
        x, _ = batch
        h, xh, _ = self.matcher.sample_location_and_conditional_diff(
            x.reshape(x.shape[0], -1), normalize=True, h=None
        )
        pred_clean = self.forward(h, xh).reshape(
            x.shape[0], self.out_dim
        )  # shape [bs, 1, out_dim] -> [bs, out_dim]
        loss = F.mse_loss(pred_clean, x.reshape(x.shape[0], -1))
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self._step(batch)
        self.log("train/cond_mse", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self._step(batch)
        self.log("val/cond_mse_epoch", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), self.cfg.get("optimizer", {}) or {})
