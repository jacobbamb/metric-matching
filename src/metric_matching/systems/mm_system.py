"""MMSystem: predict conditional metric gt and regress with MSE.

Behavior:
- The system builds an encoder (default: simple flatten+linear embedding) that maps inputs x to a feature vector.
- A small metric head maps the feature vector to a flattened predicted gt (D*D entries).
- During training, we use ConditionalMetricMatching (configured via `cfg.loss`) to sample xh and compute the target gh.
- Loss is MSE between predicted flattened gt and the target flattened gh.
"""

from typing import Any, Dict
import logging

import torch
from torch import nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from metric_matching.losses.matcher import (
    ConditionalMetricMatching,
    MeanCenteredMetricMatching,
)
from metric_matching.models.utils import build_encoder
from metric_matching.systems.score_system import load_score_model_from_checkpoint
from metric_matching.utils.loading import load_system_from_checkpoint
from metric_matching.utils.optim import build_lr_scheduler, build_optimizer


def load_metric_model_from_checkpoint(
    epoch=None, preferred_root=None
) -> torch.nn.Module:
    """Load a trained MMSystem from the newest checkpoint under preferred_root
    (or the exact epoch when given)."""
    return load_system_from_checkpoint(
        MMSystem, epoch=epoch, preferred_root=preferred_root
    )


def _quadform_batch(U: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Compute a^T (U U^T) a without building any D×D matrices.
    U: [B, D, r]   (D = out_dim, r = emb_dim)
    a: [B, D]
    returns: [B]
    """
    Z = torch.einsum("bdr,bd->br", U, a)  # Z = U^T a
    return (Z * Z).sum(-1)  # ||U^T a||^2


def _frobenius_norm_sq_gram(U: torch.Tensor) -> torch.Tensor:
    """
    ||U U^T||_F^2 = ||U^T U||_F^2, computed via the small r×r Gram.
    U: [B, D, r]  -> G = [B, r, r]
    returns: [B]
    """
    G = torch.einsum("bdr,bdq->brq", U, U)  # U^T U
    return (G * G).sum(dim=(1, 2))  # ||G||_F^2


class MetricHead(nn.Module):
    """Predict flattened gt matrix (D*D) from embeddings."""

    def __init__(self, rank: int, out_dim: int = None, smart_training: bool = True):
        super().__init__()
        self.rank = rank
        self.out_dim = out_dim
        self.smart_training = smart_training

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        assert (
            len(emb.shape) == 3
            and emb.shape[1] == self.rank
            and emb.shape[2] == self.out_dim
        )
        if not self.smart_training:
            pred_gh = emb.transpose(1, 2) @ emb  # gives symmetric and PSD
            return pred_gh
        else:
            return emb


class MMSystem(LightningModule):
    """LightningModule that regresses conditional metric gt with MSE.

    It uses `ConditionalMetricMatching` to sample targets (gh) from inputs.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        try:
            # save_hyperparameters accepts both dict-like and Namespace in many setups,
            # catch exceptions to remain robust.
            self.save_hyperparameters(cfg)
        except Exception:
            pass

        self.cfg = cfg or {}

        model_cfg: Dict = self.cfg.get("model", {}) or {}
        self.rank = model_cfg.get("params", {}).get("rank", 32)
        self.out_dim = model_cfg.get("params", {}).get("output_dim", 28 * 28)

        loss_cfg = self.cfg.get("loss", {}) or {}
        self.mean_centered = bool(
            loss_cfg.get("mean_centered", loss_cfg.get("mean_centred", False))
        )
        # The matcher (and, for the mean-centred loss, its pretrained score
        # model) is only needed to compute training targets - build it lazily
        # so checkpoints can be loaded for inference even when the score run
        # recorded in their config no longer exists on disk.
        self._matcher = None
        self.smart_training = loss_cfg.get("smart_training", True)
        self.tikhonov_epsilon = loss_cfg.get("tikhonov_epsilon", 0.0)

        self.encoder = build_encoder(model_cfg, default_image_size=(28, 28))
        self.head = MetricHead(
            rank=self.rank,
            out_dim=self.out_dim,
            smart_training=self.smart_training,
        )

    @property
    def matcher(self):
        if self._matcher is None:
            self._matcher = self._build_matcher()
        return self._matcher

    def _build_matcher(self):
        loss_cfg = self.cfg.get("loss", {}) or {}
        common = dict(
            h_max=loss_cfg.get("h_max", 1.0),
            h_min=loss_cfg.get("h_min", 0.001),
            sampling_method=loss_cfg.get("sampling_method", "uniform"),
        )
        if not self.mean_centered:
            return ConditionalMetricMatching(**common)

        score_path = loss_cfg.get("score_path", None)
        if not score_path:
            raise ValueError(
                "loss.mean_centered=True requires loss.score_path to be set."
            )
        try:
            score_encoder = load_score_model_from_checkpoint(
                epoch=loss_cfg.get("score_epoch", None),
                preferred_root=score_path,
            )
        except FileNotFoundError as e:
            dataset = (self.cfg.get("data", {}) or {}).get("name", "<dataset>")
            raise FileNotFoundError(
                f"No score model found under '{score_path}'. Mean-centred "
                "metric matching needs a pretrained score model - train one "
                f"first with:\n    python scripts/train_score.py --config-name {dataset}\n"
                "or set loss.score_path to an existing score run directory "
                "(with loss.score_epoch=<N> to pin an exact epoch)."
            ) from e
        return MeanCenteredMetricMatching(score_model=score_encoder, **common)

    def on_fit_start(self) -> None:
        if self.mean_centered:
            # access score_model through matcher when needed
            self.matcher.score_model.to(
                self.device
            )  # score_model is not a submodule so params won't be update
            self.matcher.score_model.eval()
            self.matcher.score_model.requires_grad_(False)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(h, x)  # shape [bs, rank*out_dim]
        emb = emb.reshape(emb.shape[0], -1, self.out_dim)  # flatten if needed
        return self.head(emb)

    def _step(self, batch: Any) -> torch.Tensor:
        """Shared computation for training and validation steps."""
        x, _ = batch
        if self.smart_training:  # more efficient when the rank is lower than out_dim
            h, xh, diff = self.matcher.sample_location_and_conditional_diff(
                x.reshape(x.shape[0], -1), normalize=True, h=None
            )

            emb = self.forward(h, xh).reshape([x.shape[0], -1, self.out_dim])

            U = emb.transpose(1, 2)  # [bs, out_dim, emb_dim]
            quad = _quadform_batch(U, diff)  # [bs]
            frob = _frobenius_norm_sq_gram(U)  # [bs]
            const = (diff.pow(2).sum(-1)).pow(
                2
            )  # doesn't contribute to gradients but makes the losses match exactly
            D = emb.shape[2]
            tikhonov = 0.0
            if self.tikhonov_epsilon != 0.0:
                tikhonov = 2.0 * self.tikhonov_epsilon * (emb.pow(2).sum(dim=(1, 2)))
            loss = (
                (frob - 2.0 * quad + const + tikhonov) / (D * D)
            ).mean()  # divide by DxD to match to scale of non smart MSE, as the frob is summed over D×D entries
        else:
            logging.getLogger(__name__).debug("Running non-smart training path")
            h, xh, gh = self.matcher.sample_location_and_conditional_metric(
                x.reshape(x.shape[0], -1), normalize=True, h=None  # flatten for matcher
            )

            bs, D, _ = gh.shape
            gh_flat = gh.reshape([bs, D * D])

            pred_flat = self.forward(h, xh).reshape([bs, D * D])
            loss = F.mse_loss(pred_flat, gh_flat)

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
        opt_cfg = self.cfg.get("optimizer", {}) or {}
        optimizer = build_optimizer(self.parameters(), opt_cfg)
        if opt_cfg.get("lr_scheduler", None) is None:
            return optimizer
        # Per-step schedule over the whole fit (needs the attached trainer).
        scheduler = build_lr_scheduler(
            optimizer, opt_cfg, total_steps=self.trainer.estimated_stepping_batches
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
