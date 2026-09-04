from __future__ import annotations

import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from metric_matching.data.sphere.performance import (
    mean_frobenius_basis_distance,
    tangent_space_basis,
)


class SphereMetricEvalCallback(Callback):
    """Evaluate sphere tangent-space reconstruction accuracy during validation."""

    def __init__(
        self,
        *,
        bandwidths: list[float],
        d: int,
        D: int,
        batch_size: int = 256,
        eval_every_n_epochs: int = 1,
        log_prefix: str = "sphere/tangent_frob_h",
    ) -> None:
        super().__init__()
        self.bandwidths = bandwidths
        self.d = d
        self.D = D
        self.batch_size = batch_size
        self.eval_every_n_epochs = eval_every_n_epochs
        self.log_prefix = log_prefix

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "val"):
            return

        dataloader = DataLoader(
            datamodule.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(datamodule, "num_workers", 0),
        )

        device = pl_module.device
        totals = torch.zeros(len(self.bandwidths), device=device)
        count = 0

        for x, _ in dataloader:
            x = x.to(device)
            U_gt = tangent_space_basis(x, d=self.d, D=self.D)

            for idx, h_val in enumerate(self.bandwidths):
                h = torch.full((x.shape[0],), h_val, device=device, dtype=x.dtype)
                out = pl_module.forward(h, x)
                U_pred = self._predict_tangent_basis(pl_module, out)
                totals[idx] += mean_frobenius_basis_distance(U_gt, U_pred) * x.shape[0]

            count += x.shape[0]

        if count == 0:
            return

        for h_val, total in zip(self.bandwidths, totals):
            pl_module.log(f"{self.log_prefix}_{h_val:g}", total / count, on_epoch=True)

    def _predict_tangent_basis(self, pl_module, out: torch.Tensor) -> torch.Tensor:
        d = self.d
        if pl_module.smart_training:
            U = out.transpose(1, 2)
            C = U.transpose(-1, -2) @ U
            evals, V = torch.linalg.eigh(C)
            U_pred = (U @ V[..., -d:]) / evals[..., -d:].clamp_min(
                1e-12
            ).sqrt().unsqueeze(-2)
        else:
            P_pred = 0.5 * (out + out.transpose(-1, -2))
            _, eivec = torch.linalg.eigh(P_pred)
            U_pred = eivec[..., -d:]
        return U_pred
