"""Tests for the optional ``optimizer.lr_scheduler`` option.

Run with: pytest tests/test_lr_scheduler.py -v
"""

import pytest
import torch

from metric_matching.systems.mm_system import MMSystem
from metric_matching.utils.optim import build_lr_scheduler, build_optimizer


def _params():
    return [torch.nn.Parameter(torch.zeros(3))]


class TestBuildLrScheduler:
    @pytest.mark.parametrize("cfg", [{}, {"lr_scheduler": None}, {"lr_scheduler": "constant"}])
    def test_no_schedule_by_default(self, cfg):
        opt = build_optimizer(_params(), {"lr": 1e-3, **cfg})
        assert build_lr_scheduler(opt, {"lr": 1e-3, **cfg}, total_steps=10) is None

    def test_cosine_decays_to_lr_min_over_total_steps(self):
        cfg = {"lr": 1e-3, "lr_scheduler": "cosine"}
        opt = build_optimizer(_params(), cfg)
        sched = build_lr_scheduler(opt, cfg, total_steps=100)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
        for _ in range(100):
            opt.step()
            sched.step()
        # default lr_min is lr / 100
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-5, rel=1e-3)

    def test_explicit_lr_min(self):
        cfg = {"lr": 1e-3, "lr_scheduler": "cosine", "lr_min": 0.0}
        opt = build_optimizer(_params(), cfg)
        sched = build_lr_scheduler(opt, cfg, total_steps=4)
        for _ in range(4):
            opt.step()
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)

    def test_unknown_schedule_raises(self):
        opt = build_optimizer(_params(), {})
        with pytest.raises(ValueError):
            build_lr_scheduler(opt, {"lr_scheduler": "step"}, total_steps=10)


class TestMMSystemConfigureOptimizers:
    CFG = {
        "model": {"name": "mlp", "params": {"input_dim": 2, "hidden_dim": 8, "num_layers": 1,
                                            "rank": 2, "output_dim": 2, "time_embedding": True}},
        "loss": {"h_min": 0.01, "h_max": 1.0, "sampling_method": "lognormal"},
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
    }

    def test_plain_optimizer_without_trainer(self):
        # No schedule requested: must not need an attached Trainer.
        out = MMSystem(self.CFG).configure_optimizers()
        assert isinstance(out, torch.optim.Optimizer)

    def test_cosine_returns_lightning_dict(self):
        from lightning.pytorch import Trainer
        from torch.utils.data import DataLoader, TensorDataset

        cfg = {**self.CFG, "optimizer": {**self.CFG["optimizer"], "lr_scheduler": "cosine"}}
        model = MMSystem(cfg)
        x = torch.randn(16, 2)
        loader = DataLoader(TensorDataset(x, torch.zeros(16)), batch_size=8)
        trainer = Trainer(max_epochs=2, accelerator="cpu", devices=1, logger=False,
                          enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False)
        trainer.fit(model, loader)  # 4 optimizer steps in total
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        assert lr == pytest.approx(1e-5, rel=1e-3)
