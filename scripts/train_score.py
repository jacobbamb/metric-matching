"""Train score/denoiser models.

Usage: python scripts/train_score.py [hydra overrides]
"""

import os

import hydra
from omegaconf import DictConfig

from metric_matching.callbacks.celeba_score_viz import CelebAScoreVizCallback
from metric_matching.callbacks.ffhq_score_viz import FFHQScoreVizCallback
from metric_matching.systems.score_system import ScoreSystem
from metric_matching.data.utils import fetch_data
from metric_matching.train_utils import (
    setup_seed,
    setup_precision,
    create_wandb_logger,
    create_checkpoint_callback,
    create_trainer,
    finalize_wandb,
)


def build_eval_callbacks(cfg: DictConfig) -> list:
    data_cfg = cfg.get("data", {}) or {}
    data_name = data_cfg.get("name", "").lower()
    eval_cfg = cfg.get("eval", {}) or {}

    if data_name == "celeba":
        h_values = eval_cfg.get("h_values")
        if h_values is None:
            h_values = [float(eval_cfg.get("h_value", 0.35))]
        return [
            CelebAScoreVizCallback(
                eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
                image_indices=eval_cfg.get("image_indices", (0, 1, 2, 3)),
                h_values=h_values,
                noise_seed=int(eval_cfg.get("noise_seed", 0)),
            )
        ]

    if data_name == "ffhq":
        h_values = eval_cfg.get("h_values")
        if h_values is None:
            h_values = [float(eval_cfg.get("h_value", 1.0))]
        return [
            FFHQScoreVizCallback(
                eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
                image_indices=eval_cfg.get("image_indices", (0, 1, 2, 3)),
                h_values=h_values,
                noise_seed=int(eval_cfg.get("noise_seed", 0)),
            )
        ]

    return []


@hydra.main(version_base="1.1", config_path="../configs/score", config_name="mnist")
def main(cfg: DictConfig):
    setup_seed(int(cfg.get("seed", 42)))
    setup_precision()

    logger = create_wandb_logger(cfg)

    data_cfg = cfg.get("data", {}) or {}
    dm = fetch_data(data_cfg.get("name", "").lower(), data_cfg)
    dm.prepare_data()
    dm.setup()

    model = ScoreSystem(cfg)
    callbacks = [create_checkpoint_callback(cfg)] + build_eval_callbacks(cfg)

    trainer = create_trainer(cfg, logger, callbacks)
    trainer.fit(model, datamodule=dm)
    finalize_wandb(trainer)

    dataset = (cfg.get("data", {}) or {}).get("name", "mnist")
    print(f"\nScore model run directory: {os.getcwd()}")
    print(
        "Metric-matching configs pick up the newest score run here "
        "automatically - train the metric model next with:\n"
        f"    python scripts/train.py --config-name {dataset}"
    )


if __name__ == "__main__":
    main()
