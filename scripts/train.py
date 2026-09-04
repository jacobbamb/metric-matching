"""Train metric matching models.

Usage: python scripts/train.py [hydra overrides]
"""

import os

import hydra
from omegaconf import DictConfig

from metric_matching.systems.mm_system import MMSystem
from metric_matching.callbacks.celeba_viz import CelebAVizCallback
from metric_matching.callbacks.ffhq_viz import FFHQVizCallback
from metric_matching.callbacks.subset_eval import (
    SubsetMarginalCallback,
    SubsetSpectralCallback,
)
from metric_matching.callbacks.sphere_metric_eval import SphereMetricEvalCallback
from metric_matching.callbacks.cifar_viz import CIFARSpectralVizCallback
from metric_matching.callbacks.mnist_viz import MNISTSpectralVizCallback
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
    """Build evaluation callbacks based on data config."""
    data_cfg = cfg.get("data", {}) or {}
    data_name = data_cfg.get("name", "").lower()
    eval_cfg = cfg.get("eval", {}) or {}
    callback_device = "cpu" if cfg.trainer.get("accelerator") == "cpu" else "cuda"

    if data_name == "celeba":
        return [
            CelebAVizCallback(
                eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
                image_indices=eval_cfg.get("image_indices", (0, 1, 2, 3)),
                h_values=eval_cfg.get(
                    "h_values", [float(eval_cfg.get("h_value", 1.0))]
                ),
                eigenvector_indices=eval_cfg.get(
                    "eigenvector_indices", (0, 1, 2, 3)
                ),
                perturb_scale=float(eval_cfg.get("perturb_scale", 12.0)),
            )
        ]

    if data_name == "ffhq":
        return [
            FFHQVizCallback(
                eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
                image_indices=eval_cfg.get("image_indices", (0, 1, 2, 3)),
                h_values=eval_cfg.get("h_values", [1.0, 5.0, 10.0]),
                eigenvector_indices=eval_cfg.get("eigenvector_indices", (0, 1, 2, 3)),
                perturb_scale=float(eval_cfg.get("perturb_scale", 12.0)),
            )
        ]

    bandwidths = [(2 ** (i - 6)) for i in range(11)]
    callbacks = [
        SubsetMarginalCallback(
            bandwidths=bandwidths,
            subset_size=eval_cfg.get("eval_subset_size", 100),
            batch_size=eval_cfg.get("batch_size", 128),
            eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
            device=callback_device,
        ),
        SubsetSpectralCallback(
            subset_size=eval_cfg.get("eval_subset_size", 100),
            batch_size=eval_cfg.get("batch_size", 128),
            eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
            device=callback_device,
        ),
    ]

    if data_name == "sphere":
        callbacks.append(
            SphereMetricEvalCallback(
                bandwidths=bandwidths,
                d=int(data_cfg.get("d", 8)),
                D=int(data_cfg.get("D", 512)),
                batch_size=eval_cfg.get("batch_size", 128),
                eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
            )
        )

    if data_name in ("", "mnist"):
        callbacks.append(
            MNISTSpectralVizCallback(
                eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
            )
        )
    elif data_name == "cifar10" and bool(eval_cfg.get("enable_cifar_viz", False)):
        callbacks.append(
            CIFARSpectralVizCallback(
                eval_every_n_epochs=eval_cfg.get("eval_every_n_epochs", 10),
                image_indices=eval_cfg.get("image_indices", (0, 2, 4, 7, 8)),
                h_value=float(eval_cfg.get("h_value", 1.0)),
                max_eigvals_to_plot=int(eval_cfg.get("max_eigvals_to_plot", 40)),
            )
        )

    return callbacks


@hydra.main(version_base="1.1", config_path="../configs", config_name="mnist")
def main(cfg: DictConfig):
    setup_seed(int(cfg.get("seed", 42)))
    setup_precision()

    logger = create_wandb_logger(cfg)

    data_cfg = cfg.get("data", {}) or {}
    dm = fetch_data(data_cfg.get("name", "").lower(), data_cfg)
    dm.prepare_data()
    dm.setup()

    model = MMSystem(cfg)
    callbacks = [create_checkpoint_callback(cfg)] + build_eval_callbacks(cfg)

    trainer = create_trainer(cfg, logger, callbacks)
    trainer.fit(model, datamodule=dm)
    finalize_wandb(trainer)
    print(f"\nMetric model run directory: {os.getcwd()}")
    print("Checkpoints are under checkpoints/ (last.ckpt = newest).")


if __name__ == "__main__":
    main()
