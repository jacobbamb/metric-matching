"""Metric matching visualization callback for FFHQ.

Subclasses CelebAVizCallback with FFHQ-specific log keys and titles.
All visualization logic (eigenvector perturbations, grayscale maps) is inherited.
"""

import matplotlib.pyplot as plt
import torch
import wandb

from metric_matching.callbacks.celeba_viz import CelebAVizCallback
from metric_matching.callbacks.wandb_utils import wandb_experiment


class FFHQVizCallback(CelebAVizCallback):
    """Log FFHQ eigendirections and image perturbations to W&B.

    Identical to CelebAVizCallback but logs under ffhq_viz/ keys.
    """

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        exp = wandb_experiment(trainer)
        if exp is None or self._val_images is None:
            return

        device = pl_module.device
        images = self._val_images.to(device)
        x_flat = images.reshape(images.shape[0], -1)

        log_dict = {"epoch": trainer.current_epoch}
        for h_value in self.h_values:
            h = torch.full(
                (images.shape[0], 1), h_value, device=device, dtype=images.dtype
            )
            out_h = pl_module(h, x_flat)
            evals, evecs = self._compute_eigendecomposition(out_h, pl_module)
            evals = evals.detach().cpu()
            evecs = evecs.detach().cpu()

            for plot_idx, sample_idx in enumerate(self.image_indices):
                figure = self._plot_sample(
                    sample_idx=sample_idx,
                    image=self._val_images[plot_idx],
                    evals=evals[plot_idx],
                    evecs=evecs[plot_idx],
                    h_value=h_value,
                )
                log_key = f"ffhq_viz/h_{self._format_h_value(h_value)}/image_{sample_idx}"
                log_dict[log_key] = wandb.Image(figure)
                plt.close(figure)

        exp.log(log_dict, commit=True)

    def _plot_sample(self, sample_idx, image, evals, evecs, h_value):
        fig = super()._plot_sample(
            sample_idx=sample_idx,
            image=image,
            evals=evals,
            evecs=evecs,
            h_value=h_value,
        )
        fig.suptitle(f"FFHQ eigendirections for validation image {sample_idx}")
        return fig
