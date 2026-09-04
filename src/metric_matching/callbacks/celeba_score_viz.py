import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from metric_matching.callbacks.wandb_utils import wandb_experiment


class CelebAScoreVizCallback(Callback):
    """Log fixed CelebA denoising examples across multiple bandwidths.

    Subclasses can retarget the W&B log keys / figure titles by overriding
    ``log_prefix`` and ``dataset_label`` (see FFHQScoreVizCallback).
    """

    log_prefix = "celeba_score_viz"
    dataset_label = "CelebA"

    def __init__(
        self,
        eval_every_n_epochs: int = 10,
        image_indices=(0, 1, 2, 3),
        h_values=(0.01, 0.1, 1.0, 10.0),
        noise_seed: int = 0,
    ):
        super().__init__()
        self.eval_every_n_epochs = int(eval_every_n_epochs)
        self.image_indices = tuple(image_indices)
        self.h_values = tuple(float(h) for h in h_values)
        self.noise_seed = int(noise_seed)
        self._clean_images = None
        self._noise = None

    def setup(self, trainer, pl_module, stage: str = None):
        if stage not in (None, "fit"):
            return

        val_ds = trainer.datamodule.val_dataloader().dataset
        clean_images = []
        for idx in self.image_indices:
            image, _ = val_ds[idx]
            clean_images.append(image.detach().cpu())
        self._clean_images = torch.stack(clean_images, dim=0)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.noise_seed)
        self._noise = torch.randn(
            self._clean_images.shape,
            generator=generator,
            dtype=self._clean_images.dtype,
        )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        exp = wandb_experiment(trainer)
        if exp is None or self._clean_images is None or self._noise is None:
            return

        clean_images = self._clean_images.to(pl_module.device)
        noise = self._noise.to(pl_module.device)
        pred_clean_by_h = {}

        with torch.no_grad():
            for h_value in self.h_values:
                noisy_images = clean_images + h_value * noise
                h = torch.full(
                    (clean_images.shape[0], 1),
                    h_value,
                    device=pl_module.device,
                    dtype=clean_images.dtype,
                )
                pred_clean = pl_module(
                    h, noisy_images.reshape(noisy_images.shape[0], -1)
                )
                pred_clean_by_h[h_value] = pred_clean.reshape_as(clean_images).detach().cpu()

        log_dict = {"epoch": trainer.current_epoch}
        for plot_idx, sample_idx in enumerate(self.image_indices):
            figure = self._plot_sample(
                sample_idx=sample_idx,
                clean_image=self._clean_images[plot_idx],
                pred_clean_by_h={
                    h_value: pred_clean[plot_idx]
                    for h_value, pred_clean in pred_clean_by_h.items()
                },
            )
            log_dict[f"{self.log_prefix}/image_{sample_idx}"] = wandb.Image(figure)
            plt.close(figure)

        exp.log(log_dict, commit=True)

    def _plot_sample(self, sample_idx: int, clean_image: torch.Tensor, pred_clean_by_h: dict):
        num_cols = len(self.h_values) + 1
        fig, axes = plt.subplots(1, num_cols, figsize=(3.5 * num_cols, 3.8))

        axes[0].imshow(self._to_display_image(clean_image))
        axes[0].set_title("Original")
        axes[0].axis("off")

        for axis, h_value in zip(axes[1:], self.h_values):
            axis.imshow(self._to_display_image(pred_clean_by_h[h_value]))
            axis.set_title(f"Predicted clean\nh={h_value:g}")
            axis.axis("off")

        plt.suptitle(
            f"{self.dataset_label} denoising predictions for validation image {sample_idx}"
        )
        plt.tight_layout()
        return fig

    def _to_display_image(self, image: torch.Tensor):
        image = (image * 0.5 + 0.5).clamp(0.0, 1.0)
        return image.permute(1, 2, 0).numpy()
