import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from metric_matching.utils.spectra import batched_eig_of_UtU
from metric_matching.callbacks.wandb_utils import wandb_experiment


class CelebAVizCallback(Callback):
    """Log CelebA eigendirections and image perturbations to W&B."""

    def __init__(
        self,
        eval_every_n_epochs: int = 10,
        image_indices=(0, 1, 2, 3),
        h_values=(1.0, 5.0, 10.0),
        eigenvector_indices=(0, 1, 2, 3),
        perturb_scale: float = 12.0,
    ):
        super().__init__()
        self.eval_every_n_epochs = int(eval_every_n_epochs)
        self.image_indices = tuple(image_indices)
        self.h_values = tuple(float(h) for h in h_values)
        self.eigenvector_indices = tuple(eigenvector_indices)
        self.perturb_scale = float(perturb_scale)
        self._val_images = None

    def setup(self, trainer, pl_module, stage: str = None):
        if stage not in (None, "fit"):
            return

        val_ds = trainer.datamodule.val_dataloader().dataset
        images = []
        for idx in self.image_indices:
            image, _ = val_ds[idx]
            images.append(image.detach().cpu())
        self._val_images = torch.stack(images, dim=0)

    @rank_zero_only
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
        with torch.no_grad():
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
                    log_key = (
                        f"celeba_viz/h_{self._format_h_value(h_value)}/image_{sample_idx}"
                    )
                    log_dict[log_key] = wandb.Image(figure)
                    plt.close(figure)

        exp.log(log_dict, commit=True)

    def _compute_eigendecomposition(self, out_h: torch.Tensor, pl_module):
        if pl_module.smart_training:
            evals, evecs = batched_eig_of_UtU(out_h)
        else:
            metric = out_h
            evals, evecs = torch.linalg.eigh(metric)
            evals = evals.flip(-1)
            evecs = evecs.flip(-1)
        return evals, evecs

    def _plot_sample(
        self,
        sample_idx: int,
        image: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        h_value: float,
    ):
        resolved_indices = self._resolve_eigenvector_indices(evals.shape[0])
        grayscale_vectors = [
            self._to_grayscale_image(evecs[:, eig_idx].reshape_as(image))
            for eig_idx in resolved_indices
        ]
        grayscale_scale = self._compute_grayscale_scale(grayscale_vectors)

        fig, axes = plt.subplots(
            3,
            len(resolved_indices) + 1,
            figsize=(3.4 * (len(resolved_indices) + 1), 8.6),
        )

        axes[0, 0].imshow(self._to_display_image(image))
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.0,
            0.9,
            f"val idx: {sample_idx}\nh: {h_value:.2f}\nperturbation scale: {self.perturb_scale:.1f}",
            va="top",
            ha="left",
            fontsize=11,
        )

        axes[2, 0].axis("off")

        for col, eig_idx in enumerate(resolved_indices, start=1):
            eigenvalue = float(evals[eig_idx])
            eigenvector = evecs[:, eig_idx].reshape_as(image)
            positive = (image + self.perturb_scale * eigenvector).clamp(-1.0, 1.0)
            negative = (image - self.perturb_scale * eigenvector).clamp(-1.0, 1.0)
            grayscale = self._to_grayscale_image(eigenvector)

            axes[0, col].imshow(self._to_display_image(positive))
            axes[0, col].set_title(f"x + v{eig_idx}\n(eig={eigenvalue:.2f})")
            axes[0, col].axis("off")

            axes[1, col].imshow(self._to_display_image(negative))
            axes[1, col].set_title(f"x - v{eig_idx}")
            axes[1, col].axis("off")

            axes[2, col].imshow(
                grayscale.numpy(),
                cmap="gray",
                vmin=-grayscale_scale,
                vmax=grayscale_scale,
            )
            axes[2, col].set_title(f"evec {eig_idx}")
            axes[2, col].axis("off")

        plt.suptitle(f"CelebA eigendirections for validation image {sample_idx}")
        plt.tight_layout()
        return fig

    def _resolve_eigenvector_indices(self, num_eigenvectors: int):
        resolved = []
        for idx in self.eigenvector_indices:
            resolved_idx = idx if idx >= 0 else num_eigenvectors + idx
            if 0 <= resolved_idx < num_eigenvectors:
                resolved.append(resolved_idx)
        if not resolved:
            raise ValueError("No valid eigenvector indices were selected for visualization.")
        return resolved

    def _to_display_image(self, image: torch.Tensor):
        image = (image * 0.5 + 0.5).clamp(0.0, 1.0)
        return image.permute(1, 2, 0).numpy()

    def _to_grayscale_image(self, image: torch.Tensor):
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        grayscale = (image * weights).sum(dim=0)
        return grayscale - grayscale.mean()

    def _compute_grayscale_scale(self, grayscale_vectors):
        stacked = torch.stack(grayscale_vectors, dim=0)
        scale = torch.quantile(stacked.abs().reshape(-1), 0.99).item()
        return max(scale, 1e-8)

    def _format_h_value(self, h_value: float) -> str:
        return f"{h_value:g}".replace("-", "neg").replace(".", "p")
