import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from metric_matching.callbacks.wandb_utils import wandb_experiment


class CIFARSpectralVizCallback(Callback):
    """Log CIFAR-10 eigenvalue decays and tangent-vector preview images to W&B."""

    def __init__(
        self,
        eval_every_n_epochs: int = 1,
        image_indices=(0, 2, 4, 7, 8),
        h_value: float = 1.0,
        max_eigvals_to_plot: int = 40,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.image_indices = tuple(image_indices)
        self.h_value = float(h_value)
        self.max_eigvals_to_plot = int(max_eigvals_to_plot)
        self._val_images = None
        self._val_labels = None

    def setup(self, trainer, pl_module, stage: str = None):
        if stage not in (None, "fit"):
            return

        val_ds = trainer.datamodule.val_dataloader().dataset
        imgs = []
        labels = []
        for idx in self.image_indices:
            img, label = val_ds[idx]
            imgs.append(img.detach().cpu())
            labels.append(int(label))

        self._val_images = torch.stack(imgs, dim=0)  # [B, 3, 32, 32]
        self._val_labels = labels

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
        imgs = self._val_images.to(device)
        x_flat = imgs.reshape(imgs.shape[0], -1)
        h = torch.full(
            (imgs.shape[0], 1), self.h_value, device=device, dtype=x_flat.dtype
        )

        with torch.no_grad():
            out_h = pl_module(h, x_flat)
            if pl_module.smart_training:
                pred_metric = torch.einsum("bkd,bke->bde", out_h, out_h)
            else:
                pred_metric = out_h
            pred_metric = pred_metric.detach().cpu()
            evals, evecs = torch.linalg.eigh(pred_metric)
            evals = evals.flip(-1).detach().cpu()
            evecs = evecs.flip(-1).detach().cpu()

        decay_image = self._plot_eigenvalue_decay(evals)
        preview_images = self._plot_tangent_previews(
            self._val_images, self._val_labels, evals, evecs
        )

        log_dict = {
            "cifar_viz/eigenvalue_decay": wandb.Image(decay_image),
            "epoch": trainer.current_epoch,
        }
        for i, img in enumerate(preview_images):
            sample_idx = self.image_indices[i]
            log_dict[f"cifar_viz/tangent_previews/image_{sample_idx}"] = wandb.Image(
                img
            )

        exp.log(log_dict, commit=True)

    def _plot_eigenvalue_decay(self, evals: torch.Tensor):
        evals_np = evals.numpy()
        bsz = evals_np.shape[0]
        ncols = max(1, bsz)
        nrows = 1
        max_eigval = min(self.max_eigvals_to_plot, evals_np.shape[1])

        yvals = [e[:max_eigval] for e in evals_np]
        ymin = min(float(y.min()) for y in yvals)
        ymax = max(float(y.max()) for y in yvals)
        ymin = max(ymin, 1e-12)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = [axes] if ncols == 1 else axes.flatten()

        for i in range(bsz):
            axes[i].plot(yvals[i].flatten(), marker="o")
            axes[i].set_title(f"Val idx {self.image_indices[i]}")
            axes[i].set_xlabel("Eigenvalue index")
            axes[i].set_ylabel("Eigenvalue")
            axes[i].set_ylim(ymin, ymax)
            axes[i].set_yscale("log")

        plt.tight_layout()
        return fig

    @staticmethod
    def _denorm_cifar(img_chw: torch.Tensor) -> torch.Tensor:
        return (img_chw * 0.5 + 0.5).clamp(0.0, 1.0)

    def _plot_tangent_previews(
        self,
        imgs: torch.Tensor,
        labels,
        evals: torch.Tensor,
        evecs: torch.Tensor,
    ):
        out = []
        indices_to_plot = [0, 1, 2, 3, -1]

        for i in range(imgs.shape[0]):
            ncols = len(indices_to_plot) + 1
            fig, axes = plt.subplots(3, ncols, figsize=(3.2 * ncols, 8.2))
            axes = axes.reshape(3, ncols)

            orig = self._denorm_cifar(imgs[i]).permute(1, 2, 0).numpy()
            axes[0, 0].imshow(orig)
            axes[0, 0].set_title(f"Image\n(label={labels[i]})")
            axes[0, 0].axis("off")
            axes[1, 0].imshow(orig)
            axes[1, 0].set_title("Original")
            axes[1, 0].axis("off")
            axes[2, 0].axis("off")
            axes[2, 0].set_title("Signed tangent")

            vecs = evecs[i]
            vecs = vecs - vecs.mean(dim=0, keepdim=True)
            s = torch.quantile(vecs.abs().reshape(-1), 0.99).item()
            s = max(s, 1e-8)
            alpha = 0.25 / s
            signed_mags = []

            for j, k in enumerate(indices_to_plot):
                vec = vecs[:, k].reshape(3, 32, 32)
                eigval = float(evals[i][k])

                plus = (imgs[i] + alpha * vec).clamp(-1.0, 1.0)
                minus = (imgs[i] - alpha * vec).clamp(-1.0, 1.0)

                plus_vis = self._denorm_cifar(plus)
                minus_vis = self._denorm_cifar(minus)
                signed_mags.append((plus_vis - minus_vis).sum(dim=0))

                axes[0, j + 1].imshow(plus_vis.permute(1, 2, 0).numpy())
                axes[0, j + 1].set_title(f"x + av{k}\n(eig={eigval:.2f})")
                axes[0, j + 1].axis("off")

                axes[1, j + 1].imshow(minus_vis.permute(1, 2, 0).numpy())
                axes[1, j + 1].set_title(f"x - av{k}\n(a={alpha:.3g})")
                axes[1, j + 1].axis("off")

            signed_scale = torch.quantile(
                torch.stack(signed_mags).abs().reshape(-1), 0.99
            ).item()
            signed_scale = max(signed_scale, 1e-8)

            for j, k in enumerate(indices_to_plot):
                axes[2, j + 1].imshow(
                    signed_mags[j].numpy(),
                    cmap="bwr",
                    vmin=-signed_scale,
                    vmax=signed_scale,
                )
                axes[2, j + 1].set_title(f"(x+av) - (x-av)\nv{k}")
                axes[2, j + 1].axis("off")

            plt.suptitle(
                f"Val idx {self.image_indices[i]} - tangent-vector addition previews"
            )
            plt.tight_layout()
            out.append(fig)

        return out
