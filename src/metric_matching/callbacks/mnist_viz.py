import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from metric_matching.callbacks.wandb_utils import wandb_experiment


class MNISTSpectralVizCallback(Callback):
    """Log MNIST eigenvalue decays and eigenvector visualizations to W&B."""

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

        self._val_images = torch.stack(imgs, dim=0)  # [B, 1, 28, 28]
        self._val_labels = labels

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        exp = wandb_experiment(trainer)
        if exp is None:
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
            evals, evecs = torch.linalg.eigh(pred_metric)
            evals = evals.flip(-1).detach().cpu()
            evecs = evecs.flip(-1).detach().cpu()

        decay_image = self._plot_eigenvalue_decay(evals)
        evec_images = self._plot_eigenvectors(
            self._val_images, self._val_labels, evals, evecs
        )

        log_dict = {
            "mnist_viz/eigenvalue_decay": wandb.Image(decay_image),
            "epoch": trainer.current_epoch,
        }
        for i, img in enumerate(evec_images):
            sample_idx = self.image_indices[i]
            log_dict[f"mnist_viz/eigenvectors/image_{sample_idx}"] = wandb.Image(img)

        exp.log(log_dict, commit=True)

    def _plot_eigenvalue_decay(self, evals: torch.Tensor):
        evals_np = evals.numpy()
        bsz = evals_np.shape[0]
        ncols = 5
        nrows = 1
        max_eigval = min(self.max_eigvals_to_plot, evals_np.shape[1])

        yvals = [e[:max_eigval] for e in evals_np]
        ymin = min(float(y.min()) for y in yvals)
        ymax = max(float(y.max()) for y in yvals)
        ymin = max(ymin, 1e-12)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten()

        for i in range(bsz):
            axes[i].plot(yvals[i].flatten(), marker="o")
            axes[i].set_title(f"Val idx {self.image_indices[i]}")
            axes[i].set_xlabel("Eigenvalue index")
            axes[i].set_ylabel("Eigenvalue")
            axes[i].set_ylim(ymin, ymax)
            axes[i].set_yscale("log")

        for j in range(bsz, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        return fig

    def _plot_eigenvectors(
        self,
        imgs: torch.Tensor,
        labels,
        evals: torch.Tensor,
        evecs: torch.Tensor,
    ):
        out = []
        indices_to_plot = [0, 1, 2, 3, -1]

        for i in range(imgs.shape[0]):
            fig, axes = plt.subplots(1, len(indices_to_plot) + 1, figsize=(16, 3))
            axes = axes.flatten()

            orig = imgs[i].squeeze(0)
            orig = (orig * 0.3081 + 0.1307).clamp(0.0, 1.0)
            axes[0].imshow(orig.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[0].set_title(f"Image\n(label={labels[i]})")
            axes[0].axis("off")

            vecs = evecs[i]
            vecs = vecs - vecs.mean(dim=0, keepdim=True)
            s = torch.quantile(vecs.abs().reshape(-1), 0.99).item()
            s = max(s, 1e-8)

            for j, k in enumerate(indices_to_plot):
                vec = vecs[:, k]
                eigval = float(evals[i][k])
                axes[j + 1].imshow(
                    vec.reshape(28, 28).numpy(), cmap="gray", vmin=-s, vmax=s
                )
                axes[j + 1].set_title(f"evec {k}\n(eig={eigval:.2f})")
                axes[j + 1].axis("off")

            plt.suptitle(f"Val idx {self.image_indices[i]} - selected eigenvectors")
            plt.tight_layout()
            out.append(fig)

        return out
