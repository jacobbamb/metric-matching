import torch
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from tqdm import tqdm
import torch.nn.functional as F

from metric_matching.utils.spectra import batched_eig_of_UtU
from metric_matching.callbacks.wandb_utils import wandb_experiment


def _extract_x(batch, single_item: bool = False):
    """Extract x and flatten each dataset item into a single row."""
    if isinstance(batch, (list, tuple)):
        x = batch[
            0
        ]  # either shape [B,d] or [B,channels,height,width] or [B,channels,time,height,width]
        if single_item:
            return x.reshape(1, -1)
        if len(x.shape) == 3:
            # flatten time dimension if present
            x = x.reshape(x.shape[0], -1)
            return x
        elif len(x.shape) == 4:
            x = x.reshape(x.shape[0], -1)
            return x
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            return x
        else:
            return x
    else:
        raise ValueError("Unsupported batch type")


def _tensorize_subset(subset_ds):
    xs = [_extract_x(subset_ds[i], single_item=True) for i in range(len(subset_ds))]
    return torch.cat(xs, dim=0)  # [m,d]


class SubsetMarginalCallback(Callback):
    def __init__(
        self,
        bandwidths: list,
        subset_size: int,
        batch_size=128,
        eval_every_n_epochs=1,
        device="cuda",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_every_n_epochs = eval_every_n_epochs
        self.train_subset_size = subset_size
        self.val_subset_size = subset_size
        self.device = device

        self.h = bandwidths

        self.train_cdc = None
        self.val_cdc = None
        self.val_subset = None
        self.train_subset = None
        self.subset_indices_train = None
        self.subset_indices_val = None

    # -------------------------------
    # 0) Tell W&B to use epoch for subset/* metrics
    # -------------------------------
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        exp = wandb_experiment(trainer)
        if exp is not None and hasattr(exp, "define_metric"):
            exp.define_metric("epoch")  # ensure epoch exists
            exp.define_metric("subset/*", step_metric="epoch")  # <-- key line

    # -------------------------------
    # 1. Precompute distances once
    # -------------------------------
    def setup(self, trainer, pl_module, stage: str = None):
        print("Setting up SubsetMarginalCallback ...")

        train_ds = trainer.datamodule.train_dataloader().dataset
        val_ds = trainer.datamodule.val_dataloader().dataset

        self.subset_indices_train = torch.randperm(len(train_ds))[
            : self.train_subset_size
        ].tolist()
        self.subset_indices_val = torch.randperm(len(val_ds))[
            : self.val_subset_size
        ].tolist()

        self.train_subset = Subset(train_ds, self.subset_indices_train)
        self.val_subset = Subset(val_ds, self.subset_indices_val)

        print("[SubsetMarginalCallback] Precomputing subset cdc ...")
        self.train_cdc = self._compute_cdc(
            self.train_subset, train_ds, pl_module
        )  # [m, H, d, d]
        self.val_cdc = self._compute_cdc(
            self.val_subset, train_ds, pl_module
        )  # [m, H, d, d]
        print("[SubsetMarginalCallback] Done precomputing.")

    # -------------------------------
    # 2. Compute CDC
    # -------------------------------
    def _compute_cdc(self, subset_ds, full_ds, pl_module=None):
        print("computing cdc for subset of size", len(subset_ds))
        device = torch.device(self.device)

        full_loader = DataLoader(full_ds, batch_size=self.batch_size, shuffle=False)
        x_b = _tensorize_subset(subset_ds).to(device)  # [m,d]
        m, d = x_b.shape

        h_vec = torch.tensor(self.h, device=device).view(1, 1, -1)
        h2 = h_vec**2  # [1,1,H]

        with torch.no_grad():
            all_cdc = torch.zeros(
                [m, len(self.h), pl_module.out_dim, pl_module.out_dim],
                dtype=torch.float32,
                device=device,
            )
            total_mass = torch.zeros(
                [m, len(self.h)], dtype=torch.float32, device=device
            )
            for a_batch in full_loader:
                x_a = _extract_x(a_batch).to(device)  # [b, d]
                dist = torch.cdist(x_a, x_b, p=2)  # [b, m]
                rbf = torch.exp(-(dist.unsqueeze(-1) ** 2) / (2 * h2))  # [b, m, H]
                total_mass = total_mass + rbf.sum(dim=0)  # [m, H]
                if (total_mass > 1e8).any():
                    print(
                        "[Warning] total mass exceeded 1e8, consider increasing bandwidths."
                    )
                diff = x_b[None, :, :] - x_a[:, None, :]  # [b, m, d]
                all_cdc = all_cdc + torch.einsum("bmh,bmi,bmj->mhij", rbf, diff, diff)

            denom = total_mass.clamp_min(1e-12)
            if (total_mass < 1e-8).any():
                print(
                    "[Warning] very small kernel mass for some (m,h); results may be noisy."
                )
            normalized_cdc = all_cdc / denom.unsqueeze(-1).unsqueeze(-1)  # [m, H, d, d]
            normalized_cdc = torch.einsum(
                "mhij,abh->mhij", normalized_cdc, 1 / (2 * h2)
            )  # 1/(2h^2)

        return normalized_cdc.cpu()

    # -------------------------------
    # 3. Periodic evaluation
    # -------------------------------
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        model = pl_module.to(self.device)
        metrics_train = self._evaluate_subset(model, "train", pl_module)
        metrics_val = self._evaluate_subset(model, "val", pl_module)

        # Scalars go through pl_module.log_dict so they reach any logger (or
        # none); Lightning attaches "epoch" to each record for the W&B x-axis.
        log_dict = {}
        for idx, loss in enumerate(metrics_train):
            log_dict[f"subset/train_loss_h_{self.h[idx]}"] = float(loss)
        for idx, loss in enumerate(metrics_val):
            log_dict[f"subset/val_loss_h_{self.h[idx]}"] = float(loss)
        pl_module.log_dict(log_dict, on_step=False, on_epoch=True)

    # -------------------------------
    # 4. Example subset evaluation
    # -------------------------------
    def _evaluate_subset(self, model, name, pl_module=None):
        all_loss = []
        if name == "train":
            cdc = self.train_cdc
            subset = self.train_subset
        else:
            cdc = self.val_cdc
            subset = self.val_subset
        m = len(subset)

        x_subset = _tensorize_subset(subset).to(self.device)  # [m,d]

        for i, h in enumerate(self.h):
            h_tensor = torch.tensor([h], device=self.device).repeat(m, 1)  # [m,1]
            out_h = model(
                h_tensor, x_subset
            )  # returns U [m, rank, out_dim] or [m, out_dim, out_dim]
            if pl_module.smart_training:
                pred_cdc_h = torch.einsum(
                    "mki,mkj->mij", out_h, out_h
                )  # returns U^TU [m, out_dim, out_dim]
            else:
                pred_cdc_h = out_h
            loss_h = F.mse_loss(
                pred_cdc_h.reshape([m, pl_module.out_dim * pl_module.out_dim]),
                cdc[:, i, :, :]
                .reshape([m, pl_module.out_dim * pl_module.out_dim])
                .to(self.device),
            )
            all_loss.append(loss_h.item())
        return all_loss


class SubsetSpectralCallback(Callback):
    def __init__(
        self,
        subset_size: int,
        batch_size=128,
        eval_every_n_epochs=1,
        device="cuda",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_every_n_epochs = eval_every_n_epochs
        self.train_subset_size = subset_size
        self.val_subset_size = subset_size
        self.device = device

        self.h = [(2 ** (i - 6)) for i in range(11)]
        # logspace base 2 from 0.0078 to 16

        self.val_subset = None
        self.train_subset = None
        self.subset_indices_train = None
        self.subset_indices_val = None

    # -------------------------------
    # 0) Tell W&B to use epoch for subset/* metrics
    # -------------------------------
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        exp = wandb_experiment(trainer)
        if exp is not None and hasattr(exp, "define_metric"):
            exp.define_metric("epoch")  # ensure epoch exists
            exp.define_metric("subset/*", step_metric="epoch")

    # -------------------------------
    # 1. Precompute distances once
    # -------------------------------
    def setup(self, trainer, pl_module, stage: str = None):
        print("Setting up SubsetMarginalCallback ...")

        train_ds = trainer.datamodule.train_dataloader().dataset
        val_ds = trainer.datamodule.val_dataloader().dataset

        self.subset_indices_train = torch.randperm(len(train_ds))[
            : self.train_subset_size
        ].tolist()
        self.subset_indices_val = torch.randperm(len(val_ds))[
            : self.val_subset_size
        ].tolist()

        self.train_subset = Subset(train_ds, self.subset_indices_train)
        self.val_subset = Subset(val_ds, self.subset_indices_val)

    # # -------------------------------
    # # 2. Compute Spectra
    # # -------------------------------
    def _get_spectra(self, model, name, pl_module=None):
        all_evals = torch.zeros(
            [len(self.train_subset), len(self.h), pl_module.out_dim],
            dtype=torch.float32,
        )
        if name == "train":
            subset = self.train_subset
        else:
            subset = self.val_subset
        m = len(subset)

        x_subset = _tensorize_subset(subset).to(self.device)  # [m,d]

        for i, h in enumerate(self.h):
            h_tensor = torch.tensor([h], device=self.device).repeat(m, 1)  # [m,1]
            out_h = model(h_tensor, x_subset)  # returns U [m, d, rank]
            if pl_module.smart_training:
                evals, evecs = batched_eig_of_UtU(out_h)  # [m, rank], [m, rank, rank]
                evals = torch.cat(
                    [
                        evals,
                        torch.zeros(
                            m, pl_module.out_dim - evals.shape[1], device=self.device
                        ),
                    ],
                    1,
                )  # pad to [m, d]
            else:
                evals, evecs = torch.linalg.eigh(out_h)  # [m, d], [m, d, d]
                evals = evals.flip(-1)  # [m, d]
                evecs = evecs.flip(-1)  # [m, d, d]
            all_evals[:, i, :] = evals.cpu()
        return all_evals  # [m, H, d]

    # # -------------------------------
    # # 3. Compute eigenvalues stats
    # # -------------------------------

    def _get_eval_stats(self, evals):
        mean_evals = evals.mean(dim=0)  # [H, d]
        # std_evals = evals.std(dim=0)  # [H, d]
        mean_trace = evals.sum(dim=2).mean(dim=0)  # [H]
        # std_trace = evals.sum(dim=1).std(dim=0)  # [H]
        mean_frob_norms = (evals**2).sum(dim=2).sqrt().mean(dim=0)  # [H]
        # std_frob_norms = (evals**2).sum(dim=1).sqrt().std(dim=0)  # [H]
        mean_stable_rank = (
            (evals**2).sum(dim=2) / (evals.max(dim=2).values ** 2)
        ).mean(
            dim=0
        )  # [H]
        norm_const = evals.sum(dim=2)  # [m, H]
        eps = 1e-12
        p = evals / norm_const.unsqueeze(-1)  # [m, H, d]
        entropy = -(p * torch.log(p + eps)).sum(dim=2)  # [m, H]
        effective_rank = torch.exp(entropy)  # [m, H]
        mean_effective_rank = effective_rank.mean(dim=0)  # [H]
        return (
            mean_evals,
            mean_trace,
            mean_frob_norms,
            mean_stable_rank,
            mean_effective_rank,
        )

    # -------------------------------
    # 3. Periodic evaluation
    # -------------------------------
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        model = pl_module.to(self.device)
        log_dict = {}
        for subset_name in ["train", "val"]:
            evals = self._get_spectra(model, subset_name, pl_module)
            (
                mean_evals,
                mean_trace,
                mean_frob_norms,
                mean_stable_rank,
                mean_effective_rank,
            ) = self._get_eval_stats(evals)
            for i, h in enumerate(self.h):
                for k in range(0, min(10, pl_module.rank)):

                    log_dict[f"spectra/{subset_name}/h_{h}_eval_{k}_mean"] = mean_evals[
                        i, k
                    ].item()

                log_dict[f"spectra/{subset_name}/h_{h}_trace_mean"] = mean_trace[
                    i
                ].item()
                log_dict[f"spectra/{subset_name}/h_{h}_frob_mean"] = mean_frob_norms[
                    i
                ].item()
                log_dict[f"spectra/{subset_name}/h_{h}_stable_rank_mean"] = (
                    mean_stable_rank[i].item()
                )
                log_dict[f"spectra/{subset_name}/h_{h}_effective_rank_mean"] = (
                    mean_effective_rank[i].item()
                )
        pl_module.log_dict(log_dict, on_step=False, on_epoch=True)
