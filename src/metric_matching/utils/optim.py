from typing import Any, Dict, Iterable

import torch


def build_optimizer(params: Iterable, opt_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build the Adam/AdamW optimizer described by ``cfg.optimizer``."""
    opt = opt_cfg or {}

    name = opt.get("name", "adamw").lower()
    if name not in ("adam", "adamw"):
        raise ValueError("Only Adam and AdamW optimizers are supported.")

    lr = float(opt.get("lr", 1e-4))
    weight_decay = float(opt.get("weight_decay", 1e-2 if name == "adamw" else 0.0))
    betas = tuple(opt.get("betas", (0.9, 0.999)))
    eps = float(opt.get("eps", 1e-8))

    optim_cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
    return optim_cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, opt_cfg: Dict[str, Any], total_steps: int
):
    """Build the optional per-step schedule described by ``cfg.optimizer.lr_scheduler``.

    ``lr_scheduler`` may be omitted / ``null`` / ``"constant"`` (no schedule, the
    default) or ``"cosine"``: cosine annealing from ``lr`` down to ``lr_min``
    (default ``lr / 100``) over all ``total_steps`` optimizer steps.
    Returns ``None`` when no schedule is requested.
    """
    opt = opt_cfg or {}
    name = opt.get("lr_scheduler", None)
    if name is None or str(name).lower() in ("none", "constant"):
        return None
    if str(name).lower() != "cosine":
        raise ValueError("lr_scheduler must be null, 'constant' or 'cosine'.")
    lr = float(opt.get("lr", 1e-4))
    lr_min = float(opt.get("lr_min", lr / 100))
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(total_steps), eta_min=lr_min
    )
