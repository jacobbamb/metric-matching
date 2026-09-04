"""Helpers for optional W&B media logging from callbacks."""

from lightning.pytorch.loggers import WandbLogger


def wandb_experiment(trainer):
    """Return the W&B run behind ``trainer.logger``, or ``None``.

    ``None`` means W&B is not in use: no logger, a non-W&B logger, or a
    W&B logger whose run could not be created. Callers should skip media
    logging in that case. Scalar metrics should go through ``pl_module.log``
    instead, which works with any logger (or none).
    """
    logger = getattr(trainer, "logger", None)
    if not isinstance(logger, WandbLogger):
        return None
    try:
        exp = logger.experiment
    except Exception:
        return None
    return exp if hasattr(exp, "log") else None
