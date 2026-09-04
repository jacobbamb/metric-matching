"""Denoising visualization callback for the FFHQ score model."""

from metric_matching.callbacks.celeba_score_viz import CelebAScoreVizCallback


class FFHQScoreVizCallback(CelebAScoreVizCallback):
    """Log fixed FFHQ denoising examples across multiple bandwidths."""

    log_prefix = "ffhq_score_viz"
    dataset_label = "FFHQ"
