"""Conditional matching classes for metric and score matching.

This module provides classes for sampling conditional perturbations and computing
conditional targets (metrics or score differences) for training neural networks.
"""

import math
from typing import Tuple, Union

import torch
from torch import Tensor
from torch import nn


def pad_h_like_x(h: Union[float, int, Tensor], x: Tensor) -> Union[float, int, Tensor]:
    """Reshape bandwidth h to broadcast with x.

    Example: x: (bs, C, W, H), h: (bs,) -> (bs, 1, 1, 1)
    """
    if isinstance(h, (float, int)):
        return h
    # Keep the batch axis and add singleton dimensions for every sample axis in x.
    return h.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalMatchingBase:
    """Base class for conditional matching with shared h-sampling logic.

    Samples noisy locations xh ~ N(x1, h²I) and provides methods for
    computing conditional targets.
    """

    SAMPLING_METHODS = {"uniform", "lognormal", "lognormalwide"}
    _LOGNORMAL_PARAMS = {"lognormal": (-1.2, 1.2), "lognormalwide": (0.0, 1.0)}

    def __init__(
        self,
        h_max: float = 1.0,
        h_min: float = 0.001,
        sampling_method: str = "uniform",
    ):
        """
        Args:
            h_max: Maximum bandwidth value.
            h_min: Minimum bandwidth value.
            sampling_method: One of "uniform", "lognormal", "lognormalwide".
        """
        if sampling_method not in self.SAMPLING_METHODS:
            raise ValueError(f"sampling_method must be one of: {self.SAMPLING_METHODS}")
        self.h_max = h_max
        self.h_min = h_min
        self.sampling_method = sampling_method

    def _sample_h(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Sample bandwidth values according to sampling_method."""
        if self.sampling_method == "uniform":
            return (
                torch.rand(batch_size, device=device, dtype=dtype)
                * (self.h_max - self.h_min)
                + self.h_min
            )

        p_mean, p_std = self._LOGNORMAL_PARAMS[self.sampling_method]
        log_sigma = p_mean + p_std * torch.randn(batch_size, device=device, dtype=dtype)
        return torch.exp(log_sigma).clamp(self.h_min, self.h_max)

    def _resolve_h(self, h: Tensor | None, x1: Tensor) -> Tensor:
        """Return provided h or sample a new one."""
        if h is None:
            return self._sample_h(x1.shape[0], x1.device, x1.dtype)
        assert len(h) == x1.shape[0], "h must have batch size dimension"
        return h

    def _sample_xh(self, x1: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample xh ~ N(x1, h²I). Returns (noise, xh)."""
        # x1: [bs, *sample_shape], h: [bs] -> broadcast h across sample dimensions.
        eps = torch.randn_like(x1) * pad_h_like_x(h, x1)
        return eps, x1 + eps


    def sample_location_and_conditional_diff(
        self, x1: Tensor, normalize: bool = True, h: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample xh and compute the conditional difference vector.

        Args:
            x1: Target minibatch, shape (bs, *dim).
            normalize: If True, normalize diff by sqrt(2) * h.
            h: Optional bandwidth values, shape (bs,).

        Returns:
            h: Bandwidth values, shape (bs,).
            xh: Noisy samples, shape (bs, *dim).
            diff: Conditional difference, shape (bs, *dim).
        """
        h = self._resolve_h(h, x1)  # [bs]
        eps, xh = self._sample_xh(x1, h)  # both [bs, *dim]
        diff = -eps  # [bs, *dim]
        if normalize:
            diff = diff / (math.sqrt(2) * pad_h_like_x(h, x1))
        return h, xh, diff


class ConditionalScoreMatching(ConditionalMatchingBase):
    """Conditional score matching - samples perturbations and score targets."""

    pass


class ConditionalMetricMatching(ConditionalMatchingBase):
    """Conditional metric matching - extends base with metric tensor computation."""

    def sample_location_and_conditional_metric(
        self, x1: Tensor, normalize: bool = True, h: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample xh and compute the conditional metric tensor.

        Args:
            x1: Target minibatch, shape (bs, D).
            normalize: If True, normalize metric by 2h².
            h: Optional bandwidth values, shape (bs,).

        Returns:
            h: Bandwidth values, shape (bs,).
            xh: Noisy samples, shape (bs, D).
            gh: Conditional metric tensor, shape (bs, D, D).
        """
        h = self._resolve_h(h, x1)
        eps, xh = self._sample_xh(x1, h)
        gh = self._compute_metric(eps, h, normalize)
        return h, xh, gh

    def _compute_metric(self, diff: Tensor, h: Tensor, normalize: bool) -> Tensor:
        """Compute outer product metric from difference vector."""
        # Metric matching works on flattened samples: diff is [bs, D].
        # This forms one outer product per batch item, producing [bs, D, D].
        gt = torch.einsum("bi,bj->bij", diff, diff)
        if normalize:
            gt = gt / (2 * h.unsqueeze(-1).unsqueeze(-1) ** 2)
        return gt

    def compute_conditional_metric(
        self, x1: Tensor, xh: Tensor, h: Tensor, normalize: bool = True
    ) -> Tensor:
        """Compute conditional metric gt = (xh - x1)(xh - x1)^T / 2h².

        Args:
            x1: Target samples, shape (bs, D).
            xh: Noisy samples, shape (bs, D).
            h: Bandwidth values, shape (bs,).
            normalize: If True, divide by 2h².

        Returns:
            gt: Conditional metric tensor, shape (bs, D, D).
        """
        return self._compute_metric(xh - x1, h, normalize)


class MeanCenteredMetricMatching(ConditionalMetricMatching):
    """Conditional metric matching with score-based mean centering.

    The pretrained score/denoiser model predicts the conditional mean of clean samples
    E[x1 | xh, h]. We use this to estimate and subtract E[diff | xh, h], yielding a
    centered target that matches the mean-centered second-moment objective.
    """

    def __init__(
        self,
        score_model: nn.Module,
        h_max: float = 1.0,
        h_min: float = 0.001,
        sampling_method: str = "uniform",
    ):
        super().__init__(h_max=h_max, h_min=h_min, sampling_method=sampling_method)
        self.score_model = score_model

    def _predict_conditional_mean(self, xh: Tensor, h: Tensor) -> Tensor:
        """Predict the conditional clean-sample mean E[x1 | xh, h].

        Supported score model outputs are [bs, D] and [bs, 1, D].
        """
        with torch.no_grad():
            pred = self.score_model(h, xh)
            if pred.ndim == 2:
                pred_clean = pred
            elif pred.ndim == 3 and pred.shape[1] == 1:
                pred_clean = pred[:, 0, :]
            else:
                raise ValueError(
                    "score_model(h, xh) must return shape [bs, D] or [bs, 1, D]; "
                    f"got {tuple(pred.shape)}"
                )
            if pred_clean.shape != xh.shape:
                raise ValueError(
                    "Predicted conditional mean must match xh shape; "
                    f"got {tuple(pred_clean.shape)} vs {tuple(xh.shape)}"
                )
        return pred_clean

    def sample_location_and_conditional_metric(
        self, x1: Tensor, normalize: bool = True, h: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample xh and compute mean-centered conditional metric tensor."""
        h, xh, centered_diff = self.sample_location_and_conditional_diff(
            x1, normalize=normalize, h=h
        )
        # centered_diff is already normalized when requested, so do not divide by h again.
        gh = self._compute_metric(centered_diff, h, normalize=False)
        return h, xh, gh

    def sample_location_and_conditional_diff(
        self, x1: Tensor, normalize: bool = True, h: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return x1 - E[x1 | xh, h], optionally normalized by sqrt(2) * h."""
        h = self._resolve_h(h, x1)
        _, xh = self._sample_xh(x1, h)
        pred_clean = self._predict_conditional_mean(xh=xh, h=h)
        centered_diff = x1 - pred_clean
        if normalize:
            centered_diff = centered_diff / (math.sqrt(2) * pad_h_like_x(h, x1))
        return h, xh, centered_diff
