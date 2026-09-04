import torch
import pytest
from torch import nn

from metric_matching.losses.matcher import MeanCenteredMetricMatching
from metric_matching.losses.matcher import pad_h_like_x


class ZeroScore(nn.Module):
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class ZeroScoreSingletonRank(nn.Module):
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x).unsqueeze(1)


class BadShapeScore(nn.Module):
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 2, x.shape[1], dtype=x.dtype, device=x.device)


def test_mean_centered_diff_matches_expected_with_zero_score_model():
    matcher = MeanCenteredMetricMatching(
        score_model=ZeroScore(), h_min=0.5, h_max=0.5, sampling_method="uniform"
    )
    x1 = torch.randn(6, 4)
    h = torch.full((6,), 0.5)

    _, _, centered_diff = matcher.sample_location_and_conditional_diff(x1, normalize=False, h=h)

    assert torch.allclose(centered_diff, x1, atol=1e-6)


def test_mean_centered_diff_accepts_singleton_rank_score_output():
    matcher = MeanCenteredMetricMatching(
        score_model=ZeroScoreSingletonRank(),
        h_min=0.5,
        h_max=0.5,
        sampling_method="uniform",
    )
    x1 = torch.randn(6, 4)
    h = torch.full((6,), 0.5)

    _, _, centered_diff = matcher.sample_location_and_conditional_diff(x1, normalize=False, h=h)

    assert torch.allclose(centered_diff, x1, atol=1e-6)


def test_mean_centered_diff_normalization_matches_raw_diff():
    matcher = MeanCenteredMetricMatching(
        score_model=ZeroScore(), h_min=0.5, h_max=0.5, sampling_method="uniform"
    )
    x1 = torch.randn(4, 3)
    h = torch.full((4,), 0.5)

    torch.manual_seed(0)
    _, _, centered_diff_raw = matcher.sample_location_and_conditional_diff(
        x1, normalize=False, h=h
    )
    torch.manual_seed(0)
    _, _, centered_diff_norm = matcher.sample_location_and_conditional_diff(
        x1, normalize=True, h=h
    )

    expected = centered_diff_raw / (torch.sqrt(torch.tensor(2.0)) * pad_h_like_x(h, x1))
    assert torch.allclose(centered_diff_norm, expected, atol=1e-6)


def test_mean_centered_metric_is_outer_of_centered_diff():
    matcher = MeanCenteredMetricMatching(
        score_model=ZeroScore(), h_min=0.5, h_max=0.5, sampling_method="uniform"
    )
    x1 = torch.randn(4, 3)
    h = torch.full((4,), 0.5)

    torch.manual_seed(0)
    _, _, centered_diff = matcher.sample_location_and_conditional_diff(
        x1, normalize=False, h=h
    )
    torch.manual_seed(0)
    _, _, gh = matcher.sample_location_and_conditional_metric(x1, normalize=False, h=h)

    expected = torch.einsum("bi,bj->bij", centered_diff, centered_diff)
    assert torch.allclose(gh, expected, atol=1e-6)


def test_mean_centered_diff_rejects_invalid_score_output_shape():
    matcher = MeanCenteredMetricMatching(
        score_model=BadShapeScore(), h_min=0.5, h_max=0.5, sampling_method="uniform"
    )
    x1 = torch.randn(4, 3)
    h = torch.full((4,), 0.5)

    with pytest.raises(ValueError, match=r"\[bs, D\] or \[bs, 1, D\]"):
        matcher.sample_location_and_conditional_diff(x1, normalize=False, h=h)
