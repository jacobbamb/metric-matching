"""Tests for ConditionalMetricMatching.

Run with: pytest tests/test_metric_matching.py -v
"""

import math

import pytest
import torch

from metric_matching.losses.matcher import (
    ConditionalMetricMatching,
    pad_h_like_x,
)


class TestPadHLikeX:
    def test_scalar_passthrough(self):
        x = torch.randn(4, 3, 8, 8)
        assert pad_h_like_x(1.0, x) == 1.0
        assert pad_h_like_x(3, x) == 3

    def test_1d_tensor_padded_to_4d(self):
        x = torch.randn(4, 3, 8, 8)
        h = torch.ones(4)
        result = pad_h_like_x(h, x)
        assert result.shape == (4, 1, 1, 1)

    def test_1d_tensor_padded_to_2d(self):
        x = torch.randn(4, 10)
        h = torch.ones(4)
        result = pad_h_like_x(h, x)
        assert result.shape == (4, 1)


class TestInit:
    @pytest.mark.parametrize("method", ["uniform", "lognormal", "lognormalwide"])
    def test_valid_sampling_methods(self, method):
        cmm = ConditionalMetricMatching(sampling_method=method)
        assert cmm.sampling_method == method

    def test_invalid_sampling_method_raises(self):
        with pytest.raises(ValueError):
            ConditionalMetricMatching(sampling_method="cosine")

    def test_default_parameters(self):
        cmm = ConditionalMetricMatching()
        assert cmm.h_max == 1.0
        assert cmm.h_min == 0.001
        assert cmm.sampling_method == "uniform"


class TestHSampling:
    @pytest.mark.parametrize("method", ["uniform", "lognormal", "lognormalwide"])
    def test_h_shape_and_bounds_via_diff(self, method):
        h_min, h_max = 0.01, 2.0
        cmm = ConditionalMetricMatching(
            h_min=h_min, h_max=h_max, sampling_method=method
        )
        x1 = torch.randn(64, 10)
        h, xh, _ = cmm.sample_location_and_conditional_diff(x1)
        assert h.shape == (64,)
        assert (h >= h_min).all()
        assert (h <= h_max).all()

    @pytest.mark.parametrize("method", ["uniform", "lognormal", "lognormalwide"])
    def test_h_shape_and_bounds_via_metric(self, method):
        """This specifically guards against the old lognormalwide bug."""
        h_min, h_max = 0.01, 2.0
        cmm = ConditionalMetricMatching(
            h_min=h_min, h_max=h_max, sampling_method=method
        )
        x1 = torch.randn(64, 10)
        h, xh, _ = cmm.sample_location_and_conditional_metric(x1)
        assert h.shape == (64,)
        assert (h >= h_min).all()
        assert (h <= h_max).all()

    def test_provided_h_is_used(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(8, 10)
        fixed_h = torch.full((8,), 0.5)
        h, _, _ = cmm.sample_location_and_conditional_diff(x1, h=fixed_h)
        assert torch.allclose(h, fixed_h)

    def test_h_batch_size_mismatch_raises(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(8, 10)
        wrong_h = torch.ones(4)
        with pytest.raises(AssertionError):
            cmm.sample_location_and_conditional_diff(x1, h=wrong_h)


class TestSampleMetric:
    def test_output_shapes(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(8, 10)
        h, xh, gh = cmm.sample_location_and_conditional_metric(x1)
        assert h.shape == (8,)
        assert xh.shape == (8, 10)
        assert gh.shape == (8, 10, 10)

    def test_gh_is_symmetric(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(8, 10)
        _, _, gh = cmm.sample_location_and_conditional_metric(x1)
        assert torch.allclose(gh, gh.transpose(1, 2), atol=1e-6)

    def test_gh_is_psd(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(8, 10)
        _, _, gh = cmm.sample_location_and_conditional_metric(x1)
        eigenvalues = torch.linalg.eigvalsh(gh)
        assert (eigenvalues >= -1e-6).all()

    def test_normalize_false(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(4, 5)
        h = torch.full((4,), 0.5)
        torch.manual_seed(42)
        _, xh, gh = cmm.sample_location_and_conditional_metric(
            x1, normalize=False, h=h
        )
        diff = xh - x1
        expected = torch.einsum("bi,bj->bij", diff, diff)
        assert torch.allclose(gh, expected, atol=1e-6)

    def test_normalize_true_divides_by_2h2(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(4, 5)
        h = torch.full((4,), 0.5)
        torch.manual_seed(42)
        _, xh_norm, gh_norm = cmm.sample_location_and_conditional_metric(
            x1, normalize=True, h=h
        )
        torch.manual_seed(42)
        _, xh_raw, gh_raw = cmm.sample_location_and_conditional_metric(
            x1, normalize=False, h=h
        )
        expected = gh_raw / (2 * h.unsqueeze(-1).unsqueeze(-1) ** 2)
        assert torch.allclose(gh_norm, expected, atol=1e-6)


class TestSampleDiff:
    def test_output_shapes(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(8, 10)
        h, xh, diff = cmm.sample_location_and_conditional_diff(x1)
        assert h.shape == (8,)
        assert xh.shape == (8, 10)
        assert diff.shape == (8, 10)

    def test_diff_equals_neg_eps_unnormalized(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(4, 5)
        h = torch.full((4,), 0.5)
        _, xh, diff = cmm.sample_location_and_conditional_diff(x1, normalize=False, h=h)
        expected = -(xh - x1)
        assert torch.allclose(diff, expected, atol=1e-6)

    def test_diff_normalized(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(4, 5)
        h = torch.full((4,), 0.5)
        torch.manual_seed(42)
        _, xh, diff_norm = cmm.sample_location_and_conditional_diff(
            x1, normalize=True, h=h
        )
        eps = xh - x1
        expected = -eps / (math.sqrt(2) * pad_h_like_x(h, x1))
        assert torch.allclose(diff_norm, expected, atol=1e-6)


class TestComputeConditionalMetric:
    def test_zero_diff_gives_zero_metric(self):
        cmm = ConditionalMetricMatching()
        x = torch.randn(4, 5)
        h = torch.ones(4)
        gh = cmm.compute_conditional_metric(x, x, h)
        assert torch.allclose(gh, torch.zeros_like(gh), atol=1e-7)

    def test_rank_one(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.zeros(1, 5)
        xh = torch.randn(1, 5)
        h = torch.ones(1)
        gh = cmm.compute_conditional_metric(x1, xh, h, normalize=False)
        eigenvalues = torch.linalg.eigvalsh(gh.squeeze(0))
        assert (eigenvalues[:-1].abs() < 1e-5).all()
        assert eigenvalues[-1].abs() > 1e-5


class TestConsistencyMetricAndDiff:
    def test_metric_equals_outer_product_of_diff(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(8, 10)
        h = torch.full((8,), 0.3)
        torch.manual_seed(0)
        _, xh_m, gh = cmm.sample_location_and_conditional_metric(
            x1, normalize=True, h=h
        )
        torch.manual_seed(0)
        _, xh_d, diff = cmm.sample_location_and_conditional_diff(
            x1, normalize=True, h=h
        )
        assert torch.allclose(xh_m, xh_d, atol=1e-6)
        reconstructed = torch.einsum("bi,bj->bij", diff, diff)
        assert torch.allclose(gh, reconstructed, atol=1e-5)


class TestDeviceDtype:
    def test_output_dtype_matches_input(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(4, 5, dtype=torch.float64)
        h, xh, diff = cmm.sample_location_and_conditional_diff(x1)
        assert xh.dtype == torch.float64
        assert diff.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(4, 5, device="cuda")
        h, xh, gh = cmm.sample_location_and_conditional_metric(x1)
        assert h.device.type == "cuda"
        assert xh.device.type == "cuda"
        assert gh.device.type == "cuda"


class TestEdgeCases:
    def test_single_sample(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(1, 10)
        h, xh, gh = cmm.sample_location_and_conditional_metric(x1)
        assert gh.shape == (1, 10, 10)

    def test_high_dimensional_input(self):
        cmm = ConditionalMetricMatching()
        x1 = torch.randn(2, 3, 8, 8)
        h, xh, diff = cmm.sample_location_and_conditional_diff(x1)
        assert xh.shape == (2, 3, 8, 8)
        assert diff.shape == (2, 3, 8, 8)

    def test_h_min_equals_h_max_uniform(self):
        cmm = ConditionalMetricMatching(h_min=0.5, h_max=0.5, sampling_method="uniform")
        x1 = torch.randn(16, 10)
        h, _, _ = cmm.sample_location_and_conditional_diff(x1)
        assert torch.allclose(h, torch.full((16,), 0.5))
