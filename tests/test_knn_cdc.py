"""Does-it-run gate for the classical kNN CDC baseline (nothing else imports it).

Note: SklearnBruteKNN.query drops the nearest neighbor (assumed to be the
query point itself), while TorchBruteKNN keeps all k nearest points - so the
two classes are not numerically interchangeable on in-dataset queries. Each
is tested independently here.
"""

import torch

from metric_matching.classical.knn_cdc import (
    SklearnBruteKNN,
    TorchBruteKNN,
    cdc_from_dists_diffs,
)


def _sphere_points(n=200, D=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g)
    return torch.nn.functional.normalize(x, dim=1)


def _check_cdc(cdc, nq, D):
    assert cdc.shape == (nq, D, D)
    assert torch.isfinite(cdc).all()
    assert torch.allclose(cdc, cdc.transpose(1, 2), atol=1e-6)
    assert (torch.linalg.eigvalsh(cdc) >= -1e-5).all()


def test_cdc_from_dists_diffs_shapes_and_psd():
    x = _sphere_points()
    diffs = x[:10, None, :] - x[None, 10:30, :].expand(10, 20, 3)
    dists = diffs.norm(dim=-1)
    cdc = cdc_from_dists_diffs(dists, diffs, h=0.5, dist_is_squared=False)
    _check_cdc(cdc, 10, 3)


def test_sklearn_knn_cdc_runs():
    x = _sphere_points()
    cdc, dists, idx = SklearnBruteKNN().fit(x).query_cdc(x[:8], k=32, h=0.5)
    _check_cdc(cdc, 8, 3)
    assert dists.shape == idx.shape == (8, 32)


def test_torch_knn_cdc_runs():
    x = _sphere_points()
    cdc, dists, idx = TorchBruteKNN().fit(x).query_cdc(x[:8], k=32, h=0.5)
    _check_cdc(cdc, 8, 3)
    assert dists.shape == idx.shape == (8, 32)


def test_query_tangents_shape():
    x = _sphere_points()
    tangents = TorchBruteKNN().fit(x).query_tangents(x[:5], h=0.5, k=32, d=2, D=3)
    assert tangents.shape == (5, 3, 2)
    assert torch.isfinite(tangents).all()
