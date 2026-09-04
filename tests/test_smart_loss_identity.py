"""The core algebraic identity behind low-rank ("smart") training.

The smart loss never materializes a D x D matrix; per sample it computes

    ||U U^T||_F^2  -  2 d^T (U U^T) d  +  ||d||^4
  = ||U U^T - d d^T||_F^2

which must equal the naive Frobenius loss exactly (Appendix C of the paper,
with lambda = 0). This test pins that identity.
"""

import torch

from metric_matching.systems.mm_system import (
    _frobenius_norm_sq_gram,
    _quadform_batch,
)


def test_smart_loss_equals_naive_frobenius_loss():
    torch.manual_seed(0)
    bs, rank, D = 8, 5, 17
    emb = torch.randn(bs, rank, D, dtype=torch.float64)  # encoder output
    diff = torch.randn(bs, D, dtype=torch.float64)  # normalized target diff

    # Smart path (as in MMSystem._step, tikhonov = 0)
    U = emb.transpose(1, 2)  # [bs, D, rank]
    quad = _quadform_batch(U, diff)
    frob = _frobenius_norm_sq_gram(U)
    const = (diff.pow(2).sum(-1)).pow(2)
    smart = ((frob - 2.0 * quad + const) / (D * D)).mean()

    # Naive path: materialize G = U U^T and the rank-1 target d d^T
    G = U @ U.transpose(1, 2)  # [bs, D, D]
    target = torch.einsum("bi,bj->bij", diff, diff)
    naive = ((G - target) ** 2).sum(dim=(1, 2)).div(D * D).mean()

    assert torch.allclose(smart, naive, rtol=1e-10, atol=1e-10)


def test_quadform_and_gram_helpers_match_dense_computation():
    torch.manual_seed(1)
    bs, rank, D = 4, 3, 11
    U = torch.randn(bs, D, rank, dtype=torch.float64)
    a = torch.randn(bs, D, dtype=torch.float64)

    G = U @ U.transpose(1, 2)
    assert torch.allclose(
        _quadform_batch(U, a), torch.einsum("bi,bij,bj->b", a, G, a)
    )
    assert torch.allclose(_frobenius_norm_sq_gram(U), (G**2).sum(dim=(1, 2)))
