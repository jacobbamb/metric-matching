import torch
from torch import Tensor


def tangent_space_projection(x: Tensor, d: int, D: int) -> Tensor:
    """
    Batch tangent-space projection on the unit sphere S^d embedded in R^D
    (using the convention that S^d lives in the first m=d+1 coordinates).

    Args:
            x: (..., D) batch of unit-norm points with support only in first m=d+1 coords.
            d: intrinsic sphere dimension (so tangent space has rank d).
            D: extrinsic/ambient dimension.

    Returns:
            P: (..., D, D) projection matrices onto T_x S^d in the ambient R^D.
    """
    m = d + 1
    if m > D:
        raise ValueError(f"Need d+1 <= D, got d={d}, D={D}.")
    if x.shape[-1] != D:
        raise ValueError(f"Expected x.shape[-1] == D ({D}), got {x.shape[-1]}.")

    if not torch.allclose(x[:, m:], torch.zeros_like(x[:, m:])):
        x[:, m:] = 0.0
        # project back to the sphere:
        x_proj = x / x.norm(dim=1, keepdim=True)
        x = x_proj

    # make sure that the last D-(d+1) coordinates are zero (supports arbitrary batch dims)
    assert torch.allclose(x[..., m:], torch.zeros_like(x[..., m:]))

    # P(x) = M - x x^T, with M = diag([1]*m + [0]*(D-m))
    xxT = x.unsqueeze(-1) * x.unsqueeze(-2)  # (..., D, D)

    diag = torch.zeros(D, dtype=x.dtype, device=x.device)
    diag[:m] = 1
    M = torch.diag(diag)  # (D, D), broadcasts over batch

    return M - xxT


def tangent_space_basis(x: Tensor, d: int, D: int) -> Tensor:
    """
    Returns an orthonormal basis for the tangent space at x on S^d embedded in R^D.

    Args:
        x: (B, D) batch of unit-norm points with support only in first m=d+1 coords.
        d: intrinsic sphere dimension (tangent space has dimension d).
        D: extrinsic/ambient dimension.

    Returns:
        basis: (B, D, d) with basis[b, :, i] the i-th tangent direction at x[b].
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape (B, D), got {tuple(x.shape)}.")
    if x.shape[1] != D:
        raise ValueError(f"Expected x.shape[1] == D ({D}), got {x.shape[1]}.")

    m = d + 1
    if m > D:
        raise ValueError(f"Need d+1 <= D, got d={d}, D={D}.")

    if not torch.allclose(x[:, m:], torch.zeros_like(x[:, m:])):
        x[:, m:] = 0.0
        # project back to the sphere:
        x_proj = x / x.norm(dim=1, keepdim=True)
        x = x_proj

    assert torch.allclose(x[:, m:], torch.zeros_like(x[:, m:]))

    xa = x[:, :m]  # (B, m)

    # Householder with alpha = -sign(x_last) * ||xa|| (robust; reduces cancellation)
    x_last = xa[:, -1]
    nrm = torch.linalg.norm(xa, dim=1)  # (B,)
    sgn = torch.where(
        x_last >= 0, xa.new_ones(x_last.shape), -xa.new_ones(x_last.shape)
    )
    alpha = -sgn * nrm  # (B,)

    e = xa.new_zeros(m)
    e[-1] = 1.0

    u = xa - alpha[:, None] * e[None, :]  # (B, m)
    u_norm = torch.linalg.norm(u, dim=1, keepdim=True)
    u_norm = u_norm.clamp_min(torch.finfo(xa.dtype).eps)
    v = u / u_norm

    I = torch.eye(m, dtype=xa.dtype, device=xa.device)[None, :, :]
    H = I - 2.0 * (v[:, :, None] * v[:, None, :])  # (B, m, m)

    basis_active = H[:, :, :d].transpose(1, 2).contiguous()  # (B, d, m)

    basis = x.new_zeros((x.shape[0], d, D))
    basis[:, :, :m] = basis_active  # (B, d, D)
    return basis.transpose(1, 2).contiguous()  # (B, D, d)


def mean_frobenius_basis_distance(A: Tensor, B: Tensor) -> Tensor:
    """
    Mean over batch Frobenius norm between *projectors* onto subspaces.

    Accepted:
      - Bases:      A,B in (B, D, d)  -> compares (A A^T) vs (B B^T)
      - Projectors: A,B in (B, D, D)  -> compares A vs B directly

    Returns:
        scalar tensor: mean_b ||P_A[b] - P_B[b]||_F
    """
    if A.shape != B.shape or A.ndim != 3:
        raise ValueError(
            f"Expected A,B to have same shape with ndim=3; got A={tuple(A.shape)}, B={tuple(B.shape)}"
        )

    # If already projectors (square last two dims)
    if A.shape[-2] == A.shape[-1]:
        PA, PB = A, B
    else:
        # Otherwise treat as bases (B, D, d)
        PA = A @ A.transpose(-1, -2)  # (B, D, D)
        PB = B @ B.transpose(-1, -2)  # (B, D, D)

    return torch.linalg.norm(PA - PB, ord="fro", dim=(-2, -1)).mean()
