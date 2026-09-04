import torch


def batched_eig_of_UtU(U: torch.Tensor):
    """
    Compute eigenvalues and eigenvectors of UᵀU for a batch of matrices.

    Args:
        U: (B, K, D) float32 tensor, where D >= K.

    Returns:
        evals: (B, K)  eigenvalues (σ_i²)
        evecs: (B, D, K) eigenvectors (columns correspond to evals)
    """
    # Batched thin SVD: U = Q Σ Vᵀ
    # shapes: U_svd (B, K, K), S (B, K), Vh (B, K, D)
    _, S, Vh = torch.linalg.svd(U, full_matrices=False)

    evals = S**2  # (B, K)
    evecs = Vh.transpose(-2, -1)  # (B, D, K)
    return evals, evecs
