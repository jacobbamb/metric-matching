import torch
from sklearn.neighbors import NearestNeighbors
from torch.linalg import eigh


def cdc_from_dists_diffs(
    dists: torch.Tensor,
    diffs: torch.Tensor,
    h: float,
    dist_is_squared: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Core CDC helper, called once dists and diffs are known.

    dists: [Nq, k]          (||x-xj|| or ||x-xj||^2)
    diffs: [Nq, k, D]       (x - xj)
    Returns:
      cdc: [Nq, D, D]
    """
    dist2 = dists if dist_is_squared else dists.pow(2)

    logits = -(dist2 / (2.0 * h * h)).float()  # fp32 for stability
    w = torch.softmax(logits, dim=1).to(diffs.dtype)  # rows sum to 1 exactly (up to fp)

    return torch.einsum("nk,nkd,nke->nde", w, diffs, diffs) / (h * h)


class SklearnBruteKNN:
    """CPU sklearn brute-force KNN. Returns L2 (not squared) distances."""

    dist_is_squared: bool = False

    def __init__(self, n_jobs: int = -1, metric: str = "euclidean"):
        self.n_jobs = n_jobs
        self.metric = metric
        self.nn = None
        self.x_data = None

    def fit(self, x_data: torch.Tensor):
        self.x_data = x_data
        xd = x_data.detach().cpu().float().numpy()
        self.nn = NearestNeighbors(
            algorithm="brute", metric=self.metric, n_jobs=self.n_jobs
        )
        self.nn.fit(xd)
        return self

    def query(self, x_query: torch.Tensor, k: int):
        if self.nn is None:
            raise RuntimeError("Call fit(x_data) before query(x_query, k).")

        xq = x_query.detach().cpu().float().numpy()
        dists_np, idx_np = self.nn.kneighbors(
            xq, n_neighbors=k + 1, return_distance=True
        )
        # remove self point
        dists = torch.from_numpy(dists_np).to(x_query.device, dtype=x_query.dtype)[
            :, 1:
        ]
        idx = torch.from_numpy(idx_np).to(x_query.device, dtype=torch.long)[:, 1:]
        return dists, idx

    @torch.no_grad()
    def query_cdc(self, x_query: torch.Tensor, k: int, h: float, eps: float = 1e-12):
        """
        KNN query + CDC using shared helper.
        Returns:
          cdc:   [Nq, D, D]
          dists: [Nq, k]
          idx:   [Nq, k]
        """
        if self.x_data is None:
            raise RuntimeError("Call fit(x_data) before query_cdc(...).")

        dists, idx = self.query(x_query, k)
        neighbors = self.x_data[idx]  # [Nq, k, D]
        diffs = x_query[:, None, :] - neighbors  # [Nq, k, D]

        cdc = cdc_from_dists_diffs(
            dists,
            diffs,
            h,
            dist_is_squared=self.dist_is_squared,
            eps=eps,
        )
        return cdc, dists, idx


class TorchBruteKNN:
    """Torch brute-force KNN using torch.cdist + topk. Returns L2 (not squared) distances."""

    dist_is_squared: bool = False

    def __init__(self):
        self.x_data = None

    def fit(self, x_data: torch.Tensor):
        self.x_data = x_data
        return self

    @torch.no_grad()
    def query(self, x_query: torch.Tensor, k: int):
        if self.x_data is None:
            raise RuntimeError("Call fit(x_data) before query(x_query, k).")

        dists = torch.cdist(x_query, self.x_data)  # [Nq, Nref], L2
        idx = dists.topk(k, dim=1, largest=False).indices  # [Nq, k]
        dists = dists.gather(1, idx)  # [Nq, k]
        return dists, idx

    @torch.no_grad()
    def query_cdc(self, x_query: torch.Tensor, k: int, h: float, eps: float = 1e-12):
        """
        KNN query + CDC using shared helper.
        Returns:
          cdc:   [Nq, D, D]
          dists: [Nq, k]
          idx:   [Nq, k]
        """
        if self.x_data is None:
            raise RuntimeError("Call fit(x_data) before query_cdc(...).")

        dists, idx = self.query(x_query, k)
        neighbors = self.x_data[idx]  # [Nq, k, D]
        diffs = x_query[:, None, :] - neighbors  # [Nq, k, D]

        cdc = cdc_from_dists_diffs(
            dists,
            diffs,
            h,
            dist_is_squared=self.dist_is_squared,
            eps=eps,
        )
        return cdc, dists, idx

    def query_tangents(
        self, x_query: torch.Tensor, h: float, k: int, d: int, D: int
    ) -> torch.Tensor:
        """
        KNN query + tangent space basis computation.
        Returns:
          tangents: [B, D, d]   (columns are tangent directions in ambient R^D)
        """
        if self.x_data is None:
            raise RuntimeError("Call fit(x_data) before query_tangents(...).")

        cdc, _, _ = self.query_cdc(x_query, k, h=h, eps=1e-12)
        cdc_sym = 0.5 * (cdc + cdc.transpose(-1, -2))

        eigvals, eigvecs = eigh(cdc_sym)  # (B, D), (B, D, D) # in ascending order
        top_eigenvectors = eigvecs[..., -d:]  # (B, D, d)
        return top_eigenvectors  # [B, D, d]


# Usage:
# knn = SklearnBruteKNN(n_jobs=-1).fit(x_data)
# cdc, dists, idx = knn.query_cdc(x_inf, k=64, h=1.0)
