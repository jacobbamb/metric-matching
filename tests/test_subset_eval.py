import torch
from torch.utils.data import TensorDataset

from metric_matching.callbacks.subset_eval import _tensorize_subset


def test_tensorize_subset_keeps_each_image_as_one_row():
    xs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    ys = torch.tensor([0, 1])
    subset = TensorDataset(xs, ys)

    tensorized = _tensorize_subset(subset)

    assert tensorized.shape == (2, 12)
    assert torch.equal(tensorized[0], xs[0].reshape(-1))
    assert torch.equal(tensorized[1], xs[1].reshape(-1))
