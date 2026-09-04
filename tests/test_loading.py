"""Checkpoint discovery: exact-epoch matching, legacy names, last.ckpt."""

import os
import time

import pytest

from metric_matching.utils.loading import select_checkpoint


def _touch(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_exact_epoch_matching_new_names(tmp_path):
    run = tmp_path / "run1" / "checkpoints"
    _touch(run / "epoch=0099.ckpt")
    _touch(run / "epoch=0999.ckpt")
    _touch(run / "last.ckpt")

    assert select_checkpoint(99, str(tmp_path)).name == "epoch=0099.ckpt"
    assert select_checkpoint(999, str(tmp_path)).name == "epoch=0999.ckpt"


def test_epoch_99_does_not_prefix_match_999(tmp_path):
    _touch(tmp_path / "run1" / "checkpoints" / "epoch=0999.ckpt")
    with pytest.raises(FileNotFoundError, match="available epochs"):
        select_checkpoint(99, str(tmp_path))


def test_legacy_epochepoch_names_resolve(tmp_path):
    run = tmp_path / "outputs" / "checkpoints"
    _touch(run / "epochepoch=299.ckpt")
    _touch(run / "epochepoch=1499.ckpt")

    assert select_checkpoint(299, str(tmp_path)).name == "epochepoch=299.ckpt"
    assert select_checkpoint(1499, str(tmp_path)).name == "epochepoch=1499.ckpt"


def test_epoch_none_prefers_newest_last_ckpt(tmp_path):
    now = time.time()
    _touch(tmp_path / "old_run" / "checkpoints" / "last.ckpt", mtime=now - 100)
    _touch(tmp_path / "old_run" / "checkpoints" / "epoch=0005.ckpt", mtime=now - 50)
    newest = _touch(tmp_path / "new_run" / "checkpoints" / "last.ckpt", mtime=now)

    assert select_checkpoint(None, str(tmp_path)) == newest


def test_epoch_none_falls_back_to_newest_any_ckpt(tmp_path):
    now = time.time()
    _touch(tmp_path / "run" / "checkpoints" / "epoch=0001.ckpt", mtime=now - 10)
    newest = _touch(tmp_path / "run" / "checkpoints" / "epoch=0002.ckpt", mtime=now)

    assert select_checkpoint(None, str(tmp_path)) == newest


def test_missing_root_raises_guiding_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        select_checkpoint(None, str(tmp_path / "does_not_exist"))


def test_empty_root_raises_guiding_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        select_checkpoint(None, str(tmp_path))
