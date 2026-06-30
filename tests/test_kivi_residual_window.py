"""Unit tests for KIVI residual fp16 window helpers (no upstream KIVI required)."""
from __future__ import annotations

import torch

from exactkv.compressors.kivi_adapter import (
    _merge_prefix_residual,
    _split_residual,
)


def test_split_residual_zero_disables_window():
    t = torch.randn(1, 4, 64, 32)
    prefix, suffix = _split_residual(t, 0)
    assert prefix is not None and suffix is None
    assert torch.equal(prefix, t)


def test_split_residual_short_seq_all_fp16():
    t = torch.randn(1, 4, 16, 32)
    prefix, suffix = _split_residual(t, 32)
    assert prefix is None
    assert suffix is not None
    assert torch.equal(suffix, t)


def test_split_and_merge_roundtrip():
    t = torch.randn(1, 2, 48, 64)
    residual_length = 32
    prefix, suffix = _split_residual(t, residual_length)
    assert prefix is not None and suffix is not None
    assert prefix.shape[2] == 16
    assert suffix.shape[2] == 32
    merged = _merge_prefix_residual(prefix, suffix)
    assert merged.shape == t.shape
    assert torch.allclose(merged, t)


def test_merge_prefix_only():
    t = torch.randn(1, 2, 8, 64)
    merged = _merge_prefix_residual(t, None)
    assert torch.equal(merged, t)


def test_merge_residual_only():
    t = torch.randn(1, 2, 8, 64)
    merged = _merge_prefix_residual(None, t)
    assert torch.equal(merged, t)
