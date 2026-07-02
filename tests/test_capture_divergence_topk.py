"""Unit tests for capture_divergence_topk compressor wiring."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from exactkv.benchmarks.runner import _materialize_lossy_kv_for_topk
from exactkv.cache.full_state import FullKVState


class _RecordingCompressor:
    name = "noop"

    def __init__(self) -> None:
        self.compress_calls: list[FullKVState] = []

    def compress(self, state: FullKVState):
        self.compress_calls.append(state)
        return object()

    def materialize_for_draft(self, compressed):
        return {"materialized": True}


def test_materialize_lossy_kv_passes_full_state_to_compress() -> None:
    compressor = _RecordingCompressor()
    full_state = MagicMock(spec=FullKVState)

    with patch("copy.deepcopy", side_effect=lambda x: x) as deepcopy:
        result = _materialize_lossy_kv_for_topk(compressor, full_state)

    assert len(compressor.compress_calls) == 1
    assert compressor.compress_calls[0] is full_state
    deepcopy.assert_called_once_with({"materialized": True})
    assert result == {"materialized": True}


def test_materialize_lossy_kv_rejects_incomplete_compressor() -> None:
    class _BadCompressor:
        def compress(self, past_kv):
            return past_kv

    with pytest.raises(TypeError, match="materialize_for_draft"):
        _materialize_lossy_kv_for_topk(_BadCompressor(), MagicMock(spec=FullKVState))
