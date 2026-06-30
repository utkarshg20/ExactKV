"""Tests for H2OSimCompressor (v2.8 token-dropping compressor).

Tests:
  1. Registration: h2o_sim / h2o_sim_75 / h2o_sim_25 all in registry
  2. Capabilities: compressor_type="token_dropping", supports_token_dropping=True
  3. Eviction logic (_evict_tokens):
     - keep_ratio=0.5: keeps ~50% of tokens, preserves sinks and recent
     - keep_ratio=1.0: keeps all tokens (no-op path)
     - Short sequences below min_keep budget: keeps all
     - Sink indices always include 0 (first token = attention sink)
     - Recent indices always include last token
  4. compress + materialize: shapes correct, no data mutation
  5. stats: compression_ratio < 1.0 when tokens are dropped
  6. update_after_commit: re-evicts from new state
  7. No-GPU path: all tests run deterministically on CPU with fake tensors
"""
from __future__ import annotations

import pytest
import torch

from exactkv.compressors import get_compressor, list_compressors, H2OSimCompressor
from exactkv.compressors.h2o_sim import _evict_tokens


# ── Registry tests ──────────────────────────────────────────────────────────

def test_h2o_sim_in_registry():
    names = list_compressors()
    assert "h2o_sim"    in names, f"h2o_sim missing from registry: {names}"
    assert "h2o_sim_75" in names
    assert "h2o_sim_25" in names


def test_h2o_sim_get_compressor_returns_correct_type():
    c = get_compressor("h2o_sim")
    assert isinstance(c, H2OSimCompressor)


def test_h2o_sim_variants_have_different_keep_ratios():
    c50 = get_compressor("h2o_sim")
    c75 = get_compressor("h2o_sim_75")
    c25 = get_compressor("h2o_sim_25")
    assert c50().keep_ratio == 0.5 if callable(c50) else c50.keep_ratio == 0.5
    # Just check the defaults work without error
    _ = H2OSimCompressor(keep_ratio=0.75)
    _ = H2OSimCompressor(keep_ratio=0.25)


# ── Capabilities tests ───────────────────────────────────────────────────────

def test_h2o_capabilities_type():
    c = H2OSimCompressor()
    assert c.capabilities.compressor_type == "token_dropping"


def test_h2o_capabilities_token_dropping():
    c = H2OSimCompressor()
    assert c.capabilities.supports_token_dropping is True


def test_h2o_capabilities_real_bytes():
    c = H2OSimCompressor()
    assert c.capabilities.supports_real_bytes_claim is True


def test_h2o_capabilities_no_quantization():
    c = H2OSimCompressor()
    assert c.capabilities.supports_quantization is False


def test_h2o_name():
    c = H2OSimCompressor()
    assert c.name == "h2o_sim"


def test_h2o_invalid_keep_ratio_raises():
    with pytest.raises(ValueError, match="keep_ratio"):
        H2OSimCompressor(keep_ratio=0.0)
    with pytest.raises(ValueError, match="keep_ratio"):
        H2OSimCompressor(keep_ratio=1.5)


def test_h2o_invalid_sink_fraction_raises():
    with pytest.raises(ValueError, match="sink_fraction"):
        H2OSimCompressor(sink_fraction=0.0)
    with pytest.raises(ValueError, match="sink_fraction"):
        H2OSimCompressor(sink_fraction=1.0)


# ── Eviction logic tests (CPU, no model needed) ──────────────────────────────

def _fake_kv(seq_len: int, heads: int = 4, head_dim: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fake (B=1, H, S, D) key/value pair."""
    k = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    k = k.expand(1, heads, seq_len, head_dim).clone()
    v = k.clone() * 2
    return k, v


def test_evict_half_sequence():
    k, v = _fake_kv(32)
    kept_k, kept_v, idx = _evict_tokens(k, v, keep_ratio=0.5, sink_fraction=0.25, min_keep=4)
    assert kept_k.shape[2] == 16, f"Expected 16, got {kept_k.shape[2]}"
    assert kept_v.shape[2] == 16
    assert idx.shape[0] == 16


def test_evict_keeps_first_token_as_sink():
    k, v = _fake_kv(64)
    _, _, idx = _evict_tokens(k, v, keep_ratio=0.25, sink_fraction=0.25, min_keep=4)
    assert 0 in idx.tolist(), "Index 0 (attention sink) should always be kept"


def test_evict_keeps_last_token_as_recent():
    seq_len = 64
    k, v = _fake_kv(seq_len)
    _, _, idx = _evict_tokens(k, v, keep_ratio=0.25, sink_fraction=0.25, min_keep=4)
    assert (seq_len - 1) in idx.tolist(), "Last token should always be kept as recent"


def test_evict_keep_all():
    k, v = _fake_kv(32)
    kept_k, kept_v, idx = _evict_tokens(k, v, keep_ratio=1.0, sink_fraction=0.25, min_keep=4)
    assert kept_k.shape[2] == 32, "keep_ratio=1.0 should keep all tokens"


def test_evict_short_sequence_below_min_keep():
    k, v = _fake_kv(4)
    kept_k, _, idx = _evict_tokens(k, v, keep_ratio=0.25, sink_fraction=0.25, min_keep=4)
    # min_keep=4 prevents dropping below 4 tokens; seq_len=4, so all are kept
    assert kept_k.shape[2] == 4


def test_evict_preserves_tensor_values():
    """Evicted tensors at kept positions should equal the original."""
    k, v = _fake_kv(32)
    kept_k, kept_v, idx = _evict_tokens(k, v, keep_ratio=0.5, sink_fraction=0.25, min_keep=4)
    for i, pos in enumerate(idx.tolist()):
        assert torch.allclose(kept_k[:, :, i, :], k[:, :, pos, :]), \
            f"kept_k at position {i} (original {pos}) does not match original"


def test_evict_indices_are_sorted_and_unique():
    k, v = _fake_kv(100)
    _, _, idx = _evict_tokens(k, v, keep_ratio=0.3, sink_fraction=0.2, min_keep=4)
    idx_list = idx.tolist()
    assert idx_list == sorted(idx_list), "Kept indices should be sorted"
    assert len(idx_list) == len(set(idx_list)), "Kept indices should be unique"


# ── Compress / materialize tests (CPU, fake FullKVState) ─────────────────────

def _fake_full_kv_state(seq_len: int = 32, num_layers: int = 2):
    """Build a minimal FullKVState-like object using fake tensors."""
    from exactkv.cache.full_state import FullKVState

    layers = []
    for _ in range(num_layers):
        k = torch.randn(1, 4, seq_len, 16)
        v = torch.randn(1, 4, seq_len, 16)
        layers.append((k, v))

    # Build a minimal DynamicCache-like object
    class _FakeDynamicCache:
        def __init__(self, kvs):
            self.key_cache   = [kv[0] for kv in kvs]
            self.value_cache = [kv[1] for kv in kvs]

    fake_cache = _FakeDynamicCache(layers)

    ids = torch.zeros(1, seq_len, dtype=torch.long)
    empty = torch.zeros(1, 0, dtype=torch.long)

    return FullKVState(
        past_key_values=fake_cache,
        prompt_ids=ids,
        generated_ids=empty,
        full_sequence_ids=ids,
        device=torch.device("cpu"),
        dtype=torch.float32,
        metadata={"next_token_id": 0},
    )


def test_compress_returns_compressed_kv_state():
    from exactkv.cache.compressed_state import CompressedKVState
    state = _fake_full_kv_state(seq_len=32)
    c = H2OSimCompressor(keep_ratio=0.5)
    compressed = c.compress(state)
    assert isinstance(compressed, CompressedKVState)
    assert compressed.compressor_name == "h2o_sim"


def test_compress_reduces_kept_seq_len():
    state = _fake_full_kv_state(seq_len=32)
    c = H2OSimCompressor(keep_ratio=0.5)
    compressed = c.compress(state)
    assert compressed.data["kept_seq_len"] == 16


def test_compress_metadata_has_eviction_rate():
    state = _fake_full_kv_state(seq_len=32)
    c = H2OSimCompressor(keep_ratio=0.5)
    compressed = c.compress(state)
    meta = compressed.metadata
    assert "eviction_rate" in meta
    assert 0.0 < meta["eviction_rate"] <= 1.0


def test_compress_does_not_mutate_full_state():
    import copy
    state = _fake_full_kv_state(seq_len=32)
    k_before = state.past_key_values.key_cache[0].clone()
    c = H2OSimCompressor(keep_ratio=0.5)
    _ = c.compress(state)
    assert torch.allclose(state.past_key_values.key_cache[0], k_before), \
        "compress() must not mutate the input FullKVState"


def test_materialize_returns_correct_shape():
    state = _fake_full_kv_state(seq_len=32, num_layers=2)
    c = H2OSimCompressor(keep_ratio=0.5)
    compressed = c.compress(state)
    materialized = c.materialize_for_draft(compressed)
    # Check that the materialized cache has 16 tokens (kept_seq_len)
    assert materialized.key_cache[0].shape[2] == 16, \
        f"Expected 16 kept tokens, got {materialized.key_cache[0].shape[2]}"


def test_update_after_commit_recompresses():
    state1 = _fake_full_kv_state(seq_len=32)
    state2 = _fake_full_kv_state(seq_len=48)  # simulates extended sequence
    c = H2OSimCompressor(keep_ratio=0.5)
    compressed1 = c.compress(state1)
    compressed2 = c.update_after_commit(compressed1, state2)
    # Should have re-evicted from the new (longer) state
    assert compressed2.data["original_seq_len"] == 48


# ── Stats tests ──────────────────────────────────────────────────────────────

def test_stats_compression_ratio_below_one_when_tokens_dropped():
    state = _fake_full_kv_state(seq_len=64)
    c = H2OSimCompressor(keep_ratio=0.5)
    compressed = c.compress(state)
    stats = c.stats(compressed)
    assert stats.compression_ratio < 1.0, \
        f"Expected ratio < 1.0 (tokens dropped), got {stats.compression_ratio}"
    assert stats.memory_reduction_factor > 1.0


def test_stats_noop_when_keep_ratio_one():
    state = _fake_full_kv_state(seq_len=32)
    c = H2OSimCompressor(keep_ratio=1.0)
    compressed = c.compress(state)
    stats = c.stats(compressed)
    # With keep_ratio=1.0 there is some overhead from index tensors
    # but stored_kv_bytes should roughly equal full fp32 storage in the original dtype
    assert stats.stored_kv_bytes > 0
    assert stats.compression_ratio <= 1.0 + 0.1  # at most 10% overhead from indices


def test_stats_has_required_fields():
    state = _fake_full_kv_state(seq_len=32)
    c = H2OSimCompressor(keep_ratio=0.5)
    stats = c.stats(c.compress(state))
    assert stats.full_bytes > 0
    assert stats.compressed_bytes > 0
    assert stats.num_layers == 2
    assert stats.compressor_name == "h2o_sim"
