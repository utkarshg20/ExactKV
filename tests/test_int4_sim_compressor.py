"""Unit tests for Int4SimCompressor.

Gate: simulated INT4 compressor unit gate.

Verifies:
  * Registered under "int4_sim" in the compressor registry.
  * get_compressor("int4_sim") returns an Int4SimCompressor instance.
  * capabilities.is_simulated is True.
  * capabilities.supports_real_bytes_claim is False.
  * stats() returns non-negative, finite byte counts.
  * Quant/dequant output is finite (no NaN or Inf).
  * Materialized cache is forward-usable (model.forward does not crash).
  * compress() and materialize_for_draft() do not mutate the authoritative FullKVState.
  * All-zero tensor edge case is handled safely.
"""
from __future__ import annotations

import copy
import os

import pytest
import torch

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"

import exactkv.compressors  # noqa: F401 — registers built-ins


@pytest.fixture(scope="module")
def runtime():
    from exactkv.runtime.model_runtime import ModelRuntime
    return ModelRuntime(MODEL_NAME, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def full_state(runtime):
    from exactkv.runtime.prefill import prefill_to_full_state
    return prefill_to_full_state(runtime, PROMPT)


@pytest.fixture(scope="module")
def compressor():
    from exactkv.compressors.int4_sim import Int4SimCompressor
    return Int4SimCompressor()


@pytest.fixture(scope="module")
def compressed(compressor, full_state):
    return compressor.compress(full_state)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_int4_sim_is_registered():
    from exactkv.compressors import list_compressors
    assert "int4_sim" in list_compressors(), (
        f"'int4_sim' missing from registry: {list_compressors()}"
    )


def test_get_int4_sim_returns_correct_type():
    from exactkv.compressors import get_compressor
    from exactkv.compressors.int4_sim import Int4SimCompressor

    comp = get_compressor("int4_sim")
    assert isinstance(comp, Int4SimCompressor)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

def test_capabilities_is_simulated(compressor):
    assert compressor.capabilities.is_simulated is True


def test_capabilities_supports_real_bytes_claim_false(compressor):
    assert compressor.capabilities.supports_real_bytes_claim is False


def test_capabilities_supports_quantization_true(compressor):
    assert compressor.capabilities.supports_quantization is True


def test_capabilities_compressor_type(compressor):
    assert compressor.capabilities.compressor_type == "quantization"


def test_capabilities_name(compressor):
    assert compressor.capabilities.name == "int4_sim"


def test_capabilities_notes_mentions_simulated(compressor):
    notes_lower = compressor.capabilities.notes.lower()
    assert "simulated" in notes_lower or "int8" in notes_lower, (
        "capabilities.notes should mention that storage is int8, not real 4-bit packed"
    )


# ---------------------------------------------------------------------------
# Compression stats
# ---------------------------------------------------------------------------

def test_stats_full_bytes_positive(compressor, compressed):
    stats = compressor.stats(compressed)
    assert stats.full_bytes > 0


def test_stats_compressed_bytes_positive(compressor, compressed):
    stats = compressor.stats(compressed)
    assert stats.compressed_bytes > 0


def test_stats_compression_ratio_in_range(compressor, compressed):
    stats = compressor.stats(compressed)
    # Actual int8 storage vs fp32: should be < 1 (roughly 0.25 + scale overhead)
    assert 0.0 < stats.compression_ratio < 1.0, (
        f"compression_ratio should be in (0, 1) for int4_sim, got {stats.compression_ratio}"
    )


def test_stats_memory_reduction_factor_greater_than_one(compressor, compressed):
    stats = compressor.stats(compressed)
    assert stats.memory_reduction_factor > 1.0, (
        f"memory_reduction_factor should be > 1 for int4_sim, got {stats.memory_reduction_factor}"
    )


def test_stats_seq_len_positive(compressor, compressed):
    stats = compressor.stats(compressed)
    assert stats.seq_len > 0


def test_stats_num_layers_positive(compressor, compressed):
    stats = compressor.stats(compressed)
    assert stats.num_layers > 0


# ---------------------------------------------------------------------------
# Quant / dequant correctness
# ---------------------------------------------------------------------------

def test_quant_values_in_int4_range(compressor, full_state):
    """Quantised values must lie in [-8, 7]."""
    compressed = compressor.compress(full_state)
    for layer in compressed.data["layers"]:
        for key in ("k_q", "v_q"):
            q = layer[key]
            assert q.dtype == torch.int8
            assert int(q.min()) >= -8, f"{key} min {q.min()} < -8"
            assert int(q.max()) <= 7, f"{key} max {q.max()} > 7"


def test_dequant_output_finite(compressor, full_state):
    compressed = compressor.compress(full_state)
    mat = compressor.materialize_for_draft(compressed)

    from exactkv.cache.utils import extract_kv_tensors
    k_tensors, v_tensors, _ = extract_kv_tensors(mat)
    for k, v in zip(k_tensors, v_tensors):
        assert torch.isfinite(k).all(), "Dequantised K contains NaN or Inf"
        assert torch.isfinite(v).all(), "Dequantised V contains NaN or Inf"


def test_all_zero_tensor_handled_safely():
    """Scale = 1.0 should be used when max(abs(t)) == 0."""
    from exactkv.compressors.int4_sim import _quantize_int4

    t = torch.zeros(4, 8)
    q, scale = _quantize_int4(t)
    assert scale == 1.0
    assert (q == 0).all()


# ---------------------------------------------------------------------------
# Forward-usability
# ---------------------------------------------------------------------------

def test_materialized_cache_is_forward_usable(runtime, compressor, full_state):
    compressed = compressor.compress(full_state)
    mat = compressor.materialize_for_draft(compressed)

    next_tok = compressed.next_token_id
    tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=runtime.device)

    out = runtime.forward(tok_tensor, past_key_values=mat, use_cache=True)
    assert out.logits is not None
    assert out.logits.shape[-1] > 0, "Logits vocab dimension should be > 0"


# ---------------------------------------------------------------------------
# No-mutation guarantee
# ---------------------------------------------------------------------------

def test_compress_does_not_mutate_full_state(runtime, compressor):
    """compress() must not modify the authoritative FullKVState."""
    from exactkv.cache.utils import extract_kv_tensors
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, PROMPT)
    k_before, v_before, _ = extract_kv_tensors(state.past_key_values)
    k_before = [t.clone() for t in k_before]
    v_before = [t.clone() for t in v_before]

    compressor.compress(state)

    k_after, v_after, _ = extract_kv_tensors(state.past_key_values)
    for i, (kb, ka) in enumerate(zip(k_before, k_after)):
        assert torch.equal(kb, ka), f"compress() mutated K tensor at layer {i}"
    for i, (vb, va) in enumerate(zip(v_before, v_after)):
        assert torch.equal(vb, va), f"compress() mutated V tensor at layer {i}"


def test_materialize_does_not_mutate_compressed_data(compressor, full_state):
    """materialize_for_draft() must not modify compressed.data."""
    compressed = compressor.compress(full_state)

    k_before = [layer["k_q"].clone() for layer in compressed.data["layers"]]
    v_before = [layer["v_q"].clone() for layer in compressed.data["layers"]]

    compressor.materialize_for_draft(compressed)

    for i, layer in enumerate(compressed.data["layers"]):
        assert torch.equal(k_before[i], layer["k_q"]), (
            f"materialize_for_draft() mutated k_q at layer {i}"
        )
        assert torch.equal(v_before[i], layer["v_q"]), (
            f"materialize_for_draft() mutated v_q at layer {i}"
        )
