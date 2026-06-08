"""Int8Compressor unit tests — require the real model (Qwen/Qwen2.5-0.5B, fp32).

Tests
-----
1. stats.compression_ratio > 0
2. stats.full_bytes > 0
3. stats.compressed_bytes > 0
4. Materialized cache is forward-usable
5. Compress + materialize do not mutate authoritative full_state
6. Quant/dequant error is finite (no NaN or Inf)
"""
from __future__ import annotations

import copy

import pytest
import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import extract_kv_tensors, kv_seq_len
from exactkv.compressors.int8 import Int8Compressor
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"
PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def runtime() -> ModelRuntime:
    return ModelRuntime(model_name=MODEL_NAME, device="auto", dtype=DTYPE)


@pytest.fixture(scope="module")
def full_state(runtime: ModelRuntime) -> FullKVState:
    """FullKVState after prefill on PROMPT."""
    prompt_ids = runtime.encode(PROMPT)
    with torch.no_grad():
        out = runtime.forward(prompt_ids, past_key_values=None, use_cache=True)
    next_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
    empty_gen = torch.zeros(1, 0, dtype=torch.long, device=runtime.device)
    return FullKVState(
        past_key_values=out.past_key_values,
        prompt_ids=prompt_ids,
        generated_ids=empty_gen,
        full_sequence_ids=prompt_ids,
        device=runtime.device,
        dtype=runtime.dtype,
        metadata={"next_token_id": next_tok},
    )


@pytest.fixture(scope="module")
def compressor() -> Int8Compressor:
    return Int8Compressor()


# ---------------------------------------------------------------------------


def test_stats_compression_ratio_positive(
    compressor: Int8Compressor, full_state: FullKVState
) -> None:
    compressed = compressor.compress(full_state)
    stats = compressor.stats(compressed)
    assert stats.compression_ratio > 0.0, f"ratio={stats.compression_ratio}"


def test_stats_full_bytes_positive(
    compressor: Int8Compressor, full_state: FullKVState
) -> None:
    compressed = compressor.compress(full_state)
    stats = compressor.stats(compressed)
    assert stats.full_bytes > 0, f"full_bytes={stats.full_bytes}"


def test_stats_compressed_bytes_positive(
    compressor: Int8Compressor, full_state: FullKVState
) -> None:
    compressed = compressor.compress(full_state)
    stats = compressor.stats(compressed)
    assert stats.compressed_bytes > 0, f"compressed_bytes={stats.compressed_bytes}"


def test_stats_compressed_smaller_than_full(
    compressor: Int8Compressor, full_state: FullKVState
) -> None:
    """INT8 should yield fewer bytes than fp32."""
    compressed = compressor.compress(full_state)
    stats = compressor.stats(compressed)
    assert stats.compressed_bytes < stats.full_bytes, (
        f"Expected compressed_bytes < full_bytes, got "
        f"{stats.compressed_bytes} vs {stats.full_bytes}"
    )


def test_materialized_cache_is_forward_usable(
    runtime: ModelRuntime, compressor: Int8Compressor, full_state: FullKVState
) -> None:
    """Model.forward must not raise when given the materialised cache."""
    compressed = compressor.compress(full_state)
    mat = compressor.materialize_for_draft(compressed)

    next_tok = compressed.next_token_id
    tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=runtime.device)

    with torch.no_grad():
        out = runtime.forward(tok_tensor, past_key_values=copy.deepcopy(mat))

    assert out.logits is not None
    assert out.logits.shape[-1] > 0
    assert torch.isfinite(out.logits).all(), "logits contain non-finite values"


def test_compress_does_not_mutate_full_state(
    compressor: Int8Compressor, full_state: FullKVState
) -> None:
    kv_len_before = kv_seq_len(full_state.past_key_values)
    next_tok_before = full_state.next_token_id

    _ = compressor.compress(full_state)

    assert kv_seq_len(full_state.past_key_values) == kv_len_before
    assert full_state.next_token_id == next_tok_before


def test_materialize_does_not_mutate_full_state(
    runtime: ModelRuntime, compressor: Int8Compressor, full_state: FullKVState
) -> None:
    kv_len_before = kv_seq_len(full_state.past_key_values)

    compressed = compressor.compress(full_state)
    mat = compressor.materialize_for_draft(compressed)

    # Run a forward pass through the materialized cache (deep-copy to avoid
    # mat itself being mutated by the forward).
    tok = torch.tensor([[compressed.next_token_id]], dtype=torch.long, device=runtime.device)
    with torch.no_grad():
        runtime.forward(tok, past_key_values=copy.deepcopy(mat))

    assert kv_seq_len(full_state.past_key_values) == kv_len_before, (
        "full_state.past_key_values was mutated"
    )


def test_quant_dequant_error_is_finite(
    compressor: Int8Compressor, full_state: FullKVState
) -> None:
    """Dequantised tensors must be finite (no NaN or Inf)."""
    compressed = compressor.compress(full_state)
    mat = compressor.materialize_for_draft(compressed)

    k_tensors, v_tensors, _ = extract_kv_tensors(mat)
    for i, (k, v) in enumerate(zip(k_tensors, v_tensors)):
        assert torch.isfinite(k).all(), f"Layer {i} key has non-finite values"
        assert torch.isfinite(v).all(), f"Layer {i} value has non-finite values"
