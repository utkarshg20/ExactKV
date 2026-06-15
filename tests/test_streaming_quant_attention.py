"""Unit tests for streaming quantized-KV attention reference (Phase 16A)."""
from __future__ import annotations

import torch

from exactkv.attention.streaming_quant_attention import (
    attention_full,
    attention_materialized_compressed,
    attention_streaming_compressed,
    dequantize_kv_materialized,
    estimate_attention_memory_bytes,
    quantize_kv_int8_reference,
    run_attention_feasibility_cell,
)


def _tensors(
    *,
    b: int = 1,
    h: int = 2,
    q: int = 1,
    t: int = 64,
    d: int = 32,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    k = torch.randn(b, h, t, d, generator=gen, dtype=dtype)
    v = torch.randn(b, h, t, d, generator=gen, dtype=dtype)
    q_t = torch.randn(b, h, q, d, generator=gen, dtype=dtype)
    return q_t, k, v


def test_quantize_dequantize_shapes() -> None:
    _, k, v = _tensors()
    qkv = quantize_kv_int8_reference(k, v)
    assert qkv.k_q.shape == k.shape
    assert qkv.v_q.shape == v.shape
    assert qkv.k_q.dtype == torch.int8
    dk, dv = dequantize_kv_materialized(qkv)
    assert dk.shape == k.shape
    assert dv.shape == v.shape


def test_materialized_compressed_output_shape() -> None:
    q, k, v = _tensors(q=4)
    qkv = quantize_kv_int8_reference(k, v)
    out = attention_materialized_compressed(q, qkv)
    assert out.shape == (1, 2, 4, 32)


def test_streaming_compressed_output_shape() -> None:
    q, k, v = _tensors(q=4, t=128)
    qkv = quantize_kv_int8_reference(k, v)
    out = attention_streaming_compressed(q, qkv, chunk_size=16)
    assert out.shape == (1, 2, 4, 32)


def test_streaming_matches_materialized_fp32() -> None:
    q, k, v = _tensors(t=128, q=4)
    qkv = quantize_kv_int8_reference(k, v)
    mat = attention_materialized_compressed(q, qkv)
    for chunk_size in (8, 16, 32, 64):
        stream = attention_streaming_compressed(q, qkv, chunk_size)
        max_err = (stream - mat).abs().max().item()
        assert max_err < 5e-4, f"chunk={chunk_size} err={max_err}"


def test_streaming_matches_materialized_fp16() -> None:
    q, k, v = _tensors(t=96, q=2, dtype=torch.float16)
    qkv = quantize_kv_int8_reference(k, v)
    mat = attention_materialized_compressed(q, qkv)
    stream = attention_streaming_compressed(q, qkv, chunk_size=24)
    max_err = (stream - mat).abs().max().item()
    assert max_err < 2e-2


def test_single_token_query() -> None:
    q, k, v = _tensors(q=1, t=64)
    result = run_attention_feasibility_cell(q=q, k=k, v=v, chunk_size=16)
    assert result.full_output.shape[-2] == 1
    assert result.passed


def test_multi_token_query() -> None:
    q, k, v = _tensors(q=4, t=128)
    result = run_attention_feasibility_cell(q=q, k=k, v=v, chunk_size=32)
    assert result.full_output.shape[-2] == 4
    assert result.passed


def test_causal_last_query_positions() -> None:
    q, k, v = _tensors(q=4, t=64)
    qkv = quantize_kv_int8_reference(k, v)
    mat = attention_materialized_compressed(q, qkv, causal=True)
    stream = attention_streaming_compressed(q, qkv, chunk_size=17, causal=True)
    assert (stream - mat).abs().max().item() < 5e-4


def test_memory_accounting_fields_non_negative() -> None:
    mem = estimate_attention_memory_bytes(
        batch=1,
        heads=2,
        seq_len=128,
        head_dim=32,
        element_size_fp=4,
        chunk_size=16,
    )
    assert mem.full_kv_bytes > 0
    assert mem.stored_quantized_kv_bytes > 0
    assert mem.materialized_working_kv_bytes > 0
    assert mem.streaming_peak_chunk_working_kv_bytes > 0
    assert mem.metadata_bytes >= 0
    assert mem.num_chunks == 8
    assert 0.0 <= mem.theoretical_streaming_working_reduction_vs_materialized < 1.0


def test_streaming_peak_lower_than_materialized_when_chunk_lt_t() -> None:
    mem = estimate_attention_memory_bytes(
        batch=1,
        heads=4,
        seq_len=512,
        head_dim=64,
        element_size_fp=4,
        chunk_size=32,
    )
    assert mem.streaming_peak_chunk_working_kv_bytes < mem.materialized_working_kv_bytes
    assert mem.theoretical_streaming_working_reduction_vs_materialized > 0.0


def test_full_attention_reference_shape() -> None:
    q, k, v = _tensors(q=2, t=32)
    out = attention_full(q, k, v)
    assert out.shape == q.shape
