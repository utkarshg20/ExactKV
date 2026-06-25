"""Tests for Phase E KV compression kernel."""
from __future__ import annotations

import torch

from exactkv.kernel.kv_compression_kernel import (
    KVCompressionKernel,
    compress_from_phase_d_output,
    estimate_kv_memory,
)
from exactkv.runtime.kv_probe_to_kernel_bridge import (
    compress_kv_via_kernel,
    resolve_kernel_mode,
)


def _sample_kv(
    *,
    batch: int = 1,
    heads: int = 8,
    seq_len: int = 128,
    head_dim: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, heads, seq_len, head_dim, generator=gen)
    v = torch.randn(batch, heads, seq_len, head_dim, generator=gen)
    return k, v


def test_noop_identity() -> None:
    k, v = _sample_kv()
    kernel = KVCompressionKernel()
    result = kernel.compress_kv(k, v, "noop")
    assert torch.allclose(result.k_dequant, k)
    assert torch.allclose(result.v_dequant, v)
    assert result.metadata["compression_ratio"] == 1.0


def test_int8_reduces_storage_memory() -> None:
    k, v = _sample_kv()
    kernel = KVCompressionKernel()
    result = kernel.compress_kv(k, v, "int8")
    assert result.metadata["memory_after"] < result.metadata["memory_before"]
    assert result.k_compressed.dtype == torch.int8
    assert result.k_dequant.shape == k.shape


def test_int4_reduces_more_than_int8_storage() -> None:
    k, v = _sample_kv()
    kernel = KVCompressionKernel()
    int8_res = kernel.compress_kv(k, v, "int8")
    int4_res = kernel.compress_kv(k, v, "int4")
    assert int4_res.metadata["memory_after"] < int8_res.metadata["memory_after"]


def test_block_sparse_reduces_seq_len() -> None:
    k, v = _sample_kv(seq_len=64)
    kernel = KVCompressionKernel(block_size=8, block_drop_rate=0.5)
    result = kernel.compress_kv(k, v, "block_sparse", seed=42)
    assert result.k_compressed.shape[-2] < k.shape[-2]
    assert result.metadata["compressed_seq_len"] < result.metadata["original_seq_len"]


def test_deterministic_under_seed() -> None:
    k, v = _sample_kv()
    kernel = KVCompressionKernel()
    r1 = kernel.compress_kv(k, v, "block_sparse", seed=7)
    r2 = kernel.compress_kv(k, v, "block_sparse", seed=7)
    assert torch.equal(r1.k_compressed, r2.k_compressed)
    assert torch.equal(r1.v_compressed, r2.v_compressed)


def test_compress_from_phase_d_output() -> None:
    k, v = _sample_kv()
    result = compress_from_phase_d_output({"k_cache": k, "v_cache": v}, "int8")
    assert result.metadata["compression_mode"] == "int8"
    assert result.k_dequant.shape == k.shape


def test_bridge_maps_phase_d_modes() -> None:
    assert resolve_kernel_mode("int8_sim") == "int8"
    assert resolve_kernel_mode("kv_dropout_sim") == "block_sparse"
    k_list = [_sample_kv()[0]]
    v_list = [_sample_kv()[1]]
    k_out, v_out = compress_kv_via_kernel(k_list, v_list, "int4_sim", seed=0)
    assert k_out[0].shape == k_list[0].shape
    assert v_out[0].shape == v_list[0].shape


def test_estimate_kv_memory_counts_metadata() -> None:
    k, v = _sample_kv()
    scale = torch.tensor([1.0])
    mem = estimate_kv_memory(k, v, metadata={"scale": scale})
    assert mem["total_bytes"] == mem["k_bytes"] + mem["v_bytes"] + mem["metadata_bytes"]
