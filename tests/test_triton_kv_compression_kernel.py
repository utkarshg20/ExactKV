"""Tests for Phase F Triton KV compression kernel."""
from __future__ import annotations

import pytest
import torch

from exactkv.kernel.kv_compression_kernel import KVCompressionKernel
from exactkv.kernel.triton_kv_compression_kernel import (
    TritonKVCompressionKernel,
    compress_kv_triton,
    is_triton_available,
)
from exactkv.runtime.kv_kernel_backend_selector import (
    backend_info,
    compress_kv,
    set_force_torch_backend,
)


def _sample_kv(seed: int = 0, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k = torch.randn(1, 4, 32, 64, generator=gen, device=device)
    v = torch.randn(1, 4, 32, 64, generator=gen, device=device)
    return k, v


def test_backend_info() -> None:
    info = backend_info()
    assert "active_backend" in info
    assert "modes" in info


def test_torch_fallback_parity_int8() -> None:
    k, v = _sample_kv()
    torch_k = KVCompressionKernel()
    triton_k = TritonKVCompressionKernel(force_torch=True)
    r_torch = torch_k.compress_kv(k, v, "int8")
    r_triton = triton_k.compress_kv(k, v, "int8")
    assert r_torch.k_compressed.shape == r_triton.k_compressed.shape
    assert r_torch.metadata["compression_ratio"] == pytest.approx(
        r_triton.metadata["compression_ratio"],
    )
    assert torch.equal(r_torch.k_compressed, r_triton.k_compressed)


def test_torch_fallback_parity_int4() -> None:
    k, v = _sample_kv(seed=1)
    torch_k = KVCompressionKernel()
    triton_k = TritonKVCompressionKernel(force_torch=True)
    r_torch = torch_k.compress_kv(k, v, "int4")
    r_triton = triton_k.compress_kv(k, v, "int4")
    assert r_torch.k_compressed.shape == r_triton.k_compressed.shape
    assert r_torch.metadata["memory_after"] == r_triton.metadata["memory_after"]


def test_block_sparse_shape_and_determinism() -> None:
    k, v = _sample_kv(seed=3)
    r1 = compress_kv(k, v, "block_sparse", seed=7)
    r2 = compress_kv(k, v, "block_sparse", seed=7)
    assert r1.k_compressed.shape == r2.k_compressed.shape
    assert torch.equal(r1.k_compressed, r2.k_compressed)
    assert r1.k_compressed.shape[-2] <= k.shape[-2]


def test_selector_force_torch() -> None:
    set_force_torch_backend(True)
    k, v = _sample_kv()
    result = compress_kv(k, v, "int8")
    assert result.metadata.get("execution_backend") == "torch"
    set_force_torch_backend(False)


def test_compress_kv_triton_api() -> None:
    k, v = _sample_kv()
    result = compress_kv_triton(k, v, "noop", force_torch=True)
    assert torch.allclose(result.k_dequant, k)


@pytest.mark.skipif(not is_triton_available(), reason="CUDA+Triton required")
def test_triton_cuda_parity_int8() -> None:
    k, v = _sample_kv(device="cuda")
    k = k.cuda()
    v = v.cuda()
    torch_k = KVCompressionKernel()
    triton_k = TritonKVCompressionKernel(force_torch=False)
    r_torch = torch_k.compress_kv(k, v, "int8")
    r_triton = triton_k.compress_kv(k, v, "int8")
    assert r_torch.k_compressed.shape == r_triton.k_compressed.shape
    assert r_torch.metadata["compression_ratio"] == pytest.approx(
        r_triton.metadata["compression_ratio"],
        rel=0.05,
    )
    assert torch.allclose(r_torch.k_dequant, r_triton.k_dequant, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not is_triton_available(), reason="CUDA+Triton required")
def test_triton_faster_than_torch_int8() -> None:
    import time

    k, v = _sample_kv(device="cuda")
    k, v = k.cuda(), v.cuda()
    torch_k = TritonKVCompressionKernel(force_torch=True)
    triton_k = TritonKVCompressionKernel(force_torch=False)
    for _ in range(5):
        torch_k.compress_kv(k, v, "int8")
        triton_k.compress_kv(k, v, "int8")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        torch_k.compress_kv(k, v, "int8")
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000 / 30
    t0 = time.perf_counter()
    for _ in range(30):
        triton_k.compress_kv(k, v, "int8")
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - t0) * 1000 / 30
    assert triton_ms < torch_ms
