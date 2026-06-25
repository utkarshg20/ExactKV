"""Triton-accelerated KV compression kernels (Phase F).

Triton-first GPU path with identical fallback to Phase E torch kernels.
"""
from __future__ import annotations

from typing import Any

import torch

from exactkv.kernel.kv_compression_kernel import (
    DEFAULT_BLOCK_SPARSE_BLOCK_SIZE,
    DEFAULT_BLOCK_SPARSE_DROP_RATE,
    DEFAULT_INT4_GROUP_SIZE,
    KERNEL_MODES,
    CompressedKVResult,
    KVCompressionKernel,
    estimate_kv_memory,
    _block_sparse_compact,
    _dequant_int4,
    _dequant_int8,
    _pack_int4_nibbles,
)

PHASE_F_ID = "phaseF_triton_kv_compression_kernel"

_TRITON_AVAILABLE: bool | None = None
_INT8_KERNEL = None
_INT4_KERNEL = None


def is_triton_available() -> bool:
    """Return True when Triton + CUDA are usable."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is not None:
        return _TRITON_AVAILABLE
    try:
        import triton  # noqa: F401, PLC0415

        _TRITON_AVAILABLE = bool(torch.cuda.is_available())
    except ImportError:
        _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def _lazy_load_triton_kernels() -> bool:
    global _INT8_KERNEL, _INT4_KERNEL
    if not is_triton_available():
        return False
    if _INT8_KERNEL is not None:
        return True
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def int8_quant_row_kernel(
            x_ptr,
            q_ptr,
            scale_ptr,
            n_rows,
            n_cols,
            stride_xr,
            stride_xc,
            stride_qr,
            stride_qc,
            BLOCK_D: tl.constexpr,
        ):
            row = tl.program_id(0)
            if row >= n_rows:
                return
            cols = tl.arange(0, BLOCK_D)
            mask = cols < n_cols
            x = tl.load(
                x_ptr + row * stride_xr + cols * stride_xc,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=0)
            scale = amax / 127.0
            scale = tl.where(scale > 1e-8, scale, 1e-8)
            q = tl.extra.cuda.libdevice.round(x / scale)
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            tl.store(scale_ptr + row, scale)
            tl.store(
                q_ptr + row * stride_qr + cols * stride_qc,
                q.to(tl.int8),
                mask=mask,
            )

        @triton.jit
        def int4_group_quant_kernel(
            x_ptr,
            q_ptr,
            scale_ptr,
            zp_ptr,
            n_rows,
            n_groups,
            group_size,
            stride_xr,
            stride_xc,
            stride_qr,
            stride_qc,
            BLOCK_G: tl.constexpr,
        ):
            row = tl.program_id(0)
            group = tl.program_id(1)
            if row >= n_rows or group >= n_groups:
                return
            gcols = tl.arange(0, BLOCK_G)
            mask = gcols < group_size
            base = group * group_size
            x = tl.load(
                x_ptr + row * stride_xr + (base + gcols) * stride_xc,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            min_val = tl.min(x, axis=0)
            max_val = tl.max(x, axis=0)
            scale = (max_val - min_val) / 15.0
            scale = tl.where(scale > 1e-8, scale, 1e-8)
            zp = tl.extra.cuda.libdevice.round(-min_val / scale)
            zp = tl.minimum(tl.maximum(zp, 0.0), 15.0)
            q = tl.extra.cuda.libdevice.round(x / scale + zp)
            q = tl.minimum(tl.maximum(q, 0.0), 15.0)
            tl.store(scale_ptr + row * n_groups + group, scale)
            tl.store(zp_ptr + row * n_groups + group, zp)
            tl.store(
                q_ptr + row * stride_qr + (base + gcols) * stride_qc,
                q.to(tl.int8),
                mask=mask,
            )

        _INT8_KERNEL = int8_quant_row_kernel
        _INT4_KERNEL = int4_group_quant_kernel
        return True
    except Exception:
        _TRITON_AVAILABLE = False
        return False


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _triton_symmetric_quantize_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused INT8 quant via Triton (CUDA only)."""
    if not _lazy_load_triton_kernels():
        from exactkv.kernel.kv_compression_kernel import _symmetric_quantize_int8

        return _symmetric_quantize_int8(t)

    orig_shape = t.shape
    head_dim = orig_shape[-1]
    flat = t.contiguous().reshape(-1, head_dim)
    n_rows, n_cols = flat.shape
    q = torch.empty(n_rows, n_cols, device=flat.device, dtype=torch.int8)
    scale = torch.empty(n_rows, device=flat.device, dtype=torch.float32)
    block_d = _next_power_of_2(n_cols)
    grid = (n_rows,)
    _INT8_KERNEL[grid](
        flat,
        q,
        scale,
        n_rows,
        n_cols,
        flat.stride(0),
        flat.stride(1),
        q.stride(0),
        q.stride(1),
        BLOCK_D=block_d,
    )
    scale_bc = scale.reshape(*orig_shape[:-1], 1)
    q_shaped = q.reshape(orig_shape)
    return q_shaped, scale_bc


def _triton_group_quantize_int4(
    t: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _lazy_load_triton_kernels():
        from exactkv.kernel.kv_compression_kernel import _group_quantize_int4

        return _group_quantize_int4(t, group_size)

    orig_shape = t.shape
    head_dim = orig_shape[-1]
    flat = t.contiguous().reshape(-1, head_dim)
    n_groups = (head_dim + group_size - 1) // group_size
    pad = n_groups * group_size - head_dim
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    n_rows = flat.shape[0]
    grouped_cols = n_groups * group_size
    work = flat.reshape(n_rows, grouped_cols)

    q = torch.empty(n_rows, grouped_cols, device=t.device, dtype=torch.int8)
    scale = torch.empty(n_rows, n_groups, device=t.device, dtype=torch.float32)
    zp = torch.empty(n_rows, n_groups, device=t.device, dtype=torch.float32)
    block_g = _next_power_of_2(group_size)
    grid = (n_rows, n_groups)
    _INT4_KERNEL[grid](
        work,
        q,
        scale,
        zp,
        n_rows,
        n_groups,
        group_size,
        work.stride(0),
        work.stride(1),
        q.stride(0),
        q.stride(1),
        BLOCK_G=block_g,
    )
    q_flat = q[..., :head_dim].reshape(orig_shape)
    return q_flat, scale, zp


class TritonKVCompressionKernel:
    """Phase F kernel — Triton on CUDA, Phase E torch fallback otherwise."""

    def __init__(
        self,
        *,
        int4_group_size: int = DEFAULT_INT4_GROUP_SIZE,
        block_size: int = DEFAULT_BLOCK_SPARSE_BLOCK_SIZE,
        block_drop_rate: float = DEFAULT_BLOCK_SPARSE_DROP_RATE,
        force_torch: bool = False,
    ) -> None:
        self.int4_group_size = int4_group_size
        self.block_size = block_size
        self.block_drop_rate = block_drop_rate
        self.force_torch = force_torch
        self._torch_kernel = KVCompressionKernel(
            int4_group_size=int4_group_size,
            block_size=block_size,
            block_drop_rate=block_drop_rate,
        )

    @property
    def backend(self) -> str:
        if self.force_torch or not is_triton_available():
            return "torch"
        return "triton"

    def compress_kv(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mode: str,
        *,
        seed: int = 0,
    ) -> CompressedKVResult:
        if mode not in KERNEL_MODES:
            msg = f"unsupported mode {mode!r}; choose from {KERNEL_MODES}"
            raise ValueError(msg)

        use_triton = (
            self.backend == "triton"
            and k_cache.is_cuda
            and mode in ("int8", "int4")
        )
        if mode == "noop" or not use_triton:
            result = self._torch_kernel.compress_kv(k_cache, v_cache, mode, seed=seed)
            result.metadata["execution_backend"] = "torch"
            return result

        memory_before = estimate_kv_memory(k_cache, v_cache)["total_bytes"]
        if mode == "int8":
            result = self._compress_int8_triton(k_cache, v_cache, memory_before)
        else:
            result = self._compress_int4_triton(k_cache, v_cache, memory_before)
        result.metadata["execution_backend"] = "triton"
        return result

    def _compress_int8_triton(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_before: int,
    ) -> CompressedKVResult:
        k_q, k_scale = _triton_symmetric_quantize_int8(k)
        v_q, v_scale = _triton_symmetric_quantize_int8(v)
        k_d = _dequant_int8(k_q, k_scale)
        v_d = _dequant_int8(v_q, v_scale)
        meta: dict[str, Any] = {
            "k_scale": k_scale,
            "v_scale": v_scale,
            "storage_dtype": "int8",
        }
        memory_after = estimate_kv_memory(k_q, v_q, metadata=meta)["total_bytes"]
        meta.update(
            KVCompressionKernel._base_metadata("int8", memory_before, memory_after, k, v),
        )
        return CompressedKVResult(
            k_compressed=k_q,
            v_compressed=v_q,
            k_dequant=k_d,
            v_dequant=v_d,
            metadata=meta,
        )

    def _compress_int4_triton(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_before: int,
    ) -> CompressedKVResult:
        k_q, k_scale, k_zp = _triton_group_quantize_int4(k, self.int4_group_size)
        v_q, v_scale, v_zp = _triton_group_quantize_int4(v, self.int4_group_size)
        k_packed = _pack_int4_nibbles(k_q)
        v_packed = _pack_int4_nibbles(v_q)
        k_d = _dequant_int4(k_q, k_scale, k_zp, self.int4_group_size, k.shape)
        v_d = _dequant_int4(v_q, v_scale, v_zp, self.int4_group_size, v.shape)
        meta: dict[str, Any] = {
            "k_scale": k_scale,
            "v_scale": v_scale,
            "k_zero_point": k_zp,
            "v_zero_point": v_zp,
            "group_size": self.int4_group_size,
            "storage_dtype": "int4_packed_uint8",
        }
        memory_after = estimate_kv_memory(k_packed, v_packed, metadata=meta)["total_bytes"]
        meta.update(
            KVCompressionKernel._base_metadata("int4", memory_before, memory_after, k, v),
        )
        return CompressedKVResult(
            k_compressed=k_packed,
            v_compressed=v_packed,
            k_dequant=k_d,
            v_dequant=v_d,
            metadata=meta,
        )


def compress_kv_triton(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mode: str,
    *,
    seed: int = 0,
    force_torch: bool = False,
) -> CompressedKVResult:
    """Convenience wrapper for TritonKVCompressionKernel."""
    kernel = TritonKVCompressionKernel(force_torch=force_torch)
    return kernel.compress_kv(k_cache, v_cache, mode, seed=seed)


def block_sparse_with_backend(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    drop_rate: float,
    seed: int,
) -> CompressedKVResult:
    """block_sparse uses Phase E compaction (mask parity); GPU tensors supported."""
    memory_before = estimate_kv_memory(k, v)["total_bytes"]
    k_c, v_c, mask, kept = _block_sparse_compact(
        k, v, block_size=block_size, drop_rate=drop_rate, seed=seed,
    )
    meta: dict[str, Any] = {
        "block_mask": mask,
        "block_size": block_size,
        "kept_blocks": kept,
        "original_seq_len": k.shape[-2],
        "compressed_seq_len": k_c.shape[-2],
        "execution_backend": "torch_compact",
    }
    memory_after = estimate_kv_memory(k_c, v_c, metadata={"block_mask": mask})["total_bytes"]
    meta.update(KVCompressionKernel._base_metadata("block_sparse", memory_before, memory_after, k, v))
    return CompressedKVResult(
        k_compressed=k_c,
        v_compressed=v_c,
        k_dequant=k_c,
        v_dequant=v_c,
        metadata=meta,
    )
