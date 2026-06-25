"""Minimal KV compression kernel (Phase E — tensor-level execution prototype).

Real PyTorch tensor transforms with computable storage footprint.
No model weight changes, no serving integration, no training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

PHASE_E_ID = "phaseE_kv_compression_kernel"
KERNEL_MODES: tuple[str, ...] = ("noop", "int8", "int4", "block_sparse")
DEFAULT_INT4_GROUP_SIZE = 16
DEFAULT_BLOCK_SPARSE_BLOCK_SIZE = 8
DEFAULT_BLOCK_SPARSE_DROP_RATE = 0.25


@dataclass
class CompressedKVResult:
    """Result of kernel compression on K/V cache tensors."""

    k_compressed: torch.Tensor
    v_compressed: torch.Tensor
    k_dequant: torch.Tensor
    v_dequant: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        meta = dict(self.metadata)
        for key in ("k_scale", "v_scale", "k_zero_point", "v_zero_point", "block_mask"):
            if key in meta and isinstance(meta[key], torch.Tensor):
                meta[key] = meta[key].tolist()
        return {
            "metadata": meta,
            "k_shape": list(self.k_compressed.shape),
            "v_shape": list(self.v_compressed.shape),
        }


def estimate_tensor_bytes(t: torch.Tensor) -> int:
    """Return actual storage bytes for a tensor."""
    return int(t.nelement() * t.element_size())


def estimate_kv_memory(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Compute real KV tensor footprint in bytes."""
    k_bytes = estimate_tensor_bytes(k)
    v_bytes = estimate_tensor_bytes(v)
    meta_bytes = 0
    if metadata:
        for val in metadata.values():
            if isinstance(val, torch.Tensor):
                meta_bytes += estimate_tensor_bytes(val)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, torch.Tensor):
                        meta_bytes += estimate_tensor_bytes(item)
    total = k_bytes + v_bytes + meta_bytes
    return {
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "metadata_bytes": meta_bytes,
        "total_bytes": total,
    }


class KVCompressionKernel:
    """Tensor-level KV compression kernel with dequant materialization."""

    def __init__(
        self,
        *,
        int4_group_size: int = DEFAULT_INT4_GROUP_SIZE,
        block_size: int = DEFAULT_BLOCK_SPARSE_BLOCK_SIZE,
        block_drop_rate: float = DEFAULT_BLOCK_SPARSE_DROP_RATE,
    ) -> None:
        self.int4_group_size = int4_group_size
        self.block_size = block_size
        self.block_drop_rate = block_drop_rate

    def compress_kv(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mode: str,
        *,
        seed: int = 0,
    ) -> CompressedKVResult:
        """Compress K/V tensors; return stored form + dequantized materialization."""
        if mode not in KERNEL_MODES:
            msg = f"unsupported mode {mode!r}; choose from {KERNEL_MODES}"
            raise ValueError(msg)

        memory_before = estimate_kv_memory(k_cache, v_cache)["total_bytes"]

        if mode == "noop":
            return self._compress_noop(k_cache, v_cache, memory_before)
        if mode == "int8":
            return self._compress_int8(k_cache, v_cache, memory_before)
        if mode == "int4":
            return self._compress_int4(k_cache, v_cache, memory_before)
        return self._compress_block_sparse(k_cache, v_cache, memory_before, seed=seed)

    def _compress_noop(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_before: int,
    ) -> CompressedKVResult:
        k_c = k.clone()
        v_c = v.clone()
        memory_after = estimate_kv_memory(k_c, v_c)["total_bytes"]
        return CompressedKVResult(
            k_compressed=k_c,
            v_compressed=v_c,
            k_dequant=k_c,
            v_dequant=v_c,
            metadata=self._base_metadata("noop", memory_before, memory_after, k, v),
        )

    def _compress_int8(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_before: int,
    ) -> CompressedKVResult:
        k_q, k_scale = _symmetric_quantize_int8(k)
        v_q, v_scale = _symmetric_quantize_int8(v)
        k_d = _dequant_int8(k_q, k_scale)
        v_d = _dequant_int8(v_q, v_scale)
        meta = {
            "k_scale": k_scale,
            "v_scale": v_scale,
            "storage_dtype": "int8",
        }
        memory_after = estimate_kv_memory(k_q, v_q, metadata=meta)["total_bytes"]
        meta.update(self._base_metadata("int8", memory_before, memory_after, k, v))
        return CompressedKVResult(
            k_compressed=k_q,
            v_compressed=v_q,
            k_dequant=k_d,
            v_dequant=v_d,
            metadata=meta,
        )

    def _compress_int4(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_before: int,
    ) -> CompressedKVResult:
        k_q, k_scale, k_zp = _group_quantize_int4(k, self.int4_group_size)
        v_q, v_scale, v_zp = _group_quantize_int4(v, self.int4_group_size)
        k_packed = _pack_int4_nibbles(k_q)
        v_packed = _pack_int4_nibbles(v_q)
        k_d = _dequant_int4(k_q, k_scale, k_zp, self.int4_group_size, k.shape)
        v_d = _dequant_int4(v_q, v_scale, v_zp, self.int4_group_size, v.shape)
        meta = {
            "k_scale": k_scale,
            "v_scale": v_scale,
            "k_zero_point": k_zp,
            "v_zero_point": v_zp,
            "group_size": self.int4_group_size,
            "storage_dtype": "int4_packed_uint8",
        }
        memory_after = estimate_kv_memory(k_packed, v_packed, metadata=meta)["total_bytes"]
        meta.update(self._base_metadata("int4", memory_before, memory_after, k, v))
        return CompressedKVResult(
            k_compressed=k_packed,
            v_compressed=v_packed,
            k_dequant=k_d,
            v_dequant=v_d,
            metadata=meta,
        )

    def _compress_block_sparse(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        memory_before: int,
        *,
        seed: int,
    ) -> CompressedKVResult:
        k_c, v_c, mask, kept_blocks = _block_sparse_compact(
            k,
            v,
            block_size=self.block_size,
            drop_rate=self.block_drop_rate,
            seed=seed,
        )
        meta = {
            "block_mask": mask,
            "block_size": self.block_size,
            "kept_blocks": kept_blocks,
            "original_seq_len": k.shape[-2],
            "compressed_seq_len": k_c.shape[-2],
        }
        memory_after = estimate_kv_memory(k_c, v_c, metadata={"block_mask": mask})["total_bytes"]
        meta.update(self._base_metadata("block_sparse", memory_before, memory_after, k, v))
        return CompressedKVResult(
            k_compressed=k_c,
            v_compressed=v_c,
            k_dequant=k_c,
            v_dequant=v_c,
            metadata=meta,
        )

    @staticmethod
    def _base_metadata(
        mode: str,
        memory_before: int,
        memory_after: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> dict[str, Any]:
        ratio = memory_after / memory_before if memory_before > 0 else 1.0
        return {
            "compression_mode": mode,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "compression_ratio": ratio,
            "dtype_before": str(k.dtype),
            "dtype_after_storage": str(k.dtype),
        }


def _symmetric_quantize_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (t / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale.to(dtype=torch.float32)


def _dequant_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(dtype=scale.dtype) * scale


def _pack_int4_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4-range values (stored as int8) two per byte."""
    flat = (q.to(torch.int16) & 0x0F).to(torch.uint8).reshape(-1)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).contiguous()


def _group_quantize_int4(
    t: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise INT4 quantization; stores values in int8 containers."""
    orig_shape = t.shape
    head_dim = orig_shape[-1]
    flat = t.reshape(-1, head_dim)
    n_groups = (head_dim + group_size - 1) // group_size
    pad = n_groups * group_size - head_dim
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(flat.shape[0], n_groups, group_size)
    min_val = grouped.amin(dim=-1, keepdim=True)
    max_val = grouped.amax(dim=-1, keepdim=True)
    scale = ((max_val - min_val) / 15.0).clamp(min=1e-8)
    zero_point = (-min_val / scale).round().clamp(0, 15)
    q = ((grouped / scale) + zero_point).round().clamp(0, 15).to(torch.int8)
    q_flat = q.reshape(flat.shape[0], n_groups * group_size)[..., :head_dim]
    return q_flat.reshape(orig_shape), scale.squeeze(-1), zero_point.squeeze(-1)


def _dequant_int4(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
    orig_shape: torch.Size,
) -> torch.Tensor:
    head_dim = orig_shape[-1]
    flat = q.reshape(-1, head_dim)
    n_groups = (head_dim + group_size - 1) // group_size
    pad = n_groups * group_size - head_dim
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(flat.shape[0], n_groups, group_size)
    scale_e = scale.unsqueeze(-1)
    zp_e = zero_point.unsqueeze(-1)
    deq = (grouped.to(scale.dtype) - zp_e) * scale_e
    deq_flat = deq.reshape(flat.shape[0], n_groups * group_size)[..., :head_dim]
    return deq_flat.reshape(orig_shape)


def _block_sparse_compact(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    drop_rate: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Drop contiguous seq blocks deterministically; compact surviving tokens."""
    seq_len = k.shape[-2]
    n_blocks = max(seq_len // block_size, 1)
    gen = torch.Generator(device=k.device)
    gen.manual_seed(seed)
    keep = torch.rand(n_blocks, generator=gen, device=k.device) > drop_rate
    if not keep.any():
        keep[0] = True

    blocks_k: list[torch.Tensor] = []
    blocks_v: list[torch.Tensor] = []
    for i in range(n_blocks):
        if not bool(keep[i].item()):
            continue
        s = i * block_size
        e = min(s + block_size, seq_len)
        blocks_k.append(k[..., s:e, :])
        blocks_v.append(v[..., s:e, :])

    tail = seq_len % block_size
    if tail and n_blocks * block_size < seq_len:
        if bool(keep[-1].item()) if n_blocks > 0 else True:
            blocks_k.append(k[..., n_blocks * block_size :, :])
            blocks_v.append(v[..., n_blocks * block_size :, :])

    k_out = torch.cat(blocks_k, dim=-2) if blocks_k else k[..., :0, :].clone()
    v_out = torch.cat(blocks_v, dim=-2) if blocks_v else v[..., :0, :].clone()
    return k_out, v_out, keep, int(keep.sum().item())


def compress_from_phase_d_output(
    phase_d_output: dict[str, Any],
    mode: str,
    *,
    seed: int = 0,
    kernel: KVCompressionKernel | None = None,
) -> CompressedKVResult:
    """Drop-in: accept Phase D-style dict with k_cache / v_cache tensors."""
    k = phase_d_output["k_cache"]
    v = phase_d_output["v_cache"]
    k_kernel = kernel or KVCompressionKernel()
    return k_kernel.compress_kv(k, v, mode, seed=seed)
