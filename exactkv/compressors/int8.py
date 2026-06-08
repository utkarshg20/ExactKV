"""Int8Compressor — per-tensor symmetric INT8 quantisation for KV cache.

This is the first real compressor in ExactKV.  It is correctness-first:
materialize_for_draft dequantises back to fp32 so the full model can use the
cache directly.  In a performance-optimised implementation, the compressed cache
would stay in INT8 and kernel-level dequantisation would happen on the fly.

Quantisation formula (per-layer, per-key and per-value tensor independently):

    scale = max(abs(tensor)) / 127         # symmetric around zero
    q     = round(tensor / scale).clamp(-128, 127).to(torch.int8)

All-zero tensors are handled by using scale = 1.0.

The DynamicCache mutation issue: ``compress`` only *reads* the tensors from
``past_key_values`` — it never calls ``forward`` or modifies any tensor in
place.  The quantised copies (torch.int8) are independent objects.
``materialize_for_draft`` rebuilds a fresh cache from dequantised tensors;
it does not touch the stored compressed data.
"""
from __future__ import annotations

from typing import Any

import torch

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import (
    extract_kv_tensors,
    kv_seq_len,
    rebuild_cache,
)
from exactkv.compressors.base import CompressionStats

# Bytes per element for the original fp32 tensors.
_FP32_BYTES = 4
# Bytes per element for int8.
_INT8_BYTES = 1
# Bytes for two float64 scales per layer (k_scale, v_scale).
_SCALE_BYTES = 16


def _quantize(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric INT8 quantisation.

    Returns (q, scale) where q is on the same device as t.
    """
    abs_max = float(t.abs().max().item())
    scale = abs_max / 127.0 if abs_max != 0.0 else 1.0
    q = (t / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def _dequantize(
    q: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct an approximation of the original tensor."""
    return q.to(dtype=dtype, device=device) * scale


class Int8Compressor:
    """Per-tensor symmetric INT8 compressor for KV cache tensors."""

    name: str = "int8"

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Quantise all KV tensors to INT8.  Does NOT mutate ``state``."""
        k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
        seq_len = kv_seq_len(state.past_key_values)

        layers: list[dict[str, Any]] = []
        for k, v in zip(k_tensors, v_tensors):
            k_q, k_scale = _quantize(k)
            v_q, v_scale = _quantize(v)
            layers.append(
                {"k_q": k_q, "k_scale": k_scale, "v_q": v_q, "v_scale": v_scale}
            )

        data: dict[str, Any] = {
            "layers": layers,
            "cache_format": cache_format,
            "seq_len": seq_len,
            "dtype": state.dtype,
            "device": state.device,
        }

        return CompressedKVState(
            data=data,
            metadata={"next_token_id": state.next_token_id},
            compressor_name=self.name,
            logical_seq_len=state.seq_len,
            generated_ids=state.generated_ids,
            device=state.device,
        )

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """Dequantise and return a cache suitable for a forward pass.

        Creates fresh tensors — does NOT modify ``compressed.data``.
        """
        d = compressed.data
        dtype: torch.dtype = d["dtype"]
        device: torch.device = d["device"]
        seq_len: int = d["seq_len"]
        cache_format: str = d["cache_format"]

        k_dequant = [_dequantize(l["k_q"], l["k_scale"], dtype, device) for l in d["layers"]]
        v_dequant = [_dequantize(l["v_q"], l["v_scale"], dtype, device) for l in d["layers"]]

        return rebuild_cache(k_dequant, v_dequant, cache_format, seq_len)

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Recompress from the updated authoritative full state (V1 strategy)."""
        return self.compress(new_full_state)

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        layers = compressed.data["layers"]
        seq_len = compressed.data["seq_len"]

        full_bytes = sum(
            l["k_q"].nelement() * _FP32_BYTES + l["v_q"].nelement() * _FP32_BYTES
            for l in layers
        )
        compressed_bytes = sum(
            l["k_q"].nelement() * _INT8_BYTES
            + l["v_q"].nelement() * _INT8_BYTES
            + _SCALE_BYTES
            for l in layers
        )

        compression_ratio = compressed_bytes / max(full_bytes, 1)
        memory_reduction_factor = full_bytes / max(compressed_bytes, 1)

        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            memory_reduction_factor=memory_reduction_factor,
            seq_len=seq_len,
            num_layers=len(layers),
        )
