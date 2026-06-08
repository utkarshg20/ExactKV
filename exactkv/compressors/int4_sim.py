"""Int4SimCompressor — simulated INT4 quantisation for KV cache.

What "simulated" means here
---------------------------
This compressor applies per-tensor symmetric INT4-range quantisation:
the quantised values are clipped to [-8, 7] (signed 4-bit range), but
the *storage* format is ``torch.int8`` (1 byte per element).  No real
4-bit bit-packing is performed.

Consequence for memory statistics
-----------------------------------
``compressed_bytes`` in ``CompressionStats`` reflects the **actual storage**
(int8 containers, 1 byte/element + scale overhead).  The theoretical
minimum for truly packed 4-bit storage would be 0.5 bytes/element — roughly
half that figure.  Because ``CompressorCapabilities.supports_real_bytes_claim``
is ``False``, consumers must not interpret these byte counts as real INT4
memory savings.

Quantisation formula (per-layer, per-key and per-value tensor independently)::

    scale = max(abs(tensor)) / 7       # symmetric, centred on zero
    q     = round(tensor / scale).clamp(-8, 7).to(torch.int8)

All-zero tensors are handled by substituting scale = 1.0.

DynamicCache safety: ``compress`` only *reads* tensors from
``past_key_values`` and creates independent copies.  It never calls
``forward`` or modifies any tensor in-place.  ``materialize_for_draft``
rebuilds a fresh cache from dequantised tensors; it does not touch
``compressed.data``.
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
from exactkv.compressors.base import CompressorCapabilities, CompressionStats

# Signed INT4 range used for quantisation.
_Q_MIN: int = -8
_Q_MAX: int = 7
_Q_SCALE_MAX: float = 7.0   # max(abs(q)) for symmetric range

# Storage is int8 (1 byte/element); theoretical INT4 would be 0.5 bytes/element.
_STORAGE_BYTES_PER_ELEMENT: int = 1   # int8 container
# Bytes for two float64 scales per layer (k_scale, v_scale).
_SCALE_BYTES: int = 16
# Bytes for a full fp32 element (reference for computing compression_ratio).
_FP32_BYTES: int = 4

_NOTES = (
    "Simulated INT4: values quantised to the signed 4-bit range [-8, 7] but "
    "stored in torch.int8 (1 byte/element, NOT real 4-bit packed). "
    "Theoretical INT4 packed storage would be ~0.5 bytes/element; "
    "compressed_bytes here reports actual int8 storage. "
    "Use this compressor for acceptance-rate and correctness experiments only. "
    "Do not cite its CompressionStats as evidence of real INT4 memory savings."
)


def _quantize_int4(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric INT4-range quantisation stored in int8.

    Args:
        t: Input floating-point tensor (any shape, any device).

    Returns:
        (q, scale) where ``q`` is torch.int8 with values in [-8, 7].
    """
    abs_max = float(t.abs().max().item())
    scale = abs_max / _Q_SCALE_MAX if abs_max != 0.0 else 1.0
    q = (t / scale).round().clamp(_Q_MIN, _Q_MAX).to(torch.int8)
    return q, scale


def _dequantize(
    q: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct an approximation of the original tensor."""
    return q.to(dtype=dtype, device=device) * scale


class Int4SimCompressor:
    """Simulated INT4 compressor for KV cache tensors.

    Quantises KV values to the INT4 numeric range ([-8, 7]) but stores them in
    ``torch.int8`` containers.  Dequantises back to the model's fp32/fp16/bf16
    dtype when materialising for draft.

    This compressor is useful for measuring how INT4-level quantisation
    error affects the ExactKV acceptance rate and correction frequency,
    without requiring real 4-bit CUDA kernel support.

    Do NOT use for memory-efficiency claims; see ``capabilities.notes``.
    """

    name: str = "int4_sim"

    capabilities: CompressorCapabilities = CompressorCapabilities(
        name="int4_sim",
        compressor_type="quantization",
        is_simulated=True,
        supports_real_bytes_claim=False,
        supports_token_dropping=False,
        supports_quantization=True,
        key_bit_width=4,
        value_bit_width=4,
        asymmetric=False,
        notes=_NOTES,
    )

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Quantise all KV tensors to INT4 range.  Does NOT mutate ``state``."""
        k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
        seq_len = kv_seq_len(state.past_key_values)

        layers: list[dict[str, Any]] = []
        for k, v in zip(k_tensors, v_tensors):
            k_q, k_scale = _quantize_int4(k)
            v_q, v_scale = _quantize_int4(v)
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
        """Return byte-level statistics.

        ``compressed_bytes`` reflects **actual int8 storage** (1 byte/element),
        NOT theoretical INT4 packed size (0.5 bytes/element).  See
        ``capabilities.notes`` and ``capabilities.supports_real_bytes_claim``
        for the correct interpretation.

        V5 workspace-aware fields:
            stored_kv_bytes             int8 tensor bytes only (no scales).
            metadata_bytes              float64 scale bytes (2 per layer × 8 B).
            materialized_working_kv_bytes  full_bytes (dequantised for attention).
            total_kv_footprint_bytes    stored + materialized + metadata + scratch.
        """
        layers = compressed.data["layers"]
        seq_len = compressed.data["seq_len"]

        # fp32 reference bytes (4 bytes/element).
        full_bytes = sum(
            l["k_q"].nelement() * _FP32_BYTES + l["v_q"].nelement() * _FP32_BYTES
            for l in layers
        )
        # V5: separate tensor storage from scale metadata.
        # stored_kv_bytes = int8 container bytes only (NOT real 4-bit packed).
        # metadata_bytes  = two float64 scales per layer (k_scale, v_scale = 16 B).
        stored_tensor_bytes = sum(
            l["k_q"].nelement() * _STORAGE_BYTES_PER_ELEMENT
            + l["v_q"].nelement() * _STORAGE_BYTES_PER_ELEMENT
            for l in layers
        )
        scale_bytes = len(layers) * _SCALE_BYTES

        # compressed_bytes kept as original sum for backward compatibility.
        actual_bytes = stored_tensor_bytes + scale_bytes

        compression_ratio = actual_bytes / max(full_bytes, 1)
        memory_reduction_factor = full_bytes / max(actual_bytes, 1)

        # Dequantisation creates a full-precision working copy for attention.
        materialized_working = full_bytes
        total_footprint = stored_tensor_bytes + materialized_working + scale_bytes

        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=actual_bytes,       # int8 storage, NOT real 4-bit
            compression_ratio=compression_ratio,
            memory_reduction_factor=memory_reduction_factor,
            seq_len=seq_len,
            num_layers=len(layers),
            stored_kv_bytes=stored_tensor_bytes,
            materialized_working_kv_bytes=materialized_working,
            metadata_bytes=scale_bytes,
            temporary_workspace_bytes=0,
            total_kv_footprint_bytes=total_footprint,
        )
