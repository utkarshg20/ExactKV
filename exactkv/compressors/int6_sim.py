"""Int6SimCompressor — per-tensor symmetric INT6 quantisation simulation.

INT6 quantisation clips KV tensors to 64 discrete levels (−32 to +31 after
symmetric quantisation).  This sits between INT8 (256 levels, low error) and
INT4-sim (16 levels, high error), giving ExactKV a *non-catastrophic* compressor
that produces measurable but non-trivial divergence.

Quantisation formula (per-layer, per-tensor):

    scale = max(abs(tensor)) / 31         # symmetric, 6-bit signed range
    q     = round(tensor / scale).clamp(-32, 31)   # 64 discrete levels

Like ``int4_sim``, this is a *simulation*: values are stored as float32/float16
(no actual int6 packing), so the byte claim is NOT faithful.  It demonstrates
the drift profile of 6-bit KV quantisation.

Compression ratio (simulated):
    theoretical  = 6 / 16  = 0.375 (int6 vs fp16 per element)
    stored here  = 1.0     (float16 simulation, no packing)
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

# 6-bit signed range: -32 to +31
_INT6_MIN = -32
_INT6_MAX = 31
_INT6_LEVELS = 64  # 2^6

# Byte accounting
_FP32_BYTES = 4
_FP16_BYTES = 2  # simulated storage unit
_SCALE_BYTES = 16  # two float64 per layer


def _quantize_int6(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric INT6 quantisation simulation.

    Returns (q_approx, scale) where q_approx is a float tensor rounded to
    INT6 discrete values, stored as the original dtype.
    """
    abs_max = float(t.abs().max().item())
    scale = abs_max / float(_INT6_MAX) if abs_max != 0.0 else 1.0
    q_int = (t / scale).round().clamp(_INT6_MIN, _INT6_MAX)
    # Dequantize immediately — this is a simulation, not real packing
    return q_int * scale, scale


class Int6SimCompressor:
    """Per-tensor symmetric INT6 quantisation simulation for KV cache.

    Non-catastrophic intermediate compressor: divergence expected to fall
    between int8 (near-zero) and int4_sim (50–90%) across task families.
    Use for compression-curve experiments or as a stepping stone toward
    production INT6/FP6 KV quantisation.
    """

    name: str = "int6_sim"

    capabilities: CompressorCapabilities = CompressorCapabilities(
        name="int6_sim",
        compressor_type="quantization",
        is_simulated=True,
        supports_real_bytes_claim=False,
        supports_token_dropping=False,
        supports_quantization=True,
        key_bit_width=6,
        value_bit_width=6,
        asymmetric=False,
        notes=(
            "Per-tensor symmetric INT6 quantisation simulation (64 levels, "
            "scale = max(|x|) / 31). Stored as float16 — no real 6-bit packing. "
            "Theoretical compression: 6/16 = 0.375 vs fp16. "
            "Divergence expected between int8 (near-zero) and int4_sim (50–90%). "
            "Use for compression-curve experiments only."
        ),
    )

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Simulate INT6 quantisation. Does NOT mutate ``state``."""
        k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
        seq_len = kv_seq_len(state.past_key_values)

        layers: list[dict[str, Any]] = []
        for k, v in zip(k_tensors, v_tensors):
            k_approx, k_scale = _quantize_int6(k)
            v_approx, v_scale = _quantize_int6(v)
            layers.append({
                "k_approx": k_approx,
                "k_scale": k_scale,
                "v_approx": v_approx,
                "v_scale": v_scale,
            })

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
        """Return a cache from the INT6-quantised approximation."""
        d = compressed.data
        dtype: torch.dtype = d["dtype"]
        device: torch.device = d["device"]
        seq_len: int = d["seq_len"]
        cache_format: str = d["cache_format"]

        k_out = [l["k_approx"].to(dtype=dtype, device=device) for l in d["layers"]]
        v_out = [l["v_approx"].to(dtype=dtype, device=device) for l in d["layers"]]

        return rebuild_cache(k_out, v_out, cache_format, seq_len)

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Recompress from the updated authoritative full state."""
        return self.compress(new_full_state)

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        layers = compressed.data["layers"]
        seq_len = compressed.data["seq_len"]

        full_bytes = sum(
            l["k_approx"].nelement() * _FP32_BYTES
            + l["v_approx"].nelement() * _FP32_BYTES
            for l in layers
        )
        # Simulated: stored as fp16 (not actually packed to 6 bits)
        stored_bytes = sum(
            l["k_approx"].nelement() * _FP16_BYTES
            + l["v_approx"].nelement() * _FP16_BYTES
            for l in layers
        )
        scale_bytes = len(layers) * _SCALE_BYTES
        compressed_bytes = stored_bytes + scale_bytes
        # Theoretical 6-bit ratio
        theoretical_bytes = sum(
            int(l["k_approx"].nelement() * 6 / 8)
            + int(l["v_approx"].nelement() * 6 / 8)
            for l in layers
        ) + scale_bytes

        compression_ratio = theoretical_bytes / max(full_bytes, 1)
        memory_reduction_factor = full_bytes / max(theoretical_bytes, 1)
        materialized_working = full_bytes

        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            memory_reduction_factor=memory_reduction_factor,
            seq_len=seq_len,
            num_layers=len(layers),
            stored_kv_bytes=stored_bytes,
            materialized_working_kv_bytes=materialized_working,
            metadata_bytes=scale_bytes,
            temporary_workspace_bytes=0,
            total_kv_footprint_bytes=stored_bytes + materialized_working + scale_bytes,
        )
