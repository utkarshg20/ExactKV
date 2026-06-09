"""LayerAwareVSimCompressor — simulated layer-aware V precision policy.

V7 Phase B: conservative simulated policy motivated by Phase A proxy divergence
analysis.  This does **not** use attention weights, Sparse V dequantization, or
TurboQuant+ layer-aware backends.

Registered default: ``k8_v4_boundary_v8_sim``
------------------------------------------------
* K: INT8 simulated quantisation on **all** layers.
* V: INT8 on **boundary** layers (first and last ``boundary_layers`` layers).
* V: INT4-range simulation on **interior** layers (stored in int8 containers).

Default ``boundary_layers=1`` → layer 0 and layer N-1 use V8; interior use V4-sim.

This is a **simulation** only: ``is_simulated=True``,
``supports_real_bytes_claim=False``.  ``stored_kv_bytes`` reflects int8 container
storage, not packed 4-bit bytes.
"""
from __future__ import annotations

from typing import Any

import torch

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import extract_kv_tensors, kv_seq_len, rebuild_cache
from exactkv.compressors.asymmetric_sim import (
    _FP32_BYTES,
    _SCALE_BYTES,
    _compress_side,
    _materialize_side,
    _side_metadata_bytes,
    _side_tensor_bytes,
)
from exactkv.compressors.base import CompressorCapabilities, CompressionStats


def _boundary_layer_indices(num_layers: int, boundary_layers: int) -> set[int]:
    """Return layer indices treated as boundary (first/last ``boundary_layers``)."""
    if boundary_layers < 0:
        raise ValueError(f"boundary_layers must be >= 0, got {boundary_layers}")
    if boundary_layers == 0:
        return set()
    n = min(boundary_layers, num_layers)
    return set(range(n)) | set(range(max(num_layers - n, 0), num_layers))


def _v_bits_for_layer(
    layer_idx: int,
    num_layers: int,
    boundary_layers: int,
    boundary_v_bits: int,
    interior_v_bits: int,
) -> int:
    if layer_idx in _boundary_layer_indices(num_layers, boundary_layers):
        return boundary_v_bits
    return interior_v_bits


class LayerAwareVSimCompressor:
    """Simulated layer-aware V compression with uniform K quantisation.

    Args:
        boundary_layers: Number of layers at the start and end of the stack that
            receive ``boundary_v_bits`` on the V side.  Default ``1``.
        k_bits: Key bit-width for all layers (default ``8``).
        boundary_v_bits: V bit-width on boundary layers (default ``8``).
        interior_v_bits: V bit-width on interior layers (default ``4``).
        name: Registry name (default generated from policy parameters).
    """

    def __init__(
        self,
        boundary_layers: int = 1,
        k_bits: int = 8,
        boundary_v_bits: int = 8,
        interior_v_bits: int = 4,
        name: str | None = None,
    ) -> None:
        if k_bits != 8:
            raise ValueError("LayerAwareVSimCompressor currently supports k_bits=8 only.")
        if boundary_v_bits != 8 or interior_v_bits != 4:
            raise ValueError(
                "k8_v4_boundary_v8_sim policy requires boundary_v_bits=8 and "
                "interior_v_bits=4."
            )
        if boundary_layers < 0:
            raise ValueError(f"boundary_layers must be >= 0, got {boundary_layers}")

        self._boundary_layers = boundary_layers
        self._k_bits = k_bits
        self._boundary_v_bits = boundary_v_bits
        self._interior_v_bits = interior_v_bits
        self.name = name or (
            f"k{k_bits}_v{interior_v_bits}_boundary_v{boundary_v_bits}_sim"
        )
        self.capabilities = self._build_capabilities()

    def _build_capabilities(self) -> CompressorCapabilities:
        notes = (
            "Layer-aware simulated V policy (V7 Phase B): K=INT8 on all layers; "
            f"V=INT8 on first/last {self._boundary_layers} layer(s), "
            f"V=INT4-sim on interior layers. "
            "This is a simulation — no true attention weights or Sparse V "
            "dequantization. "
            "Stored bytes use int8 containers for all quantised sides, not packed "
            "4-bit storage. supports_real_bytes_claim=False."
        )
        return CompressorCapabilities(
            name=self.name,
            compressor_type="quantization",
            is_simulated=True,
            supports_real_bytes_claim=False,
            supports_token_dropping=False,
            supports_quantization=True,
            key_bit_width=self._k_bits,
            value_bit_width=None,  # mixed per-layer: 8 on boundary, 4-sim interior
            asymmetric=True,
            notes=notes,
        )

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Compress K/V per layer.  Does NOT mutate ``state``."""
        k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
        seq_len = kv_seq_len(state.past_key_values)
        num_layers = len(k_tensors)

        layers: list[dict[str, Any]] = []
        for layer_idx, (k, v) in enumerate(zip(k_tensors, v_tensors)):
            v_bits = _v_bits_for_layer(
                layer_idx,
                num_layers,
                self._boundary_layers,
                self._boundary_v_bits,
                self._interior_v_bits,
            )
            layers.append({
                "k_data": _compress_side(k, self._k_bits),
                "v_data": _compress_side(v, v_bits),
                "v_bits": v_bits,
            })

        data: dict[str, Any] = {
            "layers": layers,
            "cache_format": cache_format,
            "seq_len": seq_len,
            "dtype": state.dtype,
            "device": state.device,
            "k_bits": self._k_bits,
            "boundary_layers": self._boundary_layers,
            "boundary_v_bits": self._boundary_v_bits,
            "interior_v_bits": self._interior_v_bits,
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
        """Dequantise per layer and return an HF-forward-usable cache."""
        d = compressed.data
        dtype: torch.dtype = d["dtype"]
        device: torch.device = d["device"]
        seq_len: int = d["seq_len"]
        cache_format: str = d["cache_format"]
        k_bits: int = d["k_bits"]

        k_out = [
            _materialize_side(layer["k_data"], k_bits, dtype, device)
            for layer in d["layers"]
        ]
        v_out = [
            _materialize_side(layer["v_data"], layer["v_bits"], dtype, device)
            for layer in d["layers"]
        ]
        return rebuild_cache(k_out, v_out, cache_format, seq_len)

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Recompress from the updated authoritative full state."""
        return self.compress(new_full_state)

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        """Return V5 workspace-aware byte statistics."""
        d = compressed.data
        k_bits: int = d["k_bits"]
        layers = d["layers"]
        seq_len = d["seq_len"]

        full_bytes = 0
        stored_tensor_bytes = 0
        scale_bytes = 0

        for layer in layers:
            k_data = layer["k_data"]
            v_data = layer["v_data"]
            v_bits: int = layer["v_bits"]

            k_nelems = int(k_data["q"].nelement())
            v_nelems = int(v_data["q"].nelement())

            full_bytes += (k_nelems + v_nelems) * _FP32_BYTES
            stored_tensor_bytes += (
                _side_tensor_bytes(k_data, k_bits)
                + _side_tensor_bytes(v_data, v_bits)
            )
            scale_bytes += (
                _side_metadata_bytes(k_bits)
                + _side_metadata_bytes(v_bits)
            )

        full_bytes = max(full_bytes, 1)
        stored_tensor_bytes = max(stored_tensor_bytes, 1)
        compressed_bytes = max(stored_tensor_bytes + scale_bytes, 1)
        materialized_working = full_bytes
        total_footprint = stored_tensor_bytes + materialized_working + scale_bytes

        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compressed_bytes / full_bytes,
            memory_reduction_factor=full_bytes / compressed_bytes,
            seq_len=seq_len,
            num_layers=len(layers),
            stored_kv_bytes=stored_tensor_bytes,
            materialized_working_kv_bytes=materialized_working,
            metadata_bytes=scale_bytes,
            temporary_workspace_bytes=0,
            total_kv_footprint_bytes=total_footprint,
        )


class K8V4BoundaryV8SimCompressor(LayerAwareVSimCompressor):
    """K=INT8 all layers; V=INT8 boundary / INT4-sim interior.  boundary_layers=1."""

    def __init__(self) -> None:
        super().__init__(boundary_layers=1, name="k8_v4_boundary_v8_sim")
