"""KV-cache memory estimation metrics.

V1 disclaimer: These are byte-count estimates based on tensor sizes, not
wall-clock or peak-memory measurements.  No speedup claims are made.

V5 workspace-aware extension: MemorySummary now distinguishes stored KV bytes,
materialized working bytes (dequantised for attention), compression metadata
(scales / zero-points), and temporary scratch buffers.  All figures are
conservative estimates derived from tensor sizes; no device profiling is
performed.

Field naming:
    compression_ratio       = compressed_bytes / full_bytes  (< 1 means smaller)
    memory_reduction_factor = full_bytes / compressed_bytes  (> 1 means savings)
    total_kv_footprint_bytes = stored + materialized_working + metadata + temporary
                               (conservative; counts working copy separately from stored)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from exactkv.cache.utils import kv_total_bytes
from exactkv.runtime.prefill import prefill_to_full_state


@dataclass
class MemorySummary:
    """Byte-level KV-cache memory estimates for one prompt after prefill.

    Existing fields (V1-V4, unchanged):
        full_bytes              fp32 KV cache after prefilling the prompt.
        compressed_bytes        stored + metadata bytes (backward-compat alias for
                                stored_kv_bytes + metadata_bytes).
        compression_ratio       compressed_bytes / full_bytes; < 1 means compressed.
        memory_reduction_factor full_bytes / compressed_bytes; > 1 means savings.

    V5 workspace-aware fields (default 0 / False / "" for backward compatibility):
        stored_kv_bytes         Bytes of quantised/compressed tensors only (no scales).
        materialized_working_kv_bytes
                                Bytes when stored cache is dequantised for attention.
                                Equals full_bytes for all current ExactKV compressors.
        metadata_bytes          Per-tensor scales, zero-points, or similar metadata.
        temporary_workspace_bytes
                                Conservative transient scratch estimate.
        total_kv_footprint_bytes
                                stored + materialized_working + metadata + temporary.
        supports_real_bytes_claim
                                True when all byte figures reflect real storage.
        is_simulated            True when sub-INT8 widths are stored in int8 containers.
        memory_claim_note       Human-readable honesty note about the figures.
    """
    # ---- existing fields (V1–V4) ----------------------------------------
    full_bytes: int
    compressed_bytes: int
    compression_ratio: float
    memory_reduction_factor: float
    # ---- V5 workspace-aware fields (additive, backward-compatible) --------
    stored_kv_bytes: int = field(default=0)
    materialized_working_kv_bytes: int = field(default=0)
    metadata_bytes: int = field(default=0)
    temporary_workspace_bytes: int = field(default=0)
    total_kv_footprint_bytes: int = field(default=0)
    supports_real_bytes_claim: bool = field(default=False)
    is_simulated: bool = field(default=False)
    memory_claim_note: str = field(default="")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_memory_claim_note(
    compressor_name: str,
    is_simulated: bool,
    supports_real_bytes_claim: bool,
) -> str:
    """Construct a human-readable honesty note for a MemorySummary."""
    if compressor_name == "noop":
        return (
            "NoOp compressor: stored_kv_bytes == full_bytes (no compression). "
            "materialized_working_kv_bytes == full_bytes. "
            "No memory reduction is claimed."
        )
    if compressor_name == "debug_noise":
        return (
            "DebugNoise compressor: stores noisy fp-precision copies of all KV tensors. "
            "No real compression. Exists only to test rejection/correction code paths."
        )
    if is_simulated:
        return (
            f"Simulated compressor ({compressor_name!r}): sub-INT8 values are stored in "
            "int8 containers — no real bit-packing. "
            "stored_kv_bytes reflects int8 container reality, NOT theoretical packed size. "
            "supports_real_bytes_claim=False. "
            "materialized_working_kv_bytes == full_bytes (dequantised for attention). "
            "Do not cite these figures as evidence of real packed-INT4/INT2 memory savings."
        )
    return (
        f"Real-storage compressor ({compressor_name!r}): "
        "stored_kv_bytes reflects genuine quantised storage. "
        "materialized_working_kv_bytes == full_bytes (dequantised for attention). "
        "supports_real_bytes_claim=True."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_kv_memory(
    runtime: Any,         # ModelRuntime
    prompt: str,
    compressor: Any,      # KVCompressor
) -> MemorySummary:
    """Run a single prefill and report byte-level memory estimates.

    This is a standalone helper — it does NOT share state with any ongoing
    ExactKV generation loop.  It runs one extra forward pass and discards the
    resulting state.

    Args:
        runtime:    Loaded ModelRuntime.
        prompt:     Plain-text prompt string.
        compressor: A KVCompressor instance.

    Returns:
        MemorySummary with V1–V4 fields and V5 workspace-aware fields.
    """
    full_state = prefill_to_full_state(runtime, prompt)
    full_bytes = kv_total_bytes(full_state.past_key_values)

    compressed = compressor.compress(full_state)
    stats = compressor.stats(compressed)
    compressed_bytes = max(stats.compressed_bytes, 1)

    compression_ratio = compressed_bytes / max(full_bytes, 1)
    memory_reduction_factor = full_bytes / compressed_bytes

    # V5: pull workspace-aware fields from the enriched CompressionStats.
    stored_kv_bytes = stats.stored_kv_bytes
    materialized_working_kv_bytes = stats.materialized_working_kv_bytes
    metadata_bytes = stats.metadata_bytes
    temporary_workspace_bytes = stats.temporary_workspace_bytes
    total_kv_footprint_bytes = stats.total_kv_footprint_bytes

    # Capabilities for honesty fields (graceful fallback if not present).
    caps = getattr(compressor, "capabilities", None)
    is_simulated: bool = caps.is_simulated if caps is not None else False
    supports_real: bool = caps.supports_real_bytes_claim if caps is not None else False

    note = _build_memory_claim_note(compressor.name, is_simulated, supports_real)

    return MemorySummary(
        full_bytes=full_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=compression_ratio,
        memory_reduction_factor=memory_reduction_factor,
        stored_kv_bytes=stored_kv_bytes,
        materialized_working_kv_bytes=materialized_working_kv_bytes,
        metadata_bytes=metadata_bytes,
        temporary_workspace_bytes=temporary_workspace_bytes,
        total_kv_footprint_bytes=total_kv_footprint_bytes,
        supports_real_bytes_claim=supports_real,
        is_simulated=is_simulated,
        memory_claim_note=note,
    )
