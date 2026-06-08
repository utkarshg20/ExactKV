"""KV-cache memory estimation metrics.

V1 disclaimer: These are byte-count estimates based on tensor sizes, not
wall-clock or peak-memory measurements.  No speedup claims are made.

For INT8, the compressed cache is ~4x smaller in tensor bytes (fp32 → int8),
plus a small overhead for scale factors.  For NoOp, compressed ≈ full.
For DebugNoise, compressed ≈ full (noise stored at fp32).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from exactkv.cache.utils import kv_total_bytes
from exactkv.runtime.prefill import prefill_to_full_state


@dataclass
class MemorySummary:
    """Byte-level KV-cache memory estimates for one prompt after prefill.

    Naming convention (matches docs/METRICS.md and CompressionStats):
        compression_ratio       = compressed_bytes / full_bytes  (< 1 means smaller, 1.0 for NoOp)
        memory_reduction_factor = full_bytes / compressed_bytes  (> 1 means savings, 1.0 for NoOp)
    """
    full_bytes: int                 # fp32 KV cache after prefilling the prompt
    compressed_bytes: int           # bytes according to compressor.stats()
    compression_ratio: float        # compressed_bytes / full_bytes; < 1 means compressed
    memory_reduction_factor: float  # full_bytes / compressed_bytes; > 1 means savings

    def to_dict(self) -> dict:
        return asdict(self)


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
        MemorySummary with full_bytes, compressed_bytes, and ratios.
    """
    # Prefill via shared helper (avoids duplicating encode → forward → FullKVState)
    full_state = prefill_to_full_state(runtime, prompt)
    full_bytes = kv_total_bytes(full_state.past_key_values)

    compressed = compressor.compress(full_state)
    stats = compressor.stats(compressed)
    compressed_bytes = max(stats.compressed_bytes, 1)

    compression_ratio = compressed_bytes / max(full_bytes, 1)
    memory_reduction_factor = full_bytes / compressed_bytes

    return MemorySummary(
        full_bytes=full_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=compression_ratio,
        memory_reduction_factor=memory_reduction_factor,
    )
