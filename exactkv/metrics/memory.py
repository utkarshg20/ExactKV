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

import torch

from exactkv.cache.utils import kv_total_bytes


@dataclass
class MemorySummary:
    """Byte-level KV-cache memory estimates for one prompt after prefill."""
    full_bytes: int           # fp32 KV cache after prefilling the prompt
    compressed_bytes: int     # bytes according to compressor.stats()
    compression_ratio: float  # full_bytes / compressed_bytes; 1.0 for NoOp
    memory_reduction_factor: float  # alias for compression_ratio (kept separate
                                    # so callers can add wall-time later)

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
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
    from exactkv.cache.full_state import FullKVState

    prompt_ids = runtime.encode(prompt)
    out = runtime.forward(prompt_ids, past_key_values=None, use_cache=True)
    next_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())

    empty_gen = torch.zeros(1, 0, dtype=torch.long, device=runtime.device)
    full_state = FullKVState(
        past_key_values=out.past_key_values,
        prompt_ids=prompt_ids,
        generated_ids=empty_gen,
        full_sequence_ids=prompt_ids,
        device=runtime.device,
        dtype=runtime.dtype,
        metadata={"next_token_id": next_tok},
    )

    full_bytes = kv_total_bytes(full_state.past_key_values)

    compressed = compressor.compress(full_state)
    stats = compressor.stats(compressed)
    compressed_bytes = max(stats.compressed_bytes, 1)

    ratio = full_bytes / compressed_bytes

    return MemorySummary(
        full_bytes=full_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=ratio,
        memory_reduction_factor=ratio,
    )
