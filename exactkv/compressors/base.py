"""KVCompressor protocol and CompressionStats data structure.

All compressors in ExactKV must satisfy the KVCompressor protocol.  This is a
structural protocol (typing.Protocol), so compressors do NOT need to inherit
from any base class — they just need to provide the methods listed below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState


@dataclass
class CompressionStats:
    """Byte-level compression statistics for one compressed state snapshot."""
    compressor_name: str
    full_bytes: int
    compressed_bytes: int
    compression_ratio: float   # full_bytes / compressed_bytes; 1.0 for NoOp
    seq_len: int
    num_layers: int


class KVCompressor(Protocol):
    """Structural interface all ExactKV compressors must satisfy.

    Implementations
    ---------------
    NoOpCompressor   — identity; returns full KV unchanged (debugging).
    Int8Compressor   — per-tensor symmetric INT8 quantisation (Step 9+).

    The three-method lifecycle:

    1. ``compress(full_state)``
       Called once after prefill and once after each commit (via
       ``update_after_commit``).  Creates a CompressedKVState from the
       authoritative FullKVState.

    2. ``materialize_for_draft(compressed)``
       Returns the past_key_values (or equivalent) to feed to the draft
       model.  For NoOp this is the identical object; for INT8 it would
       dequantise to fp32/fp16.

    3. ``update_after_commit(compressed, new_full_state)``
       V1 strategy: recompress from the authoritative full state.
       This is inefficient (ignores incremental structure) but correct.
    """

    name: str

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Create a compressed representation of the authoritative KV state."""
        ...

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """Return past_key_values suitable for feeding to the draft model."""
        ...

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Recompress after committing tokens (V1: always recompresses from full)."""
        ...

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        """Return byte-level statistics for this compressed state."""
        ...
