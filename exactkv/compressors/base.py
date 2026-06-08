"""KVCompressor protocol and CompressionStats data structure.

All compressors in ExactKV must satisfy the KVCompressor protocol.  This is a
structural protocol (typing.Protocol), so compressors do NOT need to inherit
from any base class — they just need to provide the methods listed below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState


@dataclass
class CompressorCapabilities:
    """Metadata describing a compressor's nature and claims.

    These fields help downstream tools (CLI, report renderer, sweep planner)
    understand what a compressor can and cannot claim.  They are purely
    informational and do not affect the compression algorithm itself.

    Fields
    ------
    name
        Canonical string name used in the registry (e.g. ``"int8"``).
    compressor_type
        Broad category: ``"identity"`` | ``"quantization"`` | ``"debug"``.
        Future categories include ``"token_dropping"`` and ``"mixed"``.
    is_simulated
        ``True`` if the compressor simulates a target algorithm in a wider
        dtype (e.g. INT4 stored as INT8 — no real bit-packing is done).
        ``False`` for genuinely implemented algorithms (INT8, identity, etc.).
    supports_real_bytes_claim
        ``True`` if the compressed storage genuinely uses fewer bytes than
        fp32 (e.g. INT8 q-tensors occupy 1 B/element vs 4 B for fp32).
        ``False`` for NoOp (same bytes) or debug compressors.
    supports_token_dropping
        ``True`` if the compressor discards KV tokens (e.g. SnapKV, H2O).
        All V1 compressors are ``False``.
    supports_quantization
        ``True`` if the compressor quantises KV values.
    key_bit_width
        Effective bit-width used for key tensors. ``None`` means full precision
        or not applicable (e.g. identity / noise compressors). ``8`` = INT8
        (real storage); ``4`` = INT4-range stored in an ``int8`` container
        (simulated); ``2`` = INT2-range stored in an ``int8`` container
        (simulated). Added in V4; default ``None`` is backward-compatible with
        all V1–V3 compressors.
    value_bit_width
        Effective bit-width used for value tensors. Same semantics as
        ``key_bit_width``.
    asymmetric
        ``True`` if ``key_bit_width != value_bit_width`` (or one side is
        ``None`` and the other is not). ``False`` for all symmetric V1–V3
        compressors. Added in V4; default ``False`` is backward-compatible.
    notes
        Free-form annotation for documentation / reporting.
    """

    name: str
    compressor_type: str              # "identity" | "quantization" | "debug"
    is_simulated: bool
    supports_real_bytes_claim: bool
    supports_token_dropping: bool
    supports_quantization: bool
    key_bit_width: int | None = field(default=None)
    value_bit_width: int | None = field(default=None)
    asymmetric: bool = field(default=False)
    notes: str = field(default="")


@dataclass
class CompressionStats:
    """Byte-level compression statistics for one compressed state snapshot.

    Naming convention (matches docs/METRICS.md):
        compression_ratio      = compressed_bytes / full_bytes  (< 1 means smaller, 1.0 for NoOp)
        memory_reduction_factor = full_bytes / compressed_bytes (> 1 means savings, 1.0 for NoOp)
    """
    compressor_name: str
    full_bytes: int
    compressed_bytes: int
    compression_ratio: float        # compressed_bytes / full_bytes; < 1 means compressed
    memory_reduction_factor: float  # full_bytes / compressed_bytes; > 1 means savings
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
