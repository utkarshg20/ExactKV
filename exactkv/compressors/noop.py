"""NoOpCompressor — identity compressor for debugging.

Returns the full past_key_values unchanged.  Because drafting uses the exact
same KV cache as verification, every drafted token is identical to what the
full model would predict.  Acceptance rate must be exactly 1.0.

This compressor is the ground-truth check for the ExactKV loop: if ExactKV
with NoOp does not produce the same output as generate_full_greedy, there is a
bug in the loop logic — not in the compressor.
"""
from __future__ import annotations

from typing import Any

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import kv_seq_len, kv_total_bytes
from exactkv.compressors.base import CompressionStats


class NoOpCompressor:
    """Identity compressor: full KV cache is used as-is for drafting."""

    name: str = "noop"

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Wrap the full past_key_values without modification."""
        return CompressedKVState(
            data=state.past_key_values,
            metadata={"next_token_id": state.next_token_id},
            compressor_name=self.name,
            logical_seq_len=state.seq_len,
            generated_ids=state.generated_ids,
            device=state.device,
        )

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """Return the stored past_key_values unchanged."""
        return compressed.data

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Recompress from the new authoritative full state (V1 strategy)."""
        return self.compress(new_full_state)

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        full_bytes = kv_total_bytes(compressed.data)
        num_layers = (
            len(compressed.data)
            if isinstance(compressed.data, tuple)
            else 0
        )
        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=full_bytes,    # NoOp: no reduction
            compression_ratio=1.0,          # compressed / full = 1.0
            memory_reduction_factor=1.0,    # full / compressed = 1.0
            seq_len=kv_seq_len(compressed.data),
            num_layers=num_layers,
        )
