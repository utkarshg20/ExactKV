"""DebugNoiseCompressor — intentionally broken compressor for rejection testing.

Purpose
-------
This compressor exists ONLY to exercise the rejection and correction code paths
in the VerificationEngine and ExactKVGenerator.  It must NOT appear in any
serious benchmark claim or performance comparison.

How it forces rejection
-----------------------
1. KV tensors are additively perturbed with large Gaussian noise
   (``noise_scale * N(0,1)``), making the draft model's predictions
   effectively random.

2. ``next_token_id`` in the compressed metadata is set to
   ``(full_state.next_token_id + 1) % _WRONG_TOKEN_MODULUS``, which
   guarantees the first drafted token always differs from the full model's
   prediction.  This forces a correction in EVERY round, making acceptance
   rate exactly 0.0 for normal text prompts.

DynamicCache safety: ``compress`` only reads from ``full_state.past_key_values``
and creates new tensors (the noisy copies).  It never calls ``forward`` and
never modifies any tensor in-place.  ``materialize_for_draft`` rebuilds a
fresh cache from the noisy copies.
"""
from __future__ import annotations

from typing import Any

import torch

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import (
    extract_kv_tensors,
    kv_seq_len,
    kv_total_bytes,
    rebuild_cache,
)
from exactkv.compressors.base import CompressorCapabilities, CompressionStats

# We use modular arithmetic to guarantee next_token_id differs from the full
# model's prediction.  Any modulus > 1 works; we pick a small prime so the
# result is clearly "wrong" for natural-language prompts.
_WRONG_TOKEN_MODULUS = 100_003


class DebugNoiseCompressor:
    """Intentionally breaks KV cache predictions to force ExactKV rejection.

    Do NOT use in real benchmarks.
    """

    name: str = "debug_noise"

    capabilities: CompressorCapabilities = CompressorCapabilities(
        name="debug_noise",
        compressor_type="debug",
        is_simulated=True,
        supports_real_bytes_claim=False,
        supports_token_dropping=False,
        supports_quantization=False,
        notes=(
            "Artificially perturbs KV tensors with large Gaussian noise to force "
            "rejection in every verification round.  Exists exclusively to test the "
            "rejection and correction code paths in VerificationEngine and "
            "ExactKVGenerator.  Must not appear in real benchmark comparisons."
        ),
    )

    def __init__(self, noise_scale: float = 10.0) -> None:
        self.noise_scale = noise_scale

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Add large Gaussian noise to all KV tensors.  Does NOT mutate ``state``."""
        k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
        seq_len = kv_seq_len(state.past_key_values)

        k_noisy = [k + torch.randn_like(k) * self.noise_scale for k in k_tensors]
        v_noisy = [v + torch.randn_like(v) * self.noise_scale for v in v_tensors]

        data: dict[str, Any] = {
            "k_noisy": k_noisy,
            "v_noisy": v_noisy,
            "cache_format": cache_format,
            "seq_len": seq_len,
            "dtype": state.dtype,
            "device": state.device,
        }

        # Guarantee a wrong first draft token by incrementing modulo a prime.
        # This ensures first_draft_token != full_state.next_token_id for all
        # natural-language prompts (unless the increment wraps to the correct
        # token by coincidence — astronomically unlikely in practice).
        wrong_next = (state.next_token_id + 1) % _WRONG_TOKEN_MODULUS

        return CompressedKVState(
            data=data,
            metadata={"next_token_id": wrong_next},
            compressor_name=self.name,
            logical_seq_len=state.seq_len,
            generated_ids=state.generated_ids,
            device=state.device,
        )

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """Return a cache built from the noisy tensors."""
        d = compressed.data
        return rebuild_cache(
            d["k_noisy"], d["v_noisy"], d["cache_format"], d["seq_len"]
        )

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Recompress (with fresh noise) from the updated authoritative state."""
        return self.compress(new_full_state)

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        d = compressed.data
        # Noisy tensors are the same dtype/size as originals → ratio ≈ 1.0
        k_bytes = sum(k.nelement() * k.element_size() for k in d["k_noisy"])
        v_bytes = sum(v.nelement() * v.element_size() for v in d["v_noisy"])
        total = k_bytes + v_bytes
        return CompressionStats(
            compressor_name=self.name,
            full_bytes=total,
            compressed_bytes=total,
            compression_ratio=1.0,         # compressed / full = 1.0 (no size reduction)
            memory_reduction_factor=1.0,   # full / compressed = 1.0
            seq_len=d["seq_len"],
            num_layers=len(d["k_noisy"]),
        )
