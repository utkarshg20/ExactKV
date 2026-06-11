"""Restricted local serving-context / cache-lifecycle harness (V8 Phase B).

This package models serving-style KV cache ownership and lifecycle concepts
without vLLM, LMCache, or PagedAttention integration.  It wraps existing
``FullKVState`` and ``CompressedKVState`` objects for compatibility evaluation
only — it does not replace ``ExactKVGenerator`` or ``VerificationEngine``.
"""
from __future__ import annotations

from exactkv.serving.cache_lifecycle import (
    AUTHORITATIVE_FULL,
    COMPRESSED_DRAFT,
    SERVING_HARNESS,
    CacheBlock,
    CacheOwner,
    ServingCacheEntry,
    ServingCacheLifecycleHarness,
    build_blocks,
    infer_physical_seq_len,
    validate_retained_logical_positions,
)
from exactkv.serving.sidecar_probe import (
    PROBE_INVARIANTS,
    ProbeRoundObservation,
    ServingSidecarProbe,
    run_exactkv_with_sidecar_probe,
)

__all__ = [
    "AUTHORITATIVE_FULL",
    "COMPRESSED_DRAFT",
    "SERVING_HARNESS",
    "CacheBlock",
    "CacheOwner",
    "PROBE_INVARIANTS",
    "ProbeRoundObservation",
    "ServingCacheEntry",
    "ServingCacheLifecycleHarness",
    "ServingSidecarProbe",
    "build_blocks",
    "infer_physical_seq_len",
    "run_exactkv_with_sidecar_probe",
    "validate_retained_logical_positions",
]
