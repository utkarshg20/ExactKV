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

__all__ = [
    "AUTHORITATIVE_FULL",
    "COMPRESSED_DRAFT",
    "SERVING_HARNESS",
    "CacheBlock",
    "CacheOwner",
    "ServingCacheEntry",
    "ServingCacheLifecycleHarness",
    "build_blocks",
    "infer_physical_seq_len",
    "validate_retained_logical_positions",
]
