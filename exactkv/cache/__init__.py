"""ExactKV cache state and dual-cache contracts."""
from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.dual_cache import (
    CacheMaterialization,
    CacheResidency,
    CacheRole,
    CacheView,
    DualCacheState,
    build_identity_dual_cache,
    validate_cache_view,
    validate_dual_cache_state,
)
from exactkv.cache.full_state import FullKVState

__all__ = [
    "CacheMaterialization",
    "CacheResidency",
    "CacheRole",
    "CacheView",
    "CompressedKVState",
    "DualCacheState",
    "FullKVState",
    "build_identity_dual_cache",
    "validate_cache_view",
    "validate_dual_cache_state",
]
