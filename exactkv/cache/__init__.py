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
from exactkv.cache.storage import (
    FileKVStorageBackend,
    InMemoryKVStorageBackend,
    KVStorageBackend,
    KVStorageHandle,
    KVStorageMetadata,
    KVStorageNotFoundError,
    StoredKVEntry,
    build_verifier_storage_metadata,
    cache_view_from_storage_metadata,
    dual_cache_with_stored_verifier,
    smoke_store_verifier_payload,
    validate_storage_metadata,
)

__all__ = [
    "CacheMaterialization",
    "CacheResidency",
    "CacheRole",
    "CacheView",
    "CompressedKVState",
    "DualCacheState",
    "FileKVStorageBackend",
    "FullKVState",
    "InMemoryKVStorageBackend",
    "KVStorageBackend",
    "KVStorageHandle",
    "KVStorageMetadata",
    "KVStorageNotFoundError",
    "StoredKVEntry",
    "build_identity_dual_cache",
    "build_verifier_storage_metadata",
    "cache_view_from_storage_metadata",
    "dual_cache_with_stored_verifier",
    "smoke_store_verifier_payload",
    "validate_cache_view",
    "validate_dual_cache_state",
    "validate_storage_metadata",
]
