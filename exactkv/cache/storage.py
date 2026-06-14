"""Full-KV storage manager — design spike (Phase 11C).

Pluggable backends for serializing, storing, and reloading tiny authoritative
verifier KV payloads. **Not** wired into ``ExactKVGenerator`` or verification.

This is a storage contract and tiny payload smoke, not a serving runtime.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from exactkv.cache.dual_cache import (
    CacheMaterialization,
    CacheResidency,
    CacheRole,
    CacheView,
    DualCacheState,
    validate_cache_view,
    validate_dual_cache_state,
)

_STORAGE_CLAIM_NOTE = (
    "Full-KV storage manager design spike (Phase 11C). "
    "Payload byte counts are diagnostic only. "
    "No active GPU memory savings, CPU offload, disk offload, or serving claim."
)


class KVStorageError(Exception):
    """Base error for storage manager operations."""


class KVStorageNotFoundError(KVStorageError):
    """Raised when a handle is not present in the backend."""


@dataclass(frozen=True)
class KVStorageHandle:
    """Logical address for a stored KV payload."""

    namespace: str
    key: str
    version: str = "1"

    def filename_stem(self) -> str:
        safe_ns = self.namespace.replace("/", "_")
        return f"{safe_ns}__{self.key}__v{self.version}"


@dataclass
class KVStorageMetadata:
    """Serializable metadata for a stored full-KV verifier payload."""

    role: CacheRole
    residency: CacheResidency
    materialization: CacheMaterialization
    tensor_count: int
    total_payload_bytes: int
    metadata_bytes: int
    dtype_summary: str
    shape_summary: str
    claim_note: str
    backend_name: str = "full_kv_storage"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        d["residency"] = self.residency.value
        d["materialization"] = self.materialization.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KVStorageMetadata:
        return cls(
            role=CacheRole(data["role"]),
            residency=CacheResidency(data.get("residency", CacheResidency.UNKNOWN.value)),
            materialization=CacheMaterialization(
                data.get("materialization", CacheMaterialization.UNKNOWN.value)
            ),
            tensor_count=int(data.get("tensor_count", 0)),
            total_payload_bytes=int(data.get("total_payload_bytes", 0)),
            metadata_bytes=int(data.get("metadata_bytes", 0)),
            dtype_summary=str(data.get("dtype_summary", "")),
            shape_summary=str(data.get("shape_summary", "")),
            claim_note=str(data.get("claim_note", "")),
            backend_name=str(data.get("backend_name", "full_kv_storage")),
        )


@dataclass
class StoredKVEntry:
    """Payload + metadata returned from ``get``."""

    payload: Any
    metadata: KVStorageMetadata


class KVStorageBackend(ABC):
    """Abstract full-KV storage backend."""

    @abstractmethod
    def put(self, handle: KVStorageHandle, payload: Any, metadata: KVStorageMetadata) -> None:
        """Store payload and metadata under handle."""

    @abstractmethod
    def get(self, handle: KVStorageHandle) -> StoredKVEntry:
        """Load payload and metadata; raise ``KVStorageNotFoundError`` if missing."""

    @abstractmethod
    def exists(self, handle: KVStorageHandle) -> bool:
        """Return whether handle is stored."""

    @abstractmethod
    def delete(self, handle: KVStorageHandle) -> None:
        """Remove stored entry if present."""

    @abstractmethod
    def metadata(self, handle: KVStorageHandle) -> KVStorageMetadata:
        """Return metadata only; raise ``KVStorageNotFoundError`` if missing."""


def iter_tensors(payload: Any) -> list[torch.Tensor]:
    """Collect torch tensors from nested list/tuple/dict structures."""
    found: list[torch.Tensor] = []

    def _walk(obj: Any) -> None:
        if torch.is_tensor(obj):
            found.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                _walk(item)

    _walk(payload)
    return found


def summarize_tensor_payload(payload: Any) -> tuple[int, int, str, str]:
    """Return (tensor_count, total_payload_bytes, dtype_summary, shape_summary)."""
    tensors = iter_tensors(payload)
    if not tensors:
        return 0, 0, "none", "none"
    total_bytes = sum(int(t.numel() * t.element_size()) for t in tensors)
    dtypes = sorted({str(t.dtype) for t in tensors})
    shapes = [tuple(t.shape) for t in tensors[:8]]
    more = "" if len(tensors) <= 8 else f";+{len(tensors) - 8} more"
    dtype_summary = ",".join(dtypes)
    shape_summary = repr(shapes) + more
    return len(tensors), total_bytes, dtype_summary, shape_summary


def build_verifier_storage_metadata(
    payload: Any,
    *,
    residency: CacheResidency,
    backend_name: str = "full_kv_storage",
    claim_note: str = _STORAGE_CLAIM_NOTE,
) -> KVStorageMetadata:
    """Build conservative verifier metadata from a tensor payload."""
    count, total_bytes, dtype_summary, shape_summary = summarize_tensor_payload(payload)
    meta = KVStorageMetadata(
        role=CacheRole.VERIFIER,
        residency=residency,
        materialization=CacheMaterialization.FULL,
        tensor_count=count,
        total_payload_bytes=total_bytes,
        metadata_bytes=0,  # filled by backend after JSON encode
        dtype_summary=dtype_summary,
        shape_summary=shape_summary,
        claim_note=claim_note,
        backend_name=backend_name,
    )
    meta.metadata_bytes = len(json.dumps(meta.to_dict(), sort_keys=True).encode("utf-8"))
    return meta


def validate_storage_metadata(metadata: KVStorageMetadata, payload: Any) -> list[str]:
    """Validate metadata against payload invariants."""
    errors: list[str] = []

    if metadata.total_payload_bytes < 0:
        errors.append("total_payload_bytes must be non-negative")
    if metadata.metadata_bytes < 0:
        errors.append("metadata_bytes must be non-negative")
    if metadata.tensor_count < 0:
        errors.append("tensor_count must be non-negative")

    tensors = iter_tensors(payload)
    if metadata.tensor_count != len(tensors):
        errors.append(
            f"tensor_count mismatch: metadata={metadata.tensor_count}, "
            f"payload={len(tensors)}"
        )

    _, actual_bytes, _, _ = summarize_tensor_payload(payload)
    if metadata.tensor_count > 0 and metadata.total_payload_bytes != actual_bytes:
        errors.append(
            f"total_payload_bytes mismatch: metadata={metadata.total_payload_bytes}, "
            f"actual={actual_bytes}"
        )

    if metadata.role is not CacheRole.VERIFIER:
        errors.append("storage metadata role must be VERIFIER for full-KV manager")
    if metadata.materialization is not CacheMaterialization.FULL:
        errors.append("storage metadata materialization must be FULL")

    if not metadata.claim_note.strip():
        errors.append("claim_note required on storage metadata")

    return errors


def cache_view_from_storage_metadata(metadata: KVStorageMetadata) -> CacheView:
    """Derive a conservative verifier ``CacheView`` from storage metadata."""
    return CacheView(
        role=CacheRole.VERIFIER,
        backend_name=metadata.backend_name,
        residency=metadata.residency,
        materialization=CacheMaterialization.FULL,
        kv_bytes=metadata.total_payload_bytes,
        metadata_bytes=metadata.metadata_bytes,
        temporary_workspace_bytes=0,
        supports_real_bytes_claim=False,
        claim_note=metadata.claim_note,
    )


def dual_cache_with_stored_verifier(
    draft: CacheView,
    storage_metadata: KVStorageMetadata,
) -> DualCacheState:
    """Pair a draft view with a stored verifier view derived from metadata."""
    verifier = cache_view_from_storage_metadata(storage_metadata)
    return DualCacheState(
        draft=draft,
        verifier=verifier,
        notes="Verifier backed by storage manager metadata (Phase 11C spike).",
    )


def validate_stored_verifier_dual_cache(
    draft: CacheView,
    storage_metadata: KVStorageMetadata,
    payload: Any,
) -> list[str]:
    """Validate storage metadata, payload, and resulting dual-cache state."""
    errors = validate_storage_metadata(storage_metadata, payload)
    state = dual_cache_with_stored_verifier(draft, storage_metadata)
    errors.extend(validate_dual_cache_state(state))
    return errors


class InMemoryKVStorageBackend(KVStorageBackend):
    """Process-local dict backend (residency metadata typically CPU)."""

    def __init__(self, *, backend_name: str = "in_memory_kv_storage") -> None:
        self._backend_name = backend_name
        self._entries: dict[tuple[str, str, str], StoredKVEntry] = {}

    def _key(self, handle: KVStorageHandle) -> tuple[str, str, str]:
        return (handle.namespace, handle.key, handle.version)

    def put(self, handle: KVStorageHandle, payload: Any, metadata: KVStorageMetadata) -> None:
        errors = validate_storage_metadata(metadata, payload)
        if errors:
            raise KVStorageError("; ".join(errors))
        self._entries[self._key(handle)] = StoredKVEntry(payload=payload, metadata=metadata)

    def get(self, handle: KVStorageHandle) -> StoredKVEntry:
        try:
            return self._entries[self._key(handle)]
        except KeyError as exc:
            raise KVStorageNotFoundError(
                f"KV storage handle not found: {handle.namespace}/{handle.key}@{handle.version}"
            ) from exc

    def exists(self, handle: KVStorageHandle) -> bool:
        return self._key(handle) in self._entries

    def delete(self, handle: KVStorageHandle) -> None:
        self._entries.pop(self._key(handle), None)

    def metadata(self, handle: KVStorageHandle) -> KVStorageMetadata:
        return self.get(handle).metadata


class FileKVStorageBackend(KVStorageBackend):
    """Local directory backend using ``torch.save`` / ``torch.load`` (design spike)."""

    def __init__(self, root: Path | str, *, backend_name: str = "file_kv_storage") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._backend_name = backend_name

    def _payload_path(self, handle: KVStorageHandle) -> Path:
        return self.root / f"{handle.filename_stem()}.pt"

    def _meta_path(self, handle: KVStorageHandle) -> Path:
        return self.root / f"{handle.filename_stem()}.meta.json"

    def put(self, handle: KVStorageHandle, payload: Any, metadata: KVStorageMetadata) -> None:
        errors = validate_storage_metadata(metadata, payload)
        if errors:
            raise KVStorageError("; ".join(errors))
        torch.save(payload, self._payload_path(handle))
        meta_path = self._meta_path(handle)
        meta_path.write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def get(self, handle: KVStorageHandle) -> StoredKVEntry:
        if not self.exists(handle):
            raise KVStorageNotFoundError(
                f"KV storage handle not found: {handle.namespace}/{handle.key}@{handle.version}"
            )
        payload = torch.load(self._payload_path(handle), weights_only=False)
        metadata = self.metadata(handle)
        errors = validate_storage_metadata(metadata, payload)
        if errors:
            raise KVStorageError("stored metadata invalid after load: " + "; ".join(errors))
        return StoredKVEntry(payload=payload, metadata=metadata)

    def exists(self, handle: KVStorageHandle) -> bool:
        return self._payload_path(handle).is_file() and self._meta_path(handle).is_file()

    def delete(self, handle: KVStorageHandle) -> None:
        self._payload_path(handle).unlink(missing_ok=True)
        self._meta_path(handle).unlink(missing_ok=True)

    def metadata(self, handle: KVStorageHandle) -> KVStorageMetadata:
        meta_path = self._meta_path(handle)
        if not meta_path.is_file():
            raise KVStorageNotFoundError(
                f"KV storage metadata not found: {handle.namespace}/{handle.key}@{handle.version}"
            )
        return KVStorageMetadata.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))


def smoke_store_verifier_payload(
    backend: KVStorageBackend,
    handle: KVStorageHandle,
    payload: Any,
    *,
    residency: CacheResidency,
) -> StoredKVEntry:
    """Tiny smoke helper: build metadata, put, get round-trip."""
    metadata = build_verifier_storage_metadata(
        payload,
        residency=residency,
        backend_name=getattr(backend, "_backend_name", "full_kv_storage"),
    )
    backend.put(handle, payload, metadata)
    return backend.get(handle)
