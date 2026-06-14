"""Tests for Phase 11C full-KV storage manager design spike."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from exactkv.cache.dual_cache import (
    CacheMaterialization,
    CacheResidency,
    CacheRole,
    CacheView,
    validate_cache_view,
    validate_dual_cache_state,
)
from exactkv.cache.storage import (
    FileKVStorageBackend,
    InMemoryKVStorageBackend,
    KVStorageHandle,
    KVStorageNotFoundError,
    build_verifier_storage_metadata,
    cache_view_from_storage_metadata,
    dual_cache_with_stored_verifier,
    smoke_store_verifier_payload,
    summarize_tensor_payload,
    validate_storage_metadata,
    validate_stored_verifier_dual_cache,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "FULL_KV_STORAGE_MANAGER.md"


def _handle(key: str = "smoke") -> KVStorageHandle:
    return KVStorageHandle(namespace="test", key=key, version="1")


def _payload() -> dict[str, torch.Tensor]:
    return {"k": torch.randn(2, 4), "v": torch.randn(2, 4, 8)}


def _draft_view() -> CacheView:
    return CacheView(
        role=CacheRole.DRAFT,
        backend_name="int8",
        residency=CacheResidency.GPU,
        materialization=CacheMaterialization.COMPRESSED,
        kv_bytes=512,
        supports_real_bytes_claim=False,
        claim_note="Draft side; compressed storage spike.",
    )


@pytest.fixture
def mem_backend() -> InMemoryKVStorageBackend:
    return InMemoryKVStorageBackend()


def test_in_memory_put_get_exists_delete(mem_backend: InMemoryKVStorageBackend) -> None:
    handle = _handle("mem")
    payload = _payload()
    meta = build_verifier_storage_metadata(payload, residency=CacheResidency.CPU)
    mem_backend.put(handle, payload, meta)
    assert mem_backend.exists(handle)
    entry = mem_backend.get(handle)
    assert torch.allclose(entry.payload["k"], payload["k"])
    assert entry.metadata.tensor_count == 2
    mem_backend.delete(handle)
    assert not mem_backend.exists(handle)


def test_file_backend_roundtrip(tmp_path: Path) -> None:
    backend = FileKVStorageBackend(tmp_path)
    handle = _handle("file")
    payload = _payload()
    meta = build_verifier_storage_metadata(payload, residency=CacheResidency.DISK)
    backend.put(handle, payload, meta)
    assert backend.exists(handle)
    entry = backend.get(handle)
    assert torch.allclose(entry.payload["v"], payload["v"])
    assert backend.metadata(handle).total_payload_bytes == meta.total_payload_bytes
    backend.delete(handle)
    assert not backend.exists(handle)


def test_tensor_metadata_shape_and_dtype() -> None:
    payload = _payload()
    count, total, dtypes, shapes = summarize_tensor_payload(payload)
    assert count == 2
    assert "float" in dtypes
    assert "(2, 4)" in shapes or "2, 4" in shapes
    assert total > 0


def test_missing_handle_raises_clear_error(mem_backend: InMemoryKVStorageBackend) -> None:
    with pytest.raises(KVStorageNotFoundError, match="not found"):
        mem_backend.get(_handle("missing"))


def test_metadata_serializes() -> None:
    payload = _payload()
    meta = build_verifier_storage_metadata(payload, residency=CacheResidency.CPU)
    raw = meta.to_dict()
    restored = type(meta).from_dict(raw)
    assert restored.tensor_count == meta.tensor_count
    assert restored.total_payload_bytes == meta.total_payload_bytes
    json.dumps(raw, sort_keys=True)


def test_metadata_builds_valid_verifier_cache_view() -> None:
    payload = _payload()
    meta = build_verifier_storage_metadata(payload, residency=CacheResidency.DISK)
    view = cache_view_from_storage_metadata(meta)
    assert view.role is CacheRole.VERIFIER
    assert view.materialization is CacheMaterialization.FULL
    assert view.supports_real_bytes_claim is False
    assert view.claim_note
    assert validate_cache_view(view) == []


def test_dual_cache_with_stored_verifier() -> None:
    payload = _payload()
    meta = build_verifier_storage_metadata(payload, residency=CacheResidency.CPU)
    state = dual_cache_with_stored_verifier(_draft_view(), meta)
    assert validate_dual_cache_state(state) == []
    assert validate_stored_verifier_dual_cache(_draft_view(), meta, payload) == []


def test_smoke_helper_roundtrip(mem_backend: InMemoryKVStorageBackend) -> None:
    entry = smoke_store_verifier_payload(
        mem_backend, _handle("smoke_helper"), _payload(), residency=CacheResidency.CPU
    )
    assert entry.metadata.tensor_count == 2


def test_validate_storage_metadata_tensor_count_mismatch() -> None:
    payload = _payload()
    meta = build_verifier_storage_metadata(payload, residency=CacheResidency.CPU)
    meta.tensor_count = 99
    errors = validate_storage_metadata(meta, payload)
    assert any("tensor_count mismatch" in e for e in errors)


def test_doc_caveats() -> None:
    assert _DOC.is_file()
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "storage contract",
        "not a serving runtime",
        "vllm",
        "lmcache",
        "active gpu memory savings",
        "throughput",
        "generation and verification behavior is unchanged",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    assert "forbidden" in text
    for phrase in ("achieves speedup", "production serving ready", "vericache reproduction is complete"):
        assert phrase not in text


def test_package_exports() -> None:
    from exactkv.cache import FileKVStorageBackend as F
    from exactkv.cache import InMemoryKVStorageBackend as M

    assert M is not None and F is not None
