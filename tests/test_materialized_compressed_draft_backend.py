"""Tests for Phase 11D materialized compressed-draft backend spike."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from exactkv.cache.dual_cache import (
    CacheMaterialization,
    CacheRole,
    validate_cache_view,
    validate_dual_cache_state,
)
from exactkv.cache.materialized_backend import (
    DraftBackendKind,
    MaterializedDraftMetadata,
    SyntheticMaterializedDraftBackend,
    build_draft_cache_view,
    build_draft_metadata_from_payload,
    compute_dual_cache_footprint,
    dual_cache_with_materialized_draft_and_verifier,
    smoke_materialized_dual_cache,
    validate_materialized_draft_metadata,
    validate_materialized_dual_cache,
)
from exactkv.cache.dual_cache import CacheResidency
from exactkv.cache.storage import (
    FileKVStorageBackend,
    InMemoryKVStorageBackend,
    KVStorageHandle,
    build_verifier_storage_metadata,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "MATERIALIZED_COMPRESSED_DRAFT_BACKEND.md"


def _payload() -> dict[str, torch.Tensor]:
    return {"k": torch.randn(2, 4), "v": torch.randn(2, 4, 8)}


def _handle(key: str) -> KVStorageHandle:
    return KVStorageHandle(namespace="mat", key=key, version="1")


def test_draft_metadata_round_trip() -> None:
    meta = build_draft_metadata_from_payload(
        _payload(),
        backend_name="k8_v4_sim",
        backend_kind=DraftBackendKind.SIMULATED_COMPRESSED,
    )
    restored = MaterializedDraftMetadata.from_dict(meta.to_dict())
    assert restored.backend_kind is DraftBackendKind.SIMULATED_COMPRESSED
    assert restored.stored_bytes == meta.stored_bytes


def test_identity_draft_cache_view_validates() -> None:
    meta = build_draft_metadata_from_payload(
        _payload(), backend_name="noop", backend_kind=DraftBackendKind.IDENTITY
    )
    view = build_draft_cache_view(meta)
    assert view.role is CacheRole.DRAFT
    assert view.materialization is CacheMaterialization.FULL
    assert validate_materialized_draft_metadata(meta) == []
    assert validate_cache_view(view) == []


def test_simulated_requires_caveat_note() -> None:
    meta = build_draft_metadata_from_payload(
        _payload(),
        backend_name="k8_v4_sim",
        backend_kind=DraftBackendKind.SIMULATED_COMPRESSED,
        claim_note="no compressed keyword here",
    )
    errors = validate_materialized_draft_metadata(meta)
    assert any("simulated" in e.lower() or "caveat" in e.lower() for e in errors)


def test_external_adapter_requires_materializing_caveat() -> None:
    meta = build_draft_metadata_from_payload(
        _payload(),
        backend_name="spectralquant_experimental",
        backend_kind=DraftBackendKind.EXTERNAL_ADAPTER_MATERIALIZED,
        claim_note="external adapter only",
    )
    errors = validate_materialized_draft_metadata(meta)
    assert any("materializ" in e for e in errors)


def test_negative_accounting_fails() -> None:
    meta = build_draft_metadata_from_payload(
        _payload(), backend_name="x", backend_kind=DraftBackendKind.SIMULATED_COMPRESSED
    )
    meta.stored_bytes = -1
    assert any("stored_bytes" in e for e in validate_materialized_draft_metadata(meta))


def test_materialized_cannot_claim_active_gpu_savings() -> None:
    meta = build_draft_metadata_from_payload(
        _payload(),
        backend_name="sq",
        backend_kind=DraftBackendKind.EXTERNAL_ADAPTER_MATERIALIZED,
    )
    meta.supports_real_bytes_claim = True
    errors = validate_materialized_draft_metadata(meta)
    assert any("supports_real_bytes_claim" in e for e in errors)
    view = build_draft_cache_view(meta)
    assert view.supports_real_bytes_claim is False


def test_in_memory_verifier_plus_materialized_draft() -> None:
    backend = InMemoryKVStorageBackend()
    state, footprint, _ = smoke_materialized_dual_cache(
        backend,
        _handle("mem"),
        _payload(),
        DraftBackendKind.SIMULATED_COMPRESSED,
        verifier_residency=CacheResidency.CPU,
    )
    assert validate_dual_cache_state(state) == []
    assert footprint.combined_total > 0
    assert footprint.validate() == []


def test_file_verifier_plus_materialized_draft(tmp_path: Path) -> None:
    backend = FileKVStorageBackend(tmp_path)
    state, footprint, _ = smoke_materialized_dual_cache(
        backend,
        _handle("file"),
        _payload(),
        DraftBackendKind.EXTERNAL_ADAPTER_MATERIALIZED,
        verifier_residency=CacheResidency.DISK,
    )
    assert validate_dual_cache_state(state) == []
    assert footprint.draft_total + footprint.verifier_total == footprint.combined_total


def test_footprint_reconciles() -> None:
    payload = _payload()
    draft = build_draft_metadata_from_payload(
        payload, backend_name="int8", backend_kind=DraftBackendKind.REAL_COMPRESSED_MATERIALIZED
    )
    verifier = build_verifier_storage_metadata(payload, residency=CacheResidency.CPU)
    fp = compute_dual_cache_footprint(draft, verifier)
    assert fp.draft_stored_bytes == draft.stored_bytes
    assert fp.verifier_payload_bytes == verifier.total_payload_bytes
    assert fp.combined_total >= fp.draft_total
    assert validate_materialized_dual_cache(draft, verifier, payload) == []


def test_all_three_draft_kinds_in_smoke() -> None:
    backend = InMemoryKVStorageBackend()
    for i, kind in enumerate(
        (
            DraftBackendKind.IDENTITY,
            DraftBackendKind.SIMULATED_COMPRESSED,
            DraftBackendKind.EXTERNAL_ADAPTER_MATERIALIZED,
        )
    ):
        state, _, _ = smoke_materialized_dual_cache(
            backend,
            _handle(f"kind{i}"),
            _payload(),
            kind,
            verifier_residency=CacheResidency.CPU,
        )
        assert state.draft.role is CacheRole.DRAFT
        assert state.verifier.role is CacheRole.VERIFIER


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "isolated contract",
        "not a serving runtime",
        "generation and verification behavior is unchanged",
        "active gpu memory savings",
        "vllm",
        "lmcache",
        "throughput",
        "spectralquant",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    assert "forbidden" in text
    for phrase in ("achieves speedup", "production serving ready", "vericache reproduction is complete"):
        assert phrase not in text


def test_package_exports() -> None:
    from exactkv.cache import DraftBackendKind as K
    from exactkv.cache import smoke_materialized_dual_cache as sm

    assert K is not None and sm is not None
