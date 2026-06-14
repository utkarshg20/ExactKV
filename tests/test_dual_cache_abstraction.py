"""Tests for Phase 11B dual-cache abstraction contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "DUAL_CACHE_ABSTRACTION.md"


def _draft_view(**kwargs) -> CacheView:
    defaults = {
        "role": CacheRole.DRAFT,
        "backend_name": "k8_v4_sim",
        "residency": CacheResidency.GPU,
        "materialization": CacheMaterialization.SIMULATED,
        "kv_bytes": 1000,
        "metadata_bytes": 64,
        "temporary_workspace_bytes": 0,
        "supports_real_bytes_claim": False,
        "claim_note": "Simulated compressor; no real-byte claim.",
    }
    defaults.update(kwargs)
    return CacheView(**defaults)


def _verifier_view(**kwargs) -> CacheView:
    defaults = {
        "role": CacheRole.VERIFIER,
        "backend_name": "full_kv",
        "residency": CacheResidency.GPU,
        "materialization": CacheMaterialization.FULL,
        "kv_bytes": 4000,
        "metadata_bytes": 0,
        "temporary_workspace_bytes": 0,
        "supports_real_bytes_claim": True,
        "claim_note": "Authoritative full KV on GPU.",
    }
    defaults.update(kwargs)
    return CacheView(**defaults)


def test_cache_view_round_trip() -> None:
    view = _draft_view()
    restored = CacheView.from_dict(view.to_dict())
    assert restored == view


def test_dual_cache_state_round_trip() -> None:
    state = DualCacheState(draft=_draft_view(), verifier=_verifier_view(), notes="test")
    raw = state.to_dict()
    restored = DualCacheState.from_dict(raw)
    assert restored.draft == state.draft
    assert restored.verifier == state.verifier
    assert restored.schema_version == "1.0"
    # JSON-stable
    json.dumps(raw)


def test_draft_verifier_role_invariants() -> None:
    bad = DualCacheState(
        draft=_draft_view(role=CacheRole.VERIFIER),
        verifier=_verifier_view(),
    )
    errors = validate_dual_cache_state(bad)
    assert any("draft view must have role DRAFT" in e for e in errors)

    bad2 = DualCacheState(
        draft=_draft_view(),
        verifier=_verifier_view(role=CacheRole.DRAFT),
    )
    errors2 = validate_dual_cache_state(bad2)
    assert any("verifier view must have role VERIFIER" in e for e in errors2)


def test_non_negative_accounting() -> None:
    view = _draft_view(kv_bytes=-1)
    assert any("kv_bytes" in e for e in validate_cache_view(view))


def test_simulated_requires_caveat_note() -> None:
    view = _draft_view(supports_real_bytes_claim=False, claim_note="")
    assert any("claim_note required" in e for e in validate_cache_view(view))


def test_materializing_cannot_claim_active_gpu_savings() -> None:
    view = _draft_view(
        materialization=CacheMaterialization.MATERIALIZED,
        supports_real_bytes_claim=True,
        claim_note="materializing adapter",
    )
    errors = validate_cache_view(view)
    assert any("materialized/simulated" in e for e in errors)


def test_real_byte_claim_passes_with_accounting() -> None:
    state = DualCacheState(
        draft=_draft_view(
            materialization=CacheMaterialization.COMPRESSED,
            supports_real_bytes_claim=True,
            claim_note="INT8 stored bytes reflect real quantised storage.",
            kv_bytes=2000,
        ),
        verifier=_verifier_view(kv_bytes=4000),
    )
    assert validate_dual_cache_state(state) == []


def test_verifier_must_be_full_or_unknown() -> None:
    view = _verifier_view(materialization=CacheMaterialization.COMPRESSED)
    errors = validate_cache_view(view)
    assert any("verifier cache must be FULL or UNKNOWN" in e for e in errors)


def test_backward_compatible_missing_optional_fields() -> None:
    minimal = {
        "role": "DRAFT",
        "backend_name": "legacy",
    }
    view = CacheView.from_dict(minimal)
    assert view.residency is CacheResidency.UNKNOWN
    assert view.materialization is CacheMaterialization.UNKNOWN
    assert view.kv_bytes == 0
    assert view.supports_real_bytes_claim is False


def test_identity_dual_cache_valid() -> None:
    state = build_identity_dual_cache(kv_bytes=8192)
    assert validate_dual_cache_state(state) == []


def test_build_identity_import_from_package() -> None:
    from exactkv.cache import build_identity_dual_cache as bif

    state = bif(kv_bytes=1)
    assert state.validate() == []


def test_doc_exists_with_required_caveats() -> None:
    assert _DOC.is_file()
    text = _DOC.read_text(encoding="utf-8").lower()
    required = [
        "contract layer",
        "not a serving runtime",
        "vllm",
        "lmcache",
        "active gpu memory savings",
        "throughput",
        "generation and verification behavior is unchanged",
        "stage 2",
    ]
    for phrase in required:
        assert phrase in text, f"missing doc phrase: {phrase}"


def test_docs_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    # Forbidden column in claims table is allowed; positive promotion is not.
    assert "forbidden" in text
    positive_promotion = [
        "achieves speedup",
        "production serving ready",
        "implements vllm integration",
        "implements lmcache integration",
        "vericache reproduction is complete",
    ]
    for phrase in positive_promotion:
        assert phrase not in text


def test_materializing_draft_smaller_than_verifier_with_real_claim_fails() -> None:
    state = DualCacheState(
        draft=_draft_view(
            materialization=CacheMaterialization.MATERIALIZED,
            supports_real_bytes_claim=True,
            claim_note="should fail validation",
            kv_bytes=100,
        ),
        verifier=_verifier_view(kv_bytes=5000),
    )
    errors = validate_dual_cache_state(state)
    assert errors  # materialized + real claim triggers view-level error
