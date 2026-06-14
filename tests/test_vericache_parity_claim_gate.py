"""Tests for Phase 11K VeriCache parity RC claim gate."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.claims.vericache_parity_gate import (
    VeriCacheClaimCategory,
    VeriCacheClaimStatus,
    VeriCacheParityClaim,
    VeriCacheParityClaimGate,
    build_default_vericache_parity_claim_gate,
    validate_vericache_parity_claim_gate,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "VERICACHE_PARITY_CLAIM_GATE.md"


def _claim(gate: VeriCacheParityClaimGate, category: VeriCacheClaimCategory) -> VeriCacheParityClaim:
    claim = gate.claim_for(category)
    assert claim is not None
    return claim


def test_default_gate_serializes() -> None:
    gate = build_default_vericache_parity_claim_gate()
    raw = gate.to_dict()
    restored = VeriCacheParityClaimGate.from_dict(raw)
    assert restored.full_parity_claim_allowed is False
    assert len(restored.claims) == len(gate.claims)
    json.dumps(raw, sort_keys=True)


def test_default_gate_validates() -> None:
    gate = build_default_vericache_parity_claim_gate()
    assert validate_vericache_parity_claim_gate(gate) == []


def test_algorithmic_semantics_allowed_with_scope() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.ALGORITHMIC_SEMANTICS)
    assert claim.status is VeriCacheClaimStatus.ALLOWED_WITH_SCOPE
    assert claim.allowed_wording


def test_correctness_allowed_with_scope() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.CORRECTNESS_ON_TESTED_PANELS)
    assert claim.status is VeriCacheClaimStatus.ALLOWED_WITH_SCOPE
    text = " ".join(claim.allowed_wording).lower()
    assert "panel" in text or "tested" in text or "cited" in text


def test_full_vericache_reproduction_forbidden() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.FULL_VERICACHE_REPRODUCTION)
    assert claim.status is VeriCacheClaimStatus.FORBIDDEN
    assert claim.forbidden_wording


def test_throughput_benefit_forbidden() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.THROUGHPUT_BENEFIT)
    assert claim.status is VeriCacheClaimStatus.FORBIDDEN
    assert not gate.throughput_claim_allowed


def test_memory_benefit_forbidden() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.MEMORY_BENEFIT)
    assert claim.status is VeriCacheClaimStatus.FORBIDDEN
    assert not gate.memory_claim_allowed


def test_vllm_integration_contract_only() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.VLLM_INTEGRATION)
    assert claim.status is VeriCacheClaimStatus.CONTRACT_ONLY
    assert any("integrated" in w.lower() for w in claim.forbidden_wording)


def test_lmcache_integration_contract_only() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.LMCACHE_INTEGRATION)
    assert claim.status is VeriCacheClaimStatus.CONTRACT_ONLY


def test_remote_prefix_runtime_forbidden() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.REMOTE_PREFIX_CACHE)
    assert claim.status is VeriCacheClaimStatus.FORBIDDEN


def test_paper_like_reproduction_blocked() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.PAPER_LIKE_REPRODUCTION)
    assert claim.status in (
        VeriCacheClaimStatus.BLOCKED_PENDING_EVIDENCE,
        VeriCacheClaimStatus.CONTRACT_ONLY,
    )
    assert not gate.paper_panel_claim_eligible


def test_production_serving_forbidden() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.PRODUCTION_SERVING)
    assert claim.status is VeriCacheClaimStatus.FORBIDDEN
    assert not gate.serving_claim_allowed


def test_full_parity_cannot_be_allowed_without_gates() -> None:
    gate = build_default_vericache_parity_claim_gate()
    gate.full_parity_claim_allowed = True
    claim = _claim(gate, VeriCacheClaimCategory.FULL_VERICACHE_REPRODUCTION)
    claim.status = VeriCacheClaimStatus.ALLOWED
    errors = validate_vericache_parity_claim_gate(gate)
    assert any("full_parity" in e or "FULL_VERICACHE" in e for e in errors)


def test_throughput_cannot_be_allowed_without_flag() -> None:
    gate = build_default_vericache_parity_claim_gate()
    claim = _claim(gate, VeriCacheClaimCategory.THROUGHPUT_BENEFIT)
    claim.status = VeriCacheClaimStatus.ALLOWED
    errors = validate_vericache_parity_claim_gate(gate)
    assert any("THROUGHPUT_BENEFIT" in e or "throughput_claim_allowed" in e for e in errors)


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "vericache-style algorithmic semantics",
        "full vericache serving system",
        "reproduce vericache throughput",
        "reproduce vericache memory",
        "implement vllm",
        "implement production serving",
        "full vericache reproduction remains forbidden",
        "human review",
        "contract completion",
        "paper numbers",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "full vericache parity achieved", "production-scale deployment readiness"):
        assert phrase not in text


def test_package_exports() -> None:
    from exactkv.claims import build_default_vericache_parity_claim_gate as factory

    gate = factory()
    assert gate.full_parity_claim_allowed is False
