"""Tests for Phase 11G LMCache prototype path contracts."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.integrations.lmcache_contract import (
    LMCacheIntegrationStatus,
    LMCachePrototypeGate,
    LMCachePrototypePlan,
    assert_lmcache_not_required,
    build_default_lmcache_prototype_plan,
    validate_lmcache_prototype_plan,
)
from exactkv.integrations.vllm_contract import VLLMIntegrationStatus

_DOC = Path(__file__).resolve().parents[1] / "docs" / "LMCACHE_PROTOTYPE_PATH.md"
_ROADMAP = Path(__file__).resolve().parents[1] / "docs" / "VERICACHE_SYSTEMS_ROADMAP.md"


def test_default_plan_serializes() -> None:
    plan = build_default_lmcache_prototype_plan()
    raw = plan.to_dict()
    restored = LMCachePrototypePlan.from_dict(raw)
    assert restored.status is LMCacheIntegrationStatus.CONTRACT_ONLY
    assert restored == plan
    json.dumps(raw, sort_keys=True)


def test_default_plan_validates() -> None:
    plan = build_default_lmcache_prototype_plan()
    assert validate_lmcache_prototype_plan(plan) == []


def test_unsatisfied_gates_block_prototype_ready() -> None:
    plan = build_default_lmcache_prototype_plan()
    plan.status = LMCacheIntegrationStatus.PROTOTYPE_READY
    errors = validate_lmcache_prototype_plan(plan)
    assert any("PROTOTYPE_READY" in e for e in errors)
    assert any("rollback_fallback_path" in e for e in errors)


def test_experimental_active_fails_validation() -> None:
    plan = build_default_lmcache_prototype_plan()
    plan.status = LMCacheIntegrationStatus.EXPERIMENTAL_ACTIVE
    errors = validate_lmcache_prototype_plan(plan)
    assert any("EXPERIMENTAL_ACTIVE" in e for e in errors)


def test_forbidden_claims_present() -> None:
    plan = build_default_lmcache_prototype_plan()
    forbidden = {c.lower() for c in plan.forbidden_claims}
    for term in (
        "speedup",
        "throughput improvement",
        "memory savings",
        "remote prefix caching",
        "production serving",
    ):
        assert term in forbidden


def test_remote_prefix_cannot_be_active() -> None:
    plan = build_default_lmcache_prototype_plan()
    assert plan.remote_prefix_cache_active is False
    plan.remote_prefix_cache_active = True
    errors = validate_lmcache_prototype_plan(plan)
    assert any("remote_prefix_cache_active" in e for e in errors)


def test_no_lmcache_import_required() -> None:
    assert_lmcache_not_required()
    import exactkv.integrations.lmcache_contract as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import lmcache" not in source
    assert "from lmcache" not in source


def test_vllm_referenced_as_contract_only() -> None:
    plan = build_default_lmcache_prototype_plan()
    assert plan.vllm_contract_status == VLLMIntegrationStatus.CONTRACT_ONLY.value
    gate = next(g for g in plan.gates if g.gate_name == "vllm_contract_interaction_identified")
    assert gate.satisfied
    assert "contract" in gate.evidence.lower() or "CONTRACT_ONLY" in gate.evidence


def test_vllm_active_status_blocks_lmcache_plan() -> None:
    plan = build_default_lmcache_prototype_plan()
    plan.vllm_contract_status = VLLMIntegrationStatus.EXPERIMENTAL_ACTIVE.value
    errors = validate_lmcache_prototype_plan(plan)
    assert any("vllm_contract_status" in e for e in errors)


def test_dependency_import_attempted_fails() -> None:
    plan = build_default_lmcache_prototype_plan()
    plan.dependency_import_attempted = True
    errors = validate_lmcache_prototype_plan(plan)
    assert any("dependency_import_attempted" in e for e in errors)


def test_gate_serializes() -> None:
    gate = LMCachePrototypeGate(
        gate_name="test_gate",
        required=True,
        satisfied=False,
        blocker="example blocker",
    )
    restored = LMCachePrototypeGate.from_dict(gate.to_dict())
    assert restored == gate


def test_backward_compatible_missing_fields() -> None:
    minimal = {"status": "CONTRACT_ONLY"}
    plan = LMCachePrototypePlan.from_dict(minimal)
    assert plan.dependency_import_attempted is False
    assert plan.remote_prefix_cache_active is False
    assert plan.gates == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "lmcache prototype contract",
        "not an lmcache integration",
        "lmcache is not imported",
        "remote prefix caching",
        "production serving",
        "throughput",
        "exactness",
        "memory",
        "vllm",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "lmcache integrated today", "remote prefix cache active"):
        assert phrase not in text


def test_roadmap_stage6_contract_only() -> None:
    text = _ROADMAP.read_text(encoding="utf-8").lower()
    assert "stage 6" in text
    assert "11g" in text or "contract" in text
    assert "not integrated" in text or "contract-only" in text or "contract only" in text


def test_package_exports() -> None:
    from exactkv.integrations import build_default_lmcache_prototype_plan as factory

    plan = factory()
    assert plan.status is LMCacheIntegrationStatus.CONTRACT_ONLY
