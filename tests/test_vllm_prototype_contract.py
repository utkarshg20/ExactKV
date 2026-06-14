"""Tests for Phase 11F vLLM prototype path contracts."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.integrations.vllm_contract import (
    VLLMIntegrationStatus,
    VLLMPrototypeGate,
    VLLMPrototypePlan,
    assert_vllm_not_required,
    build_default_vllm_prototype_plan,
    validate_vllm_prototype_plan,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "VLLM_PROTOTYPE_PATH.md"
_ROADMAP = Path(__file__).resolve().parents[1] / "docs" / "VERICACHE_SYSTEMS_ROADMAP.md"


def test_default_plan_serializes() -> None:
    plan = build_default_vllm_prototype_plan()
    raw = plan.to_dict()
    restored = VLLMPrototypePlan.from_dict(raw)
    assert restored.status is VLLMIntegrationStatus.CONTRACT_ONLY
    assert restored == plan
    json.dumps(raw, sort_keys=True)


def test_default_plan_validates() -> None:
    plan = build_default_vllm_prototype_plan()
    assert validate_vllm_prototype_plan(plan) == []


def test_unsatisfied_gates_block_prototype_ready() -> None:
    plan = build_default_vllm_prototype_plan()
    plan.status = VLLMIntegrationStatus.PROTOTYPE_READY
    errors = validate_vllm_prototype_plan(plan)
    assert any("PROTOTYPE_READY" in e for e in errors)
    assert any("rollback_fallback_path" in e for e in errors)


def test_experimental_active_fails_validation() -> None:
    plan = build_default_vllm_prototype_plan()
    plan.status = VLLMIntegrationStatus.EXPERIMENTAL_ACTIVE
    errors = validate_vllm_prototype_plan(plan)
    assert any("EXPERIMENTAL_ACTIVE" in e for e in errors)


def test_forbidden_claims_present() -> None:
    plan = build_default_vllm_prototype_plan()
    forbidden = {c.lower() for c in plan.forbidden_claims}
    for term in ("speedup", "throughput improvement", "memory savings", "production serving"):
        assert term in forbidden


def test_no_vllm_import_required() -> None:
    assert_vllm_not_required()
    import exactkv.integrations.vllm_contract as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import vllm" not in source
    assert "from vllm" not in source


def test_dependency_import_attempted_fails() -> None:
    plan = build_default_vllm_prototype_plan()
    plan.dependency_import_attempted = True
    errors = validate_vllm_prototype_plan(plan)
    assert any("dependency_import_attempted" in e for e in errors)


def test_allowed_claims_cannot_encode_forbidden() -> None:
    plan = build_default_vllm_prototype_plan()
    plan.allowed_claims = ["achieves speedup on vLLM path"]
    errors = validate_vllm_prototype_plan(plan)
    assert any("allowed_claims" in e and "speedup" in e for e in errors)


def test_gate_serializes() -> None:
    gate = VLLMPrototypeGate(
        gate_name="test_gate",
        required=True,
        satisfied=False,
        blocker="example blocker",
    )
    restored = VLLMPrototypeGate.from_dict(gate.to_dict())
    assert restored == gate


def test_backward_compatible_missing_fields() -> None:
    minimal = {"status": "CONTRACT_ONLY"}
    plan = VLLMPrototypePlan.from_dict(minimal)
    assert plan.dependency_import_attempted is False
    assert plan.gates == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "vllm prototype contract",
        "not a vllm integration",
        "vllm is not imported",
        "production serving",
        "throughput",
        "exactness",
        "memory",
        "placeholder",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "vllm integrated today", "latency improvement claim"):
        assert phrase not in text


def test_roadmap_stage5_contract_only() -> None:
    text = _ROADMAP.read_text(encoding="utf-8").lower()
    assert "stage 5" in text
    assert "11f" in text or "contract" in text
    assert "not integrated" in text or "contract-only" in text or "contract only" in text


def test_package_exports() -> None:
    from exactkv.integrations import build_default_vllm_prototype_plan as factory

    plan = factory()
    assert plan.status is VLLMIntegrationStatus.CONTRACT_ONLY
