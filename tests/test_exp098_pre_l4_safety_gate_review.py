"""Tests for Experiment 098 pre-L4 safety gate review (Phase 20A)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.safety.pre_l4_gate_review import (
    EXPERIMENT_098_ID,
    FORBIDDEN_NEXT_PHASES,
    GATE_NAMES,
    L4_IMPLEMENTATION_BLOCKERS,
    OUTCOME_BLOCKED_MISSING_EVIDENCE,
    OUTCOME_BLOCKED_SAFETY_FAILURE,
    OUTCOME_FORBIDDEN_L4_IMPLEMENTATION,
    OUTCOME_NOT_READY_L4_DESIGN_SPEC,
    OUTCOME_READY_L4_DESIGN_SPEC_ONLY,
    RECOMMENDED_NEXT_PHASE_20A,
    evaluate_pre_l4_gates,
    inventory_evidence,
    run_exp098_pre_l4_safety_gate_review,
    synthetic_exp097_evidence,
    validate_exp098_report,
)

FORBIDDEN_CLAIM_PHRASES = (
    "speedup achieved",
    "throughput improved",
    "latency reduced",
    "tokens_per_second",
    "runtime_seconds",
    "active_gpu_memory_savings",
    "production_memory_savings",
    "production serving supported",
    "VeriCache throughput reproduced",
    "VeriCache serving reproduced",
    "streaming attention integrated into token commit",
    "draft shadow used for token commit",
    "verifier-mediated compressed draft implemented",
)


def _setup_docs(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs/CLAIMS_AUDIT.md").write_text("# claims audit\n")


def _good_overrides() -> dict:
    return {
        "phase_19c_promoted_source_validation": synthetic_exp097_evidence(),
        "phase_18a_integration_safety_spec": {
            "safety_spec_validation": {
                "pass": True,
                "fallback_to_baseline": True,
            },
        },
    }


def test_gate_schema_validates() -> None:
    assert len(GATE_NAMES) == 10
    for name in GATE_NAMES:
        assert name.endswith("_gate")


def test_evidence_missing_handled_without_invention(tmp_path: Path) -> None:
    report = run_exp098_pre_l4_safety_gate_review(root=tmp_path)
    assert report["review_outcome"] == OUTCOME_BLOCKED_MISSING_EVIDENCE
    assert "phase_19c_promoted_source_validation" in report["evidence_missing"]
    assert report["l4_design_spec_authorized"] is False


def test_all_required_gates_pass_with_synthetic_evidence(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert report["gate_summary"]["prerequisite_gates_pass"] is True
    assert report["gate_summary"]["all_gates_pass"] is True


def test_missing_evidence_gives_blocked_missing_evidence(tmp_path: Path) -> None:
    report = run_exp098_pre_l4_safety_gate_review(root=tmp_path)
    assert report["review_outcome"] == OUTCOME_BLOCKED_MISSING_EVIDENCE


def test_safety_failure_gives_blocked_safety_failure(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    bad = synthetic_exp097_evidence(
        proposal_used_for_token_commit=True,
        source_viability_gate_summary={
            "proposal_coverage_gate": {"pass": True},
            "proposal_provenance_gate": {"pass": True},
            "proposal_isolation_gate": {"pass": False},
            "generation_parity_gate": {"pass": True},
            "exactkv_failure_gate": {"pass": True},
            "claim_boundary_gate": {"pass": True},
        },
    )
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides={
            "phase_19c_promoted_source_validation": bad,
            "phase_18a_integration_safety_spec": {
                "safety_spec_validation": {"pass": True, "fallback_to_baseline": True},
            },
        },
    )
    assert report["review_outcome"] == OUTCOME_BLOCKED_SAFETY_FAILURE


def test_review_outcome_can_be_ready_for_l4_design_spec_only(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert report["review_outcome"] == OUTCOME_READY_L4_DESIGN_SPEC_ONLY
    assert report["l4_design_spec_authorized"] is True


def test_review_outcome_cannot_be_ready_for_l4_implementation() -> None:
    assert OUTCOME_FORBIDDEN_L4_IMPLEMENTATION not in (
        "ready_for_l4_design_spec_only",
        "not_ready_for_l4_design_spec",
        "blocked_missing_evidence",
        "blocked_safety_failure",
    )


def test_l4_design_spec_authorized_can_be_true(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert report["l4_design_spec_authorized"] is True
    assert report["l4_design_spec_may_be_started"] is True


def test_l4_implementation_authorized_must_remain_false(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert report["l4_implementation_authorized"] is False
    assert report["l4_implementation_is_not_authorized"] is True


def test_implementation_blockers_are_non_empty(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert len(report["l4_implementation_blockers"]) == len(L4_IMPLEMENTATION_BLOCKERS)
    assert len(report["l4_implementation_blockers"]) > 0


def test_recommended_next_phase_is_phase20b(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_20A
    assert report["recommended_next_phase"] == "phase20b_l4_verifier_mediated_design_spec"


def test_forbidden_next_phases(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert set(report["forbidden_next_phases"]) == set(FORBIDDEN_NEXT_PHASES)
    assert "l4_implementation" in report["forbidden_next_phases"]
    assert "vllm_integration" in report["forbidden_next_phases"]


def test_report_schema_validates(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    assert report["experiment_id"] == EXPERIMENT_098_ID
    assert validate_exp098_report(report) == []


def test_no_forbidden_positive_claims_in_report(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "review_outcome_reason": report.get("review_outcome_reason"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_inventory_does_not_invent_missing_report_data(tmp_path: Path) -> None:
    inventory, found, missing = inventory_evidence(root=tmp_path)
    exp097 = next((e for e in inventory if e["id"] == "phase_19c_promoted_source_validation"), None)
    assert exp097 is None or exp097.get("report_data") is None
    assert "phase_19c_promoted_source_validation" in missing


def test_gate_results_include_all_fields(tmp_path: Path) -> None:
    _setup_docs(tmp_path)
    inventory, _, _ = inventory_evidence(
        root=tmp_path,
        evidence_overrides=_good_overrides(),
    )
    gates, _ = evaluate_pre_l4_gates(inventory)
    assert len(gates) == len(GATE_NAMES)
    for gate in gates:
        assert gate["result"] in ("pass", "fail")
        assert gate["evidence_status"]


@pytest.mark.parametrize("field", ("proposal_provenance_gate", "generation_parity_gate"))
def test_individual_gate_fails_when_viability_fails(
    tmp_path: Path,
    field: str,
) -> None:
    _setup_docs(tmp_path)
    viability = {
        "proposal_coverage_gate": {"pass": True},
        "proposal_provenance_gate": {"pass": True},
        "proposal_isolation_gate": {"pass": True},
        "generation_parity_gate": {"pass": True},
        "exactkv_failure_gate": {"pass": True},
        "claim_boundary_gate": {"pass": True},
    }
    viability[field] = {"pass": False}
    report = run_exp098_pre_l4_safety_gate_review(
        root=tmp_path,
        evidence_overrides={
            "phase_19c_promoted_source_validation": synthetic_exp097_evidence(
                source_viability_gate_summary=viability,
            ),
            "phase_18a_integration_safety_spec": {
                "safety_spec_validation": {"pass": True, "fallback_to_baseline": True},
            },
        },
    )
    assert report["review_outcome"] == OUTCOME_BLOCKED_SAFETY_FAILURE


def test_real_repo_review_when_exp097_present() -> None:
    report = run_exp098_pre_l4_safety_gate_review(root=Path("."))
    assert validate_exp098_report(report) == []
    if Path("reports/experiment_097_l3_promoted_source_validation.json").is_file():
        assert report["review_outcome"] in (
            OUTCOME_READY_L4_DESIGN_SPEC_ONLY,
            OUTCOME_NOT_READY_L4_DESIGN_SPEC,
            OUTCOME_BLOCKED_SAFETY_FAILURE,
        )
    assert report["l4_implementation_authorized"] is False
