"""Tests for Experiment 099 L4 verifier-mediated design spec (Phase 20B)."""
from __future__ import annotations

import json

from exactkv.safety.l4_verifier_mediated_design_spec import (
    DESIGN_OUTCOME_COMPLETE,
    EXPERIMENT_099_ID,
    FORBIDDEN_DESIGN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASE_20B,
    L4_OPT_IN_FLAG,
    L4_READINESS_GATE_NAMES,
    RECOMMENDED_NEXT_PHASE_20B,
    build_l4_verifier_mediated_design_spec,
    evaluate_l4_design_review,
    run_exp099_l4_verifier_mediated_design_spec,
    validate_exp099_report,
)

FORBIDDEN_CLAIM_PHRASES = FORBIDDEN_DESIGN_CLAIM_PHRASES


def test_design_spec_schema_validates() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    assert report["experiment_id"] == EXPERIMENT_099_ID
    assert validate_exp099_report(report) == []


def test_intended_flow_includes_full_verifier_source_of_truth() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    flow = " ".join(report["intended_l4_flow"]).lower()
    assert "verifier" in flow
    assert "full-kv verifier" in flow or "full verifier" in flow or "verifier evaluates" in flow


def test_intended_flow_includes_rollback_on_mismatch() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    flow = " ".join(report["intended_l4_flow"]).lower()
    assert "rollback" in flow
    assert "mismatch" in flow


def test_draft_proposal_contract_forbids_direct_commit() -> None:
    contracts = run_exp099_l4_verifier_mediated_design_spec()["mandatory_contracts"]
    dpc = contracts["draft_proposal_contract"]
    assert dpc["proposal_source_must_not_commit_directly"] is True
    assert "committed_tokens" in dpc["forbidden_proposal_sources"]


def test_verifier_contract_forbids_bypass() -> None:
    contracts = run_exp099_l4_verifier_mediated_design_spec()["mandatory_contracts"]
    assert contracts["full_verifier_contract"]["verifier_cannot_be_bypassed"] is True


def test_acceptance_contract_accepts_only_verified_matching_prefix() -> None:
    contracts = run_exp099_l4_verifier_mediated_design_spec()["mandatory_contracts"]
    assert contracts["acceptance_contract"]["only_longest_verified_matching_prefix"] is True


def test_rollback_contract_includes_required_cases() -> None:
    rc = run_exp099_l4_verifier_mediated_design_spec()["mandatory_contracts"][
        "rollback_contract"
    ]
    assert rc["rollback_on_verifier_mismatch"] is True
    assert rc["rollback_on_proposal_exception"] is True
    assert rc["rollback_on_missing_verifier_evidence"] is True
    assert rc["rollback_on_safety_gate_failure"] is True


def test_fallback_contract_preserves_default_runtime() -> None:
    fc = run_exp099_l4_verifier_mediated_design_spec()["mandatory_contracts"][
        "fallback_contract"
    ]
    assert fc["default_runtime_must_be_unchanged"] is True
    assert fc["opt_out_path_equals_existing_behavior"] is True


def test_opt_in_contract_disabled_by_default() -> None:
    oc = run_exp099_l4_verifier_mediated_design_spec()["mandatory_contracts"]["opt_in_contract"]
    assert oc["l4_disabled_by_default"] is True
    assert oc["proposed_opt_in_flag"] == L4_OPT_IN_FLAG
    assert oc["flag_implemented"] is False


def test_integration_points_listed_without_modifying_runtime() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    paths = {p["path"] for p in report["integration_points"]}
    assert "exactkv/runtime/exactkv_generator.py" in paths
    assert report["exactkv_generator_modified"] is False
    assert report["default_runtime_changed"] is False


def test_test_matrix_includes_unit_synthetic_model_tests() -> None:
    matrix = run_exp099_l4_verifier_mediated_design_spec()["l4_test_matrix"]
    assert matrix["unit_tests"]
    assert matrix["synthetic_integration_tests"]
    assert matrix["model_tests"]
    assert "performance_benchmark" in matrix["forbidden_tests_for_design_phase"]


def test_readiness_gates_include_required_gates() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    names = {g["name"] for g in report["l4_readiness_gates"]}
    assert names == set(L4_READINESS_GATE_NAMES)


def test_design_review_result_is_l4_design_spec_complete() -> None:
    spec = build_l4_verifier_mediated_design_spec()
    review = evaluate_l4_design_review(spec)
    assert review.outcome == DESIGN_OUTCOME_COMPLETE
    assert review.l4_design_spec_complete is True


def test_l4_implementation_authorized_is_false() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    assert report["l4_implementation_authorized"] is False
    review = report["design_review_result"]
    assert review["l4_implementation_authorized"] is False


def test_allowed_next_phase_is_phase20c_contract_tests() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_20B
    assert report["allowed_next_phase"] == "phase20c_l4_contract_tests_no_runtime"


def test_forbidden_next_phase_is_runtime_implementation() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    assert report["forbidden_next_phase"] == FORBIDDEN_NEXT_PHASE_20B
    assert report["forbidden_next_phase"] == "phase20c_l4_runtime_implementation"


def test_implementation_blockers_remaining_non_empty() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    assert len(report["implementation_blockers_remaining"]) > 0


def test_no_forbidden_positive_claims_in_report() -> None:
    report = run_exp099_l4_verifier_mediated_design_spec()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_design_spec_objects_are_immutable_dataclasses() -> None:
    spec = build_l4_verifier_mediated_design_spec()
    d = spec.to_dict()
    assert d["safety_level"] == "L4_VERIFIER_MEDIATED_COMPRESSED_DRAFT"
    assert d["draft_proposal_contract"]["promoted_l3_source"] == (
        "exactkv_round_log_draft_tokens"
    )
