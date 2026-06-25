"""Tests for Experiment 112 L4 Stage 3 verifier-mediated dry-run design (Phase 21K)."""
from __future__ import annotations

import json

from exactkv.safety.l4_stage3_verifier_mediated_dry_run_design import (
    DESIGN_OUTCOME_COMPLETE,
    EXPERIMENT_112_ID,
    FAILURE_MODE_IDS,
    FAILURE_RESPONSE,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_21K,
    RECOMMENDED_NEXT_PHASE_21K,
    SAFETY_INVARIANT_IDS,
    SYNTHETIC_TEST_CASE_IDS,
    TERMINAL_STATES,
    build_l4_decision_graph_model,
    build_l4_stage3_failure_modes,
    build_l4_stage3_safety_invariants,
    build_l4_stage3_synthetic_test_matrix,
    build_l4_stage3_verifier_mediated_dry_run_design,
    run_exp112_l4_stage3_verifier_mediated_dry_run_design,
    validate_exp112_report,
)


def test_design_outcome_complete() -> None:
    design = build_l4_stage3_verifier_mediated_dry_run_design()
    assert design.decision.outcome == DESIGN_OUTCOME_COMPLETE


def test_decision_graph_terminal_states() -> None:
    graph = build_l4_decision_graph_model()
    assert set(TERMINAL_STATES) <= set(graph.terminal_states)


def test_safety_invariants_all_true() -> None:
    invariants = build_l4_stage3_safety_invariants()
    assert set(SAFETY_INVARIANT_IDS) <= {i.invariant_id for i in invariants}
    assert all(i.required_value for i in invariants)


def test_failure_modes_map_to_block_dry_run() -> None:
    failures = build_l4_stage3_failure_modes()
    assert set(FAILURE_MODE_IDS) <= {f.failure_id for f in failures}
    assert all(f.required_response == FAILURE_RESPONSE for f in failures)


def test_synthetic_test_matrix_covers_cases() -> None:
    tests = build_l4_stage3_synthetic_test_matrix()
    assert set(SYNTHETIC_TEST_CASE_IDS) <= {t.test_id for t in tests}
    assert all(not t.executes_at_runtime for t in tests)


def test_full_match_expected_accept_prefix() -> None:
    tests = build_l4_stage3_synthetic_test_matrix()
    case = next(t for t in tests if t.test_id == "synthetic_full_match_trace")
    assert case.expected_terminal_state == "ACCEPT_PREFIX"


def test_missing_verifier_expected_block() -> None:
    tests = build_l4_stage3_synthetic_test_matrix()
    case = next(t for t in tests if t.test_id == "synthetic_missing_verifier_evidence")
    assert case.expected_terminal_state == "BLOCK_MISSING_EVIDENCE"


def test_aliasing_expected_invalid_trace() -> None:
    tests = build_l4_stage3_synthetic_test_matrix()
    case = next(t for t in tests if t.test_id == "synthetic_adversarial_aliasing")
    assert case.expected_terminal_state == "INVALID_TRACE"


def test_report_schema_validates() -> None:
    report = run_exp112_l4_stage3_verifier_mediated_dry_run_design()
    assert report["experiment_id"] == EXPERIMENT_112_ID
    assert validate_exp112_report(report) == []


def test_runtime_execution_not_authorized() -> None:
    report = run_exp112_l4_stage3_verifier_mediated_dry_run_design()
    assert report["runtime_execution_authorized"] is False
    assert report["runtime_commit_authorized"] is False


def test_stage_3_scaffold_authorized() -> None:
    decision = run_exp112_l4_stage3_verifier_mediated_dry_run_design()["design_decision"]
    assert decision["stage_3_scaffold_authorized"] is True


def test_exactkv_generator_modified_false() -> None:
    assert run_exp112_l4_stage3_verifier_mediated_dry_run_design()["exactkv_generator_modified"] is False


def test_model_experiments_run_false() -> None:
    assert run_exp112_l4_stage3_verifier_mediated_dry_run_design()["model_experiments_run"] is False


def test_allowed_next_phase() -> None:
    report = run_exp112_l4_stage3_verifier_mediated_dry_run_design()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21K
    assert "phase21l" in report["allowed_next_phase"]


def test_forbidden_next_phases() -> None:
    report = run_exp112_l4_stage3_verifier_mediated_dry_run_design()
    assert set(FORBIDDEN_NEXT_PHASES_21K) <= set(report["forbidden_next_phases"])


def test_safety_invariant_flags_in_report() -> None:
    flags = run_exp112_l4_stage3_verifier_mediated_dry_run_design()["safety_invariant_flags"]
    for iid in SAFETY_INVARIANT_IDS:
        assert flags[iid] is True


def test_output_schema_has_dry_run_result_fields() -> None:
    schema = run_exp112_l4_stage3_verifier_mediated_dry_run_design()["output_schema"]
    required = set(schema["required_fields"])
    assert "decision_status" in required
    assert "prefix_match_length" in required
    assert "safety_gate_results" in required


def test_no_forbidden_positive_claims() -> None:
    report = run_exp112_l4_stage3_verifier_mediated_dry_run_design()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
