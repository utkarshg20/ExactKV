"""Tests for Experiment 113 L4 Stage 3 verifier-mediated dry-run scaffold (Phase 21L)."""
from __future__ import annotations

import json

from exactkv.safety.l4_stage3_verifier_mediated_dry_run_scaffold import (
    EXPERIMENT_113_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_21L,
    PANEL_OUTCOME_COMPLETE,
    RECOMMENDED_NEXT_PHASE_21L,
    SCAFFOLD_CASE_IDS,
    build_stage3_scaffold_cases,
    execute_scaffold_case,
    execute_stage3_dry_run,
    run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold,
    simulate_prefix_walk,
    validate_exp113_report,
)


def _case_by_id(case_id: str):
    for c in build_stage3_scaffold_cases():
        if c.case_id == case_id:
            return c
    raise KeyError(case_id)


def test_prefix_walk_full_match() -> None:
    length, idx, terminal = simulate_prefix_walk([1, 2, 3], [1, 2, 3])
    assert terminal == "ACCEPT_PREFIX"
    assert length == 3
    assert idx is None


def test_prefix_walk_partial_mismatch() -> None:
    length, idx, terminal = simulate_prefix_walk([1, 2, 9], [1, 2, 3])
    assert terminal == "REJECT"
    assert length == 2
    assert idx == 2


def test_scaffold_full_match() -> None:
    result = execute_scaffold_case(_case_by_id("scaffold_full_match"))
    assert result.dry_run_result.decision_status == "ACCEPT_PREFIX"
    assert result.scaffold_test_passed is True
    assert result.rollback_simulation.rollback_triggered is False


def test_scaffold_partial_mismatch() -> None:
    result = execute_scaffold_case(_case_by_id("scaffold_partial_mismatch"))
    assert result.dry_run_result.decision_status == "REJECT"
    assert result.dry_run_result.prefix_match_length == 2
    assert result.scaffold_test_passed is True
    assert result.rollback_simulation.rollback_triggered is True


def test_scaffold_missing_verifier() -> None:
    result = execute_scaffold_case(_case_by_id("scaffold_missing_verifier"))
    assert result.dry_run_result.decision_status == "BLOCK_MISSING_EVIDENCE"
    assert result.scaffold_test_passed is True


def test_scaffold_corrupted_trace() -> None:
    result = execute_scaffold_case(_case_by_id("scaffold_corrupted_trace"))
    assert result.dry_run_result.decision_status == "INVALID_TRACE"
    assert result.scaffold_test_passed is True


def test_scaffold_adversarial_aliasing() -> None:
    result = execute_scaffold_case(_case_by_id("scaffold_adversarial_aliasing"))
    assert result.dry_run_result.decision_status == "INVALID_TRACE"
    assert result.scaffold_test_passed is True


def test_all_scaffold_cases_present() -> None:
    cases = build_stage3_scaffold_cases()
    assert set(SCAFFOLD_CASE_IDS) <= {c.case_id for c in cases}


def test_deterministic_repeat_execution() -> None:
    case = _case_by_id("scaffold_full_match")
    r1 = execute_scaffold_case(case)
    r2 = execute_scaffold_case(case)
    assert r1.dry_run_result.to_dict() == r2.dry_run_result.to_dict()


def test_safety_gates_all_true() -> None:
    result = execute_scaffold_case(_case_by_id("scaffold_full_match"))
    assert all(result.dry_run_result.safety_gate_results.values())


def test_panel_all_cases_pass() -> None:
    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    assert report["cases_passed"] == report["total_cases"]


def test_report_schema_validates() -> None:
    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    assert report["experiment_id"] == EXPERIMENT_113_ID
    assert validate_exp113_report(report) == []


def test_panel_outcome_complete() -> None:
    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    assert report["panel_outcome"] == PANEL_OUTCOME_COMPLETE


def test_runtime_execution_not_authorized() -> None:
    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    assert report["runtime_execution_authorized"] is False
    assert report["runtime_commit_authorized"] is False


def test_exactkv_generator_modified_false() -> None:
    assert run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()["exactkv_generator_modified"] is False


def test_model_experiments_run_false() -> None:
    assert run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()["model_experiments_run"] is False


def test_allowed_next_phase() -> None:
    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21L
    assert "phase21m" in report["allowed_next_phase"]


def test_forbidden_next_phases() -> None:
    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    assert set(FORBIDDEN_NEXT_PHASES_21L) <= set(report["forbidden_next_phases"])


def test_decision_graph_trace_present() -> None:
    result = execute_stage3_dry_run(_case_by_id("scaffold_full_match").record, case_id="t")
    assert len(result.decision_graph_trace.steps) >= 2
    assert result.decision_graph_trace.terminal_state == "ACCEPT_PREFIX"


def test_no_forbidden_positive_claims() -> None:
    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
