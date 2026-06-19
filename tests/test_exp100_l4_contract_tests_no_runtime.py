"""Tests for Experiment 100 L4 contract tests no runtime (Phase 20C)."""
from __future__ import annotations

import json

from exactkv.safety.l4_contract_tests_no_runtime import (
    EXPERIMENT_100_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_20C,
    RECOMMENDED_NEXT_PHASE_20C,
    build_default_l4_contract_test_suite,
    evaluate_l4_synthetic_contract_case,
    run_exp100_l4_contract_tests_no_runtime,
    run_l4_contract_test_suite,
    validate_exp100_report,
)


def _case(case_id: str):
    cases = {c.case_id: c for c in build_default_l4_contract_test_suite()}
    return cases[case_id]


def test_all_match_case_accepts_all() -> None:
    result = evaluate_l4_synthetic_contract_case(_case("all_match_accept_all"))
    assert result.case_passed
    assert result.decision.accepted_prefix == (1, 2, 3, 4)
    assert result.decision.rejected_suffix == ()


def test_partial_match_case_accepts_longest_matching_prefix() -> None:
    result = evaluate_l4_synthetic_contract_case(_case("partial_match_accept_prefix"))
    assert result.case_passed
    assert result.decision.accepted_prefix == (1, 2)
    assert result.decision.rejected_suffix == (9, 9)


def test_first_token_mismatch_accepts_empty_prefix() -> None:
    result = evaluate_l4_synthetic_contract_case(_case("first_token_mismatch_accept_none"))
    assert result.case_passed
    assert result.decision.accepted_prefix == ()
    assert result.decision.rejected_suffix == (9, 9, 9)


def test_proposal_exception_triggers_fallback() -> None:
    result = evaluate_l4_synthetic_contract_case(_case("proposal_exception_fallback"))
    assert result.case_passed
    assert result.decision.fallback_triggered
    assert result.decision.fallback_required


def test_missing_verifier_evidence_triggers_fallback() -> None:
    result = evaluate_l4_synthetic_contract_case(
        _case("missing_verifier_evidence_fallback"),
    )
    assert result.case_passed
    assert result.decision.fallback_triggered
    assert result.decision.fallback_required


def test_hidden_divergence_attempt_fails() -> None:
    result = evaluate_l4_synthetic_contract_case(_case("hidden_divergence_attempt_fails"))
    assert result.actual_status == "fail"
    assert result.case_passed
    assert result.decision.hidden_divergence_detected


def test_direct_commit_attempt_fails() -> None:
    result = evaluate_l4_synthetic_contract_case(_case("direct_commit_attempt_fails"))
    assert result.actual_status == "fail"
    assert result.case_passed
    assert result.decision.direct_commit_rejected


def test_verifier_source_of_truth_is_always_required() -> None:
    results = run_l4_contract_test_suite()
    assert all(r.decision.verifier_source_of_truth for r in results)


def test_trace_completeness_is_checked() -> None:
    results = run_l4_contract_test_suite()
    assert all(r.trace.trace_complete for r in results)
    for result in results:
        assert result.trace.decision_steps
        assert "contract_evaluator_started" in result.trace.decision_steps


def test_suite_summary_counts_pass_fail_fallback_correctly() -> None:
    report = run_exp100_l4_contract_tests_no_runtime()
    suite = report["suite_summary"]
    assert suite["total_cases"] == 7
    assert suite["passing_cases"] == 7
    assert suite["failing_cases"] == 0
    assert suite["expected_fail_cases"] == 2
    assert suite["unexpected_fail_cases"] == 0
    assert suite["fallback_cases"] == 2
    assert suite["hidden_divergence_failures_detected"] == 1
    assert suite["direct_commit_failures_detected"] == 1
    assert suite["trace_complete_cases"] == 7
    assert suite["suite_status"] == "contract_tests_complete"


def test_report_schema_validates() -> None:
    report = run_exp100_l4_contract_tests_no_runtime()
    assert report["experiment_id"] == EXPERIMENT_100_ID
    assert validate_exp100_report(report) == []


def test_exactkv_generator_modified_is_false() -> None:
    assert run_exp100_l4_contract_tests_no_runtime()["exactkv_generator_modified"] is False


def test_default_runtime_changed_is_false() -> None:
    assert run_exp100_l4_contract_tests_no_runtime()["default_runtime_changed"] is False


def test_l4_runtime_implementation_added_is_false() -> None:
    report = run_exp100_l4_contract_tests_no_runtime()
    assert report["l4_runtime_implementation_added"] is False
    assert report["runtime_generation_path_modified"] is False


def test_allowed_next_phase_is_phase20d_integration_plan_review() -> None:
    report = run_exp100_l4_contract_tests_no_runtime()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_20C
    assert report["allowed_next_phase"] == "phase20d_l4_integration_plan_review"


def test_forbidden_next_phase_includes_l4_runtime_implementation() -> None:
    report = run_exp100_l4_contract_tests_no_runtime()
    assert report["forbidden_next_phase"] in FORBIDDEN_NEXT_PHASES_20C
    assert "phase20c_l4_runtime_implementation" in report["forbidden_next_phases"]


def test_no_forbidden_positive_claims_in_report() -> None:
    report = run_exp100_l4_contract_tests_no_runtime()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
