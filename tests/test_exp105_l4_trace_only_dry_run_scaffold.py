"""Tests for Experiment 105 L4 trace-only dry-run scaffold (Phase 21D)."""
from __future__ import annotations

import json

from exactkv.safety.l4_trace_only_dry_run_scaffold import (
    DECISION_STATUSES,
    EXPERIMENT_105_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASE_21D,
    RECOMMENDED_NEXT_PHASE_21D,
    build_default_synthetic_trace_records,
    build_l4_trace_only_inputs_from_records,
    build_synthetic_exp105_report,
    evaluate_l4_trace_only_input,
    run_exp105_l4_trace_only_dry_run_scaffold,
    run_synthetic_trace_only_suite,
    validate_exp105_report,
    validate_l4_trace_only_scaffold_report,
)


def _input(cell_id: str):
    records = {r["cell_id"]: r for r in build_default_synthetic_trace_records()}
    inputs = build_l4_trace_only_inputs_from_records([records[cell_id]])
    return inputs[0]


def test_all_match_decision_accepts_all() -> None:
    dec = evaluate_l4_trace_only_input(_input("all_match"))
    assert dec.decision_status == "all_match"
    assert dec.accepted_prefix_token_ids == (1, 2, 3, 4)
    assert dec.rejected_suffix_token_ids == ()


def test_partial_match_decision_accepts_longest_prefix() -> None:
    dec = evaluate_l4_trace_only_input(_input("partial_match"))
    assert dec.decision_status == "partial_match"
    assert dec.accepted_prefix_token_ids == (1, 2)
    assert dec.rejected_suffix_token_ids == (9, 9)


def test_first_token_mismatch_accepts_empty_prefix() -> None:
    dec = evaluate_l4_trace_only_input(_input("first_token_mismatch"))
    assert dec.decision_status == "first_token_mismatch"
    assert dec.accepted_prefix_token_ids == ()
    assert dec.rejected_suffix_token_ids == (9, 9, 9)


def test_missing_proposal_blocks() -> None:
    dec = evaluate_l4_trace_only_input(_input("blocked_missing_proposal"))
    assert dec.decision_status == "blocked_missing_proposal"
    assert dec.block_reason is not None


def test_missing_verifier_evidence_blocks() -> None:
    dec = evaluate_l4_trace_only_input(_input("blocked_missing_verifier_evidence"))
    assert dec.decision_status == "blocked_missing_verifier_evidence"
    assert dec.accepted_prefix_token_ids == ()


def test_hidden_divergence_fails() -> None:
    dec = evaluate_l4_trace_only_input(_input("failed_hidden_divergence"))
    assert dec.decision_status == "failed_hidden_divergence"


def test_direct_commit_fails() -> None:
    dec = evaluate_l4_trace_only_input(_input("failed_direct_commit_attempt"))
    assert dec.decision_status == "failed_direct_commit_attempt"


def test_missing_verifier_evidence_never_treated_as_match() -> None:
    dec = evaluate_l4_trace_only_input(_input("blocked_missing_verifier_evidence"))
    assert dec.accepted_prefix_token_ids == ()
    assert dec.decision_status != "all_match"


def test_dry_run_decision_used_for_token_commit_false() -> None:
    for dec in run_synthetic_trace_only_suite():
        assert dec.dry_run_decision_used_for_token_commit is False


def test_exposed_to_generator_false() -> None:
    for dec in run_synthetic_trace_only_suite():
        assert dec.exposed_to_generator is False


def test_verifier_source_of_truth_true() -> None:
    for dec in run_synthetic_trace_only_suite():
        assert dec.verifier_source_of_truth is True


def test_trace_completeness_checks_required_fields() -> None:
    dec = evaluate_l4_trace_only_input(_input("all_match"))
    assert dec.trace_complete is True
    assert dec.proposal_source
    assert dec.verifier_evidence_source
    assert dec.decision_status


def test_report_validation_passes_safe_synthetic_report() -> None:
    report = build_synthetic_exp105_report()
    assert report["validation_result"]["valid"] is True
    assert validate_exp105_report(report) == []


def test_report_validation_fails_unsafe_synthetic_report() -> None:
    report = build_synthetic_exp105_report(unsafe=True)
    assert validate_l4_trace_only_scaffold_report(report).valid is False


def test_report_schema_validates() -> None:
    report = run_exp105_l4_trace_only_dry_run_scaffold()
    assert report["experiment_id"] == EXPERIMENT_105_ID
    assert validate_exp105_report(report) == []


def test_synthetic_suite_covers_all_statuses() -> None:
    report = run_exp105_l4_trace_only_dry_run_scaffold()
    synth = report["synthetic_suite_summary"]
    assert synth["all_statuses_covered"] is True
    assert synth["total_decisions"] == len(DECISION_STATUSES)


def test_allowed_next_phase_is_phase21e_panel_validation() -> None:
    report = run_exp105_l4_trace_only_dry_run_scaffold()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21D
    assert report["allowed_next_phase"] == "phase21e_l4_trace_only_dry_run_panel_validation"


def test_forbidden_next_phase_is_runtime_commit() -> None:
    report = run_exp105_l4_trace_only_dry_run_scaffold()
    assert report["forbidden_next_phase"] == FORBIDDEN_NEXT_PHASE_21D


def test_no_forbidden_positive_claims_in_report() -> None:
    report = run_exp105_l4_trace_only_dry_run_scaffold()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
