"""Tests for Experiment 110 L4 trace schema adversarial injection panel (Phase 21I)."""
from __future__ import annotations

import json

from exactkv.safety.l4_trace_schema_adversarial_injection_panel import (
    ADVERSARIAL_CATEGORIES,
    EXPERIMENT_110_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_21I,
    PANEL_OUTCOME_COMPLETE,
    RECOMMENDED_NEXT_PHASE_21I,
    build_adversarial_injection_cases,
    execute_adversarial_case,
    run_exp110_l4_trace_schema_adversarial_injection_panel,
    validate_exp110_report,
)


def _case_by_id(case_id: str):
    for c in build_adversarial_injection_cases():
        if c.case_id == case_id:
            return c
    raise KeyError(case_id)


def test_all_adversarial_categories_present() -> None:
    cases = build_adversarial_injection_cases()
    categories = {c.category for c in cases}
    assert set(ADVERSARIAL_CATEGORIES) <= categories
    assert len(cases) >= 14


def test_missing_field_attack_rejected() -> None:
    result = execute_adversarial_case(_case_by_id("missing_verifier_fields_entirely"))
    assert result.actual_schema_valid is False
    assert result.adversarial_test_passed is True


def test_forgery_committed_tokens_detected_poisoning() -> None:
    result = execute_adversarial_case(_case_by_id("forgery_committed_tokens_as_verifier"))
    assert result.actual_panel_classification == "detected_poisoning"
    assert result.false_acceptance is False


def test_forgery_alias_detected() -> None:
    result = execute_adversarial_case(_case_by_id("forgery_proposal_verifier_alias"))
    assert result.actual_panel_classification == "detected_poisoning"


def test_structural_poisoning_rejected() -> None:
    result = execute_adversarial_case(_case_by_id("poison_invalid_schema_version"))
    assert result.actual_schema_valid is False


def test_divergence_first_token_passes() -> None:
    result = execute_adversarial_case(_case_by_id("divergence_first_token"))
    assert result.actual_panel_classification == "pass"
    assert result.dry_run_status == "first_token_mismatch"


def test_divergence_mid_sequence_passes() -> None:
    result = execute_adversarial_case(_case_by_id("divergence_mid_sequence"))
    assert result.actual_panel_classification == "pass"
    assert result.dry_run_status == "partial_match"


def test_silent_missing_verifier_blocks() -> None:
    result = execute_adversarial_case(_case_by_id("silent_missing_verifier_no_block"))
    assert result.actual_panel_classification == "blocked_missing_verifier_evidence"


def test_silent_empty_verifier_rejected() -> None:
    result = execute_adversarial_case(_case_by_id("silent_empty_verifier_with_available_true"))
    assert result.actual_schema_valid is False


def test_false_acceptance_rate_zero() -> None:
    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    assert report["false_acceptance_rate"] == 0.0


def test_all_cases_pass() -> None:
    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    assert report["cases_passed"] == report["total_cases"]


def test_report_schema_validates() -> None:
    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    assert report["experiment_id"] == EXPERIMENT_110_ID
    assert validate_exp110_report(report) == []


def test_panel_outcome_complete() -> None:
    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    assert report["panel_outcome"] == PANEL_OUTCOME_COMPLETE


def test_runtime_instrumentation_authorized_false() -> None:
    assert run_exp110_l4_trace_schema_adversarial_injection_panel()["runtime_instrumentation_authorized"] is False


def test_runtime_commit_authorized_false() -> None:
    assert run_exp110_l4_trace_schema_adversarial_injection_panel()["runtime_commit_authorized"] is False


def test_exactkv_generator_modified_false() -> None:
    assert run_exp110_l4_trace_schema_adversarial_injection_panel()["exactkv_generator_modified"] is False


def test_model_experiments_run_false() -> None:
    assert run_exp110_l4_trace_schema_adversarial_injection_panel()["model_experiments_run"] is False


def test_allowed_next_phase() -> None:
    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21I
    assert "phase21j" in report["allowed_next_phase"]


def test_forbidden_next_phases() -> None:
    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    assert set(FORBIDDEN_NEXT_PHASES_21I) <= set(report["forbidden_next_phases"])


def test_no_forbidden_positive_claims() -> None:
    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
