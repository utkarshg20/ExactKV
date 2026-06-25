"""Tests for Experiment 109 L4 verifier trace schema example validation (Phase 21H)."""
from __future__ import annotations

import json

from exactkv.safety.l4_verifier_trace_schema_example_validation import (
    EXPERIMENT_109_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_21H,
    RECOMMENDED_NEXT_PHASE_21H,
    SCHEMA_ENFORCEMENT_RULES,
    VALIDATION_OUTCOME_COMPLETE,
    build_enforcement_rule_coverage,
    build_synthetic_schema_examples,
    execute_schema_example,
    run_diagnostic_probes,
    run_exp109_l4_verifier_trace_schema_example_validation,
    validate_exp109_report,
)


def _example_by_id(example_id: str) -> dict:
    for ex in build_synthetic_schema_examples():
        if ex["example_id"] == example_id:
            return ex
    raise KeyError(example_id)


def test_all_eight_schema_examples_execute() -> None:
    examples = build_synthetic_schema_examples()
    assert len(examples) == 8
    results = [execute_schema_example(ex) for ex in examples]
    assert len(results) == 8


def test_complete_all_match_validates_and_classifies() -> None:
    result = execute_schema_example(_example_by_id("complete_all_match_trace"))
    assert result.validation_passed is True
    assert result.classification_passed is True
    assert result.actual_classification == "all_match"


def test_complete_partial_match_classifies() -> None:
    result = execute_schema_example(_example_by_id("complete_partial_match_trace"))
    assert result.actual_classification == "partial_match"
    assert result.classification_passed is True


def test_complete_first_mismatch_classifies() -> None:
    result = execute_schema_example(_example_by_id("complete_first_mismatch_trace"))
    assert result.actual_classification == "first_token_mismatch"
    assert result.classification_passed is True


def test_missing_verifier_evidence_blocks() -> None:
    result = execute_schema_example(_example_by_id("missing_verifier_evidence_trace"))
    assert result.actual_classification == "blocked_missing_verifier_evidence"
    assert result.classification_passed is True


def test_verifier_exception_blocks() -> None:
    result = execute_schema_example(_example_by_id("verifier_exception_trace"))
    assert result.actual_classification == "blocked_missing_verifier_evidence"


def test_committed_tokens_as_verifier_rejected() -> None:
    result = execute_schema_example(_example_by_id("invalid_committed_tokens_as_verifier_trace"))
    assert result.actual_validation is False
    assert result.classification_passed is True


def test_counts_only_rejected() -> None:
    result = execute_schema_example(_example_by_id("invalid_counts_only_trace"))
    assert result.actual_validation is False


def test_proposal_verifier_alias_rejected() -> None:
    result = execute_schema_example(_example_by_id("invalid_proposal_verifier_alias_trace"))
    assert result.actual_validation is False
    assert "proposal_verifier_separation" in result.enforcement_rules_exercised


def test_diagnostic_probes_pass() -> None:
    probes = run_diagnostic_probes()
    assert all(p["passed"] for p in probes)


def test_enforcement_rule_coverage_all_exercised() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    coverage = report["enforcement_rule_coverage"]
    assert len(coverage) == len(SCHEMA_ENFORCEMENT_RULES)
    assert all(c["exercised"] for c in coverage)


def test_invalid_trace_detection_accuracy() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    summary = report["invalid_trace_detection_summary"]
    assert summary["detection_accuracy"] == 1.0


def test_blocked_evidence_correctness() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    assert report["blocked_evidence_summary"]["blocked_correct"] is True


def test_no_commit_or_generator_exposure() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    for ex in report["example_results"]:
        assert ex["dry_run_decision_used_for_token_commit"] is False
        assert ex["exposed_to_generator"] is False


def test_report_schema_validates() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    assert report["experiment_id"] == EXPERIMENT_109_ID
    assert validate_exp109_report(report) == []


def test_validation_outcome_complete() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    assert report["validation_outcome"] == VALIDATION_OUTCOME_COMPLETE
    assert report["status"] == "validation_complete"


def test_allowed_next_phase_is_phase21i() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21H
    assert "phase21i" in report["allowed_next_phase"]


def test_runtime_instrumentation_authorized_false() -> None:
    assert run_exp109_l4_verifier_trace_schema_example_validation()["runtime_instrumentation_authorized"] is False


def test_runtime_commit_authorized_false() -> None:
    assert run_exp109_l4_verifier_trace_schema_example_validation()["runtime_commit_authorized"] is False


def test_exactkv_generator_modified_false() -> None:
    assert run_exp109_l4_verifier_trace_schema_example_validation()["exactkv_generator_modified"] is False


def test_default_runtime_changed_false() -> None:
    assert run_exp109_l4_verifier_trace_schema_example_validation()["default_runtime_changed"] is False


def test_production_cli_modified_false() -> None:
    assert run_exp109_l4_verifier_trace_schema_example_validation()["production_cli_modified"] is False


def test_model_experiments_run_false() -> None:
    assert run_exp109_l4_verifier_trace_schema_example_validation()["model_experiments_run"] is False


def test_forbidden_next_phases() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    assert set(FORBIDDEN_NEXT_PHASES_21H) <= set(report["forbidden_next_phases"])


def test_no_forbidden_positive_claims() -> None:
    report = run_exp109_l4_verifier_trace_schema_example_validation()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
