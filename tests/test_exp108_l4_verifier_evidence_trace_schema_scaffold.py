"""Tests for Experiment 108 L4 verifier evidence trace schema scaffold (Phase 21G)."""
from __future__ import annotations

import json

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.l4_verifier_evidence_trace_schema_scaffold import (
    EXPERIMENT_108_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_21G,
    RECOMMENDED_NEXT_PHASE_21G,
    SCAFFOLD_OUTCOME_COMPLETE,
    TRACE_SCHEMA_VERSION,
    build_synthetic_schema_examples,
    convert_verifier_trace_to_l4_trace_only_input,
    process_schema_example,
    run_exp108_l4_verifier_evidence_trace_schema_scaffold,
    validate_exp108_report,
    validate_verifier_evidence_trace_record,
)


def _example_by_id(example_id: str) -> dict:
    for ex in build_synthetic_schema_examples():
        if ex["example_id"] == example_id:
            return ex
    raise KeyError(example_id)


def test_valid_complete_all_match_trace_validates() -> None:
    ex = _example_by_id("complete_all_match_trace")
    result = validate_verifier_evidence_trace_record(ex["record"])
    assert result.valid is True


def test_valid_partial_match_trace_validates() -> None:
    ex = _example_by_id("complete_partial_match_trace")
    assert validate_verifier_evidence_trace_record(ex["record"]).valid is True


def test_valid_first_mismatch_trace_validates() -> None:
    ex = _example_by_id("complete_first_mismatch_trace")
    assert validate_verifier_evidence_trace_record(ex["record"]).valid is True


def test_missing_verifier_evidence_trace_validates_as_blocked() -> None:
    ex = _example_by_id("missing_verifier_evidence_trace")
    assert validate_verifier_evidence_trace_record(ex["record"]).valid is True
    conv = convert_verifier_trace_to_l4_trace_only_input(ex["record"])
    assert conv.decision_status == "blocked_missing_verifier_evidence"


def test_verifier_exception_trace_validates_as_blocked() -> None:
    ex = _example_by_id("verifier_exception_trace")
    assert validate_verifier_evidence_trace_record(ex["record"]).valid is True
    conv = convert_verifier_trace_to_l4_trace_only_input(ex["record"])
    assert conv.decision_status == "blocked_missing_verifier_evidence"


def test_committed_tokens_as_verifier_trace_fails() -> None:
    ex = _example_by_id("invalid_committed_tokens_as_verifier_trace")
    result = validate_verifier_evidence_trace_record(ex["record"])
    assert result.valid is False
    assert any("forbidden" in e for e in result.errors)


def test_counts_only_trace_fails() -> None:
    ex = _example_by_id("invalid_counts_only_trace")
    result = validate_verifier_evidence_trace_record(ex["record"])
    assert result.valid is False


def test_proposal_verifier_alias_trace_fails() -> None:
    ex = _example_by_id("invalid_proposal_verifier_alias_trace")
    result = validate_verifier_evidence_trace_record(ex["record"])
    assert result.valid is False
    assert any(
        "differ" in e or "alias" in e or "forbidden" in e for e in result.errors
    )


def test_diagnostic_only_false_fails() -> None:
    ex = _example_by_id("complete_all_match_trace")
    record = dict(ex["record"])
    record["diagnostic_only"] = False
    result = validate_verifier_evidence_trace_record(record)
    assert result.valid is False
    assert result.diagnostic_only_ok is False


def test_complete_trace_converts_to_dry_run_input() -> None:
    ex = _example_by_id("complete_all_match_trace")
    conv = convert_verifier_trace_to_l4_trace_only_input(ex["record"])
    assert conv.converted is True
    assert conv.dry_run_input is not None
    assert conv.dry_run_input.proposal_source == PROPOSAL_SOURCE_ROUND_LOG


def test_missing_verifier_evidence_converts_to_blocked_dry_run_input() -> None:
    ex = _example_by_id("missing_verifier_evidence_trace")
    conv = convert_verifier_trace_to_l4_trace_only_input(ex["record"])
    assert conv.converted is True
    assert conv.decision_status == "blocked_missing_verifier_evidence"


def test_dry_run_evaluation_works_after_conversion() -> None:
    ex = _example_by_id("complete_partial_match_trace")
    processed = process_schema_example(ex)
    assert processed["actual_dry_run_status"] == "partial_match"
    assert processed["converted_to_dry_run_input"] is True


def test_forbidden_source_rejection_summary() -> None:
    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    summary = report["forbidden_source_rejection_summary"]
    assert summary["forbidden_source_rejections"] >= 2


def test_report_schema_validates() -> None:
    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    assert report["experiment_id"] == EXPERIMENT_108_ID
    assert validate_exp108_report(report) == []


def test_status_is_schema_scaffold_complete() -> None:
    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    assert report["scaffold_decision"] == SCAFFOLD_OUTCOME_COMPLETE
    assert report["status"] == "scaffold_complete"


def test_allowed_next_phase_is_phase21h() -> None:
    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21G
    assert report["allowed_next_phase"] == "phase21h_l4_trace_only_dry_run_with_schema_examples"


def test_runtime_instrumentation_authorized_false() -> None:
    assert run_exp108_l4_verifier_evidence_trace_schema_scaffold()["runtime_instrumentation_authorized"] is False


def test_runtime_commit_authorized_false() -> None:
    assert run_exp108_l4_verifier_evidence_trace_schema_scaffold()["runtime_commit_authorized"] is False


def test_exactkv_generator_modified_false() -> None:
    assert run_exp108_l4_verifier_evidence_trace_schema_scaffold()["exactkv_generator_modified"] is False


def test_default_runtime_changed_false() -> None:
    assert run_exp108_l4_verifier_evidence_trace_schema_scaffold()["default_runtime_changed"] is False


def test_production_cli_modified_false() -> None:
    assert run_exp108_l4_verifier_evidence_trace_schema_scaffold()["production_cli_modified"] is False


def test_model_experiments_run_false() -> None:
    assert run_exp108_l4_verifier_evidence_trace_schema_scaffold()["model_experiments_run"] is False


def test_forbidden_next_phases() -> None:
    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    assert set(FORBIDDEN_NEXT_PHASES_21G) <= set(report["forbidden_next_phases"])


def test_schema_version_matches_design() -> None:
    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    assert report["schema_version"] == TRACE_SCHEMA_VERSION


def test_no_forbidden_positive_claims_in_report() -> None:
    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
