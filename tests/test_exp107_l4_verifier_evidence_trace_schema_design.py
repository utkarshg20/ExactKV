"""Tests for Experiment 107 L4 verifier evidence trace schema design (Phase 21F)."""
from __future__ import annotations

import json

from exactkv.safety.l4_verifier_evidence_trace_schema_design import (
    DESIGN_OUTCOME_COMPLETE,
    EXPERIMENT_107_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_21F,
    RECOMMENDED_NEXT_PHASE_21F,
    REQUIRED_VERIFIER_EVIDENCE_FIELDS,
    build_l4_verifier_evidence_trace_schema_design,
    build_l4_verifier_evidence_trace_examples,
    evaluate_l4_verifier_evidence_schema_decision,
    run_exp107_l4_verifier_evidence_trace_schema_design,
    validate_exp107_report,
    validate_trace_example,
    validate_verifier_evidence_trace_record,
)


def test_schema_fields_include_all_required_verifier_evidence_fields() -> None:
    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    names = {f["field_name"] for f in report["schema_fields"]}
    assert set(REQUIRED_VERIFIER_EVIDENCE_FIELDS) <= names


def test_allowed_sources_include_full_kv_verifier_output() -> None:
    names = {s["source_name"] for s in run_exp107_l4_verifier_evidence_trace_schema_design()["allowed_sources"]}
    assert "full_kv_verifier_output_tokens" in names


def test_forbidden_sources_include_committed_counts_baseline_draft() -> None:
    names = {s["source_name"] for s in run_exp107_l4_verifier_evidence_trace_schema_design()["forbidden_sources"]}
    assert "committed_token_ids_unmarked" in names
    assert "accepted_token_counts_only" in names
    assert "baseline_generated_tokens" in names
    assert "round_log_proposal_tokens_as_verifier" in names


def test_validation_rules_require_explicit_evidence() -> None:
    rule_ids = {r["rule_id"] for r in run_exp107_l4_verifier_evidence_trace_schema_design()["validation_rules"]}
    assert "explicit_evidence_required" in rule_ids


def test_validation_rules_separate_proposal_and_verifier_evidence() -> None:
    rule_ids = {r["rule_id"] for r in run_exp107_l4_verifier_evidence_trace_schema_design()["validation_rules"]}
    assert "proposal_verifier_distinct" in rule_ids
    assert "missing_evidence_blocks" in rule_ids


def _example_by_id(example_id: str):
    for ex in build_l4_verifier_evidence_trace_examples():
        if ex.example_id == example_id:
            return ex
    raise KeyError(example_id)


def test_complete_all_match_example_validates() -> None:
    ex = _example_by_id("complete_all_match_trace")
    ok, _ = validate_trace_example(ex)
    assert ok is True
    passes, _ = validate_verifier_evidence_trace_record(ex.trace_record)
    assert passes is True
    assert ex.expected_dry_run_decision_status == "all_match"


def test_complete_partial_match_example_validates() -> None:
    ex = _example_by_id("complete_partial_match_trace")
    assert validate_trace_example(ex)[0] is True
    assert ex.expected_dry_run_decision_status == "partial_match"


def test_complete_first_mismatch_example_validates() -> None:
    ex = _example_by_id("complete_first_mismatch_trace")
    assert validate_trace_example(ex)[0] is True
    assert ex.expected_dry_run_decision_status == "first_token_mismatch"


def test_missing_verifier_evidence_example_validates_as_blocked() -> None:
    ex = _example_by_id("missing_verifier_evidence_trace")
    assert validate_trace_example(ex)[0] is True
    assert ex.expected_dry_run_decision_status == "blocked_missing_verifier_evidence"


def test_committed_tokens_as_verifier_example_fails() -> None:
    ex = _example_by_id("invalid_committed_tokens_as_verifier_trace")
    assert validate_trace_example(ex)[0] is True
    passes, errors = validate_verifier_evidence_trace_record(ex.trace_record)
    assert passes is False
    assert any("forbidden" in e for e in errors)


def test_counts_only_example_fails() -> None:
    ex = _example_by_id("invalid_counts_only_trace")
    assert validate_trace_example(ex)[0] is True
    passes, errors = validate_verifier_evidence_trace_record(ex.trace_record)
    assert passes is False
    assert any("num_accepted" in e or "forbidden" in e for e in errors)


def test_design_decision_is_complete() -> None:
    design = build_l4_verifier_evidence_trace_schema_design()
    decision = evaluate_l4_verifier_evidence_schema_decision(design)
    assert decision.outcome == DESIGN_OUTCOME_COMPLETE
    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    assert report["design_decision"]["outcome"] == DESIGN_OUTCOME_COMPLETE


def test_allowed_next_phase_is_phase21g_scaffold() -> None:
    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21F
    assert report["allowed_next_phase"] == "phase21g_l4_verifier_evidence_trace_schema_scaffold"


def test_runtime_instrumentation_authorized_false() -> None:
    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    assert report["runtime_instrumentation_authorized"] is False
    assert report["design_decision"]["runtime_instrumentation_authorized"] is False


def test_runtime_commit_authorized_false() -> None:
    assert run_exp107_l4_verifier_evidence_trace_schema_design()["runtime_commit_authorized"] is False


def test_exactkv_generator_modified_false() -> None:
    assert run_exp107_l4_verifier_evidence_trace_schema_design()["exactkv_generator_modified"] is False


def test_default_runtime_changed_false() -> None:
    assert run_exp107_l4_verifier_evidence_trace_schema_design()["default_runtime_changed"] is False


def test_production_cli_modified_false() -> None:
    assert run_exp107_l4_verifier_evidence_trace_schema_design()["production_cli_modified"] is False


def test_model_experiments_run_false() -> None:
    assert run_exp107_l4_verifier_evidence_trace_schema_design()["model_experiments_run"] is False


def test_forbidden_next_phases_include_runtime_commit() -> None:
    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    assert "l4_runtime_commit_implementation" in report["forbidden_next_phases"]
    assert set(FORBIDDEN_NEXT_PHASES_21F) <= set(report["forbidden_next_phases"])


def test_report_schema_validates() -> None:
    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    assert report["experiment_id"] == EXPERIMENT_107_ID
    assert validate_exp107_report(report) == []


def test_no_forbidden_positive_claims_in_report() -> None:
    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
