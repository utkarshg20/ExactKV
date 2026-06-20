"""Tests for Experiment 104 L4 trace-only dry-run design (Phase 21C)."""
from __future__ import annotations

import json

from exactkv.safety.l4_trace_only_dry_run_design import (
    DECISION_STATUSES,
    DESIGN_OUTCOME_COMPLETE,
    EXPERIMENT_104_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASE_21C,
    RECOMMENDED_NEXT_PHASE_21C,
    REQUIRED_RISK_IDS,
    STAGE_2_GATE_NAMES,
    build_l4_trace_only_dry_run_design,
    evaluate_l4_trace_only_design_decision,
    run_exp104_l4_trace_only_dry_run_design,
    validate_exp104_report,
)


def test_design_schema_validates() -> None:
    report = run_exp104_l4_trace_only_dry_run_design()
    assert report["experiment_id"] == EXPERIMENT_104_ID
    assert validate_exp104_report(report) == []


def test_intended_behavior_says_generation_runs_unchanged() -> None:
    report = run_exp104_l4_trace_only_dry_run_design()
    flow = " ".join(report["intended_trace_only_behavior"]).lower()
    assert "unchanged" in flow or "exactly as today" in flow


def test_intended_behavior_says_decisions_are_diagnostics_only() -> None:
    report = run_exp104_l4_trace_only_dry_run_design()
    flow = " ".join(report["intended_trace_only_behavior"]).lower()
    assert "diagnostic" in flow
    assert "never used to commit" in flow


def test_evidence_plan_includes_round_log_draft_tokens() -> None:
    evidence = run_exp104_l4_trace_only_dry_run_design()["evidence_source_plan"]
    sources = evidence["allowed_draft_proposal_sources"]
    assert "exactkv_round_log_draft_tokens" in sources


def test_evidence_plan_blocks_missing_verifier_evidence() -> None:
    evidence = run_exp104_l4_trace_only_dry_run_design()["evidence_source_plan"]
    assert evidence["missing_verifier_evidence_blocks_decision"] is True
    assert evidence["missing_proposal_blocks_decision"] is True


def test_decision_schema_includes_accepted_rejected_token_fields() -> None:
    schema = run_exp104_l4_trace_only_dry_run_design()["dry_run_decision_schema"]
    fields = set(schema["field_names"])
    assert "accepted_prefix_token_ids" in fields
    assert "rejected_suffix_token_ids" in fields


def test_statuses_include_all_required_statuses() -> None:
    schema = run_exp104_l4_trace_only_dry_run_design()["dry_run_decision_schema"]
    assert set(DECISION_STATUSES) <= set(schema["decision_statuses"])


def test_safety_gates_include_no_commit_effect_gate() -> None:
    gates = {g["name"] for g in run_exp104_l4_trace_only_dry_run_design()["stage_2_safety_gates"]}
    assert "no_commit_effect_gate" in gates
    assert gates == set(STAGE_2_GATE_NAMES)


def test_safety_gates_include_no_generator_exposure_gate() -> None:
    gates = {g["name"] for g in run_exp104_l4_trace_only_dry_run_design()["stage_2_safety_gates"]}
    assert "no_generator_exposure_gate" in gates


def test_risk_register_includes_required_risks() -> None:
    risks = {r["risk_id"] for r in run_exp104_l4_trace_only_dry_run_design()["risk_register"]}
    assert set(REQUIRED_RISK_IDS) <= risks


def test_design_decision_is_complete() -> None:
    design = build_l4_trace_only_dry_run_design()
    decision = evaluate_l4_trace_only_design_decision(design)
    assert decision.outcome == DESIGN_OUTCOME_COMPLETE
    report = run_exp104_l4_trace_only_dry_run_design()
    assert report["design_decision"]["outcome"] == DESIGN_OUTCOME_COMPLETE


def test_allowed_next_phase_is_phase21d_scaffold() -> None:
    report = run_exp104_l4_trace_only_dry_run_design()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21C
    assert report["allowed_next_phase"] == "phase21d_l4_trace_only_dry_run_scaffold"


def test_forbidden_next_phase_is_runtime_commit() -> None:
    report = run_exp104_l4_trace_only_dry_run_design()
    assert report["forbidden_next_phase"] == FORBIDDEN_NEXT_PHASE_21C


def test_runtime_commit_authorized_is_false() -> None:
    report = run_exp104_l4_trace_only_dry_run_design()
    assert report["runtime_commit_authorized"] is False
    assert report["design_decision"]["runtime_commit_authorized"] is False


def test_exactkv_generator_modified_is_false() -> None:
    assert run_exp104_l4_trace_only_dry_run_design()["exactkv_generator_modified"] is False


def test_default_runtime_changed_is_false() -> None:
    assert run_exp104_l4_trace_only_dry_run_design()["default_runtime_changed"] is False


def test_production_cli_modified_is_false() -> None:
    assert run_exp104_l4_trace_only_dry_run_design()["production_cli_modified"] is False


def test_model_experiments_run_is_false() -> None:
    assert run_exp104_l4_trace_only_dry_run_design()["model_experiments_run"] is False


def test_no_forbidden_positive_claims_in_report() -> None:
    report = run_exp104_l4_trace_only_dry_run_design()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
