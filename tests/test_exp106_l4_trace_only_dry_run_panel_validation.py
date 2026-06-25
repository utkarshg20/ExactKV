"""Tests for Experiment 106 L4 trace-only dry-run panel validation (Phase 21E)."""
from __future__ import annotations

import json

from exactkv.safety.guarded_draft_shadow import PROPOSAL_SOURCE_ROUND_LOG
from exactkv.safety.l4_trace_only_dry_run_scaffold import (
    EXPERIMENT_106_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASE_21E,
    RECOMMEND_PHASE21F_STAGE3_DESIGN,
    RECOMMEND_PHASE21F_VERIFIER_SCHEMA,
    VERIFIER_EVIDENCE_BLOCK_REASON,
    aggregate_trace_only_panel_breakdowns,
    build_synthetic_exp106_panel_report,
    compute_phase21f_recommendation,
    evaluate_l4_trace_only_input,
    extract_verifier_evidence_from_round_trace,
    run_exp106_l4_trace_only_dry_run_panel_validation,
    validate_exp106_panel_report,
    validate_exp106_report,
)
from exactkv.safety.l4_trace_only_dry_run_scaffold import (
    build_l4_trace_only_inputs_from_records,
)

TOKENS = (10, 11, 12)


def _generation_fn(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": list(TOKENS[:2]),
        "generated_text": "out",
        "exactkv_failures": 0,
        "result_traces": [
            {
                "round_idx": 0,
                "draft_tokens": list(TOKENS),
                "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
            },
        ],
    }


def _generation_fn_with_verifier(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": list(TOKENS[:2]),
        "generated_text": "out",
        "exactkv_failures": 0,
        "result_traces": [
            {
                "round_idx": 0,
                "draft_tokens": list(TOKENS),
                "verifier_token_ids": list(TOKENS),
                "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
            },
        ],
    }


def test_panel_schema_validates() -> None:
    report = build_synthetic_exp106_panel_report()
    assert report["experiment_id"] == EXPERIMENT_106_ID
    assert validate_exp106_panel_report(report).valid is True


def test_missing_verifier_evidence_blocks_decisions() -> None:
    report = run_exp106_l4_trace_only_dry_run_panel_validation(
        model_id="m1",
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        max_new_tokens_values=[4],
        generation_fn=_generation_fn,
    )
    cell = report["cells"][0]
    assert cell["decisions_blocked"] >= 1
    dec = cell["decisions"][0]
    assert dec["decision_status"] == "blocked_missing_verifier_evidence"
    assert dec["block_reason"] == VERIFIER_EVIDENCE_BLOCK_REASON
    assert dec["accepted_prefix_token_ids"] == []


def test_missing_verifier_evidence_does_not_fail_validation_by_itself() -> None:
    report = build_synthetic_exp106_panel_report(with_verifier_evidence=False)
    assert report["verifier_evidence_coverage_rate"] == 0.0
    assert report["validation_result"]["valid"] is True


def test_missing_verifier_evidence_treated_as_match_fails_validation() -> None:
    report = build_synthetic_exp106_panel_report()
    report["cells"][0]["decisions"][0]["accepted_prefix_token_ids"] = [1]
    report["cells"][0]["decisions"][0]["decision_status"] = (
        "blocked_missing_verifier_evidence"
    )
    assert validate_exp106_panel_report(report).valid is False


def test_dry_run_decision_used_for_commit_fails_validation() -> None:
    report = build_synthetic_exp106_panel_report()
    report["dry_run_decision_used_for_token_commit"] = True
    assert validate_exp106_panel_report(report).valid is False


def test_generator_exposure_fails_validation() -> None:
    report = build_synthetic_exp106_panel_report()
    report["exposed_to_generator"] = True
    assert validate_exp106_panel_report(report).valid is False


def test_token_parity_failure_fails_validation() -> None:
    report = build_synthetic_exp106_panel_report(unsafe=True)
    assert report["status"] == "failed"
    assert validate_exp106_panel_report(report).valid is False


def test_proposal_evidence_coverage_aggregation() -> None:
    report = run_exp106_l4_trace_only_dry_run_panel_validation(
        model_id="m1",
        prompts=[("p0", "a"), ("p1", "b")],
        compressors_requested=["noop"],
        max_new_tokens_values=[4],
        generation_fn=_generation_fn_with_verifier,
    )
    assert report["proposal_evidence_coverage_rate"] == 1.0
    assert report["trace_inputs_built"] == 2


def test_verifier_evidence_coverage_aggregation() -> None:
    report = run_exp106_l4_trace_only_dry_run_panel_validation(
        model_id="m1",
        prompts=[("p0", "a")],
        compressors_requested=["noop"],
        max_new_tokens_values=[4],
        generation_fn=_generation_fn_with_verifier,
    )
    assert report["verifier_evidence_coverage_rate"] == 1.0
    assert report["verifier_evidence_available_count"] == 1


def test_decision_status_breakdown_works() -> None:
    report = run_exp106_l4_trace_only_dry_run_panel_validation(
        model_id="m1",
        prompts=[("p0", "a")],
        compressors_requested=["noop"],
        max_new_tokens_values=[4],
        generation_fn=_generation_fn_with_verifier,
    )
    breakdowns = aggregate_trace_only_panel_breakdowns(report["cells"])
    assert "breakdowns_by_model" in breakdowns
    assert breakdowns["breakdowns_by_decision_status"].get("all_match", 0) >= 1


def test_recommendation_schema_design_when_verifier_coverage_zero() -> None:
    rec, _ = compute_phase21f_recommendation(
        verifier_evidence_coverage_rate=0.0,
        safety_gates_ok=True,
    )
    assert rec == RECOMMEND_PHASE21F_VERIFIER_SCHEMA
    report = build_synthetic_exp106_panel_report(with_verifier_evidence=False)
    assert report["decision_recommendation"] == RECOMMEND_PHASE21F_VERIFIER_SCHEMA


def test_recommendation_stage3_when_verifier_coverage_sufficient() -> None:
    rec, _ = compute_phase21f_recommendation(
        verifier_evidence_coverage_rate=1.0,
        safety_gates_ok=True,
    )
    assert rec == RECOMMEND_PHASE21F_STAGE3_DESIGN
    report = build_synthetic_exp106_panel_report(with_verifier_evidence=True)
    assert report["decision_recommendation"] == RECOMMEND_PHASE21F_STAGE3_DESIGN


def test_report_validates_safe_synthetic_panel() -> None:
    report = build_synthetic_exp106_panel_report()
    assert report["validation_result"]["valid"] is True
    assert validate_exp106_report(report) == []


def test_extract_verifier_does_not_use_accepted_tokens() -> None:
    trace = {
        "acceptance": {
            "accepted_tokens": [1, 2, 3],
            "verifier_tokens": None,
        },
    }
    ids, source = extract_verifier_evidence_from_round_trace(trace)
    assert ids == ()
    assert source is None


def test_forbidden_next_phase_is_runtime_commit() -> None:
    report = build_synthetic_exp106_panel_report()
    assert report["forbidden_next_phase"] == FORBIDDEN_NEXT_PHASE_21E


def test_no_forbidden_positive_claims_in_report() -> None:
    report = build_synthetic_exp106_panel_report()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
            "decision_reason": report.get("decision_reason"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_missing_verifier_never_all_match_via_evaluator() -> None:
    inputs = build_l4_trace_only_inputs_from_records(
        [
            {
                "cell_id": "c0",
                "prompt_id": "p0",
                "compressor": "noop",
                "round_index": 0,
                "proposal_token_ids": [1, 2],
                "verifier_evidence_token_ids": [],
                "proposal_source": PROPOSAL_SOURCE_ROUND_LOG,
                "verifier_evidence_source": "",
            },
        ],
    )
    dec = evaluate_l4_trace_only_input(inputs[0])
    assert dec.decision_status == "blocked_missing_verifier_evidence"
    assert dec.decision_status != "all_match"
