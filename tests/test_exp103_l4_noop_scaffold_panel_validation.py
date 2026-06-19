"""Tests for Experiment 103 L4 no-op scaffold panel validation (Phase 21B)."""
from __future__ import annotations

import json

from exactkv.safety.l4_noop_opt_in_scaffold import (
    EXPERIMENT_103_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASE_21B,
    RECOMMENDED_NEXT_PHASE_21B,
    build_l4_noop_scaffold_trace,
    build_synthetic_exp103_panel_report,
    default_l4_noop_opt_in_config,
    default_l4_noop_safety_gates,
    run_exp103_l4_noop_scaffold_panel_validation,
    validate_exp103_panel_report,
)

TOKENS = (100, 101, 102, 103)


def _baseline_fn(**kwargs: object) -> dict:
    del kwargs
    return {
        "generation_completed": True,
        "generated_token_ids": list(TOKENS),
        "generated_text": "out",
        "exactkv_failures": 0,
        "blockers": [],
    }


def _noop_fn(**kwargs: object) -> tuple[dict, object]:
    del kwargs
    config = default_l4_noop_opt_in_config(enabled=True)
    trace = build_l4_noop_scaffold_trace(config)
    return (
        {
            "generation_completed": True,
            "generated_token_ids": list(TOKENS),
            "generated_text": "out",
            "exactkv_failures": 0,
            "blockers": [],
        },
        trace,
    )


def test_panel_schema_validates() -> None:
    report = build_synthetic_exp103_panel_report()
    assert report["experiment_id"] == EXPERIMENT_103_ID
    result = validate_exp103_panel_report(report)
    assert result.valid is True


def test_cell_schema_validates() -> None:
    report = build_synthetic_exp103_panel_report(num_cells=1)
    cell = report["cells"][0]
    for key in (
        "model_id",
        "prompt_id",
        "compressor",
        "max_new_tokens",
        "baseline_completed",
        "noop_scaffold_completed",
        "token_match",
        "text_match",
        "safety_gates",
    ):
        assert key in cell


def test_aggregation_counts_token_text_parity_correctly() -> None:
    report = run_exp103_l4_noop_scaffold_panel_validation(
        model_ids=["m1"],
        prompts=[("p0", "a"), ("p1", "b")],
        compressors_requested=["noop", "int8"],
        max_new_tokens_values=[4, 8],
        baseline_generation_fn=_baseline_fn,
        noop_scaffold_generation_fn=_noop_fn,
    )
    assert report["total_cells"] == 8
    assert report["token_match_cells"] == 8
    assert report["text_match_cells"] == 8
    assert report["successful_baseline_cells"] == 8
    assert report["successful_noop_scaffold_cells"] == 8


def test_safety_gate_failure_fails_report() -> None:
    report = build_synthetic_exp103_panel_report()
    report["l4_runtime_commit_implemented"] = True
    assert validate_exp103_panel_report(report).valid is False


def test_token_mismatch_fails_report() -> None:
    report = build_synthetic_exp103_panel_report(unsafe=True)
    assert report["status"] == "failed"
    assert validate_exp103_panel_report(report).valid is False


def test_model_blocked_behavior() -> None:
    report = run_exp103_l4_noop_scaffold_panel_validation(
        model_ids=["blocked-model"],
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        max_new_tokens_values=[4],
        allow_model_blocked=True,
        runtime_loader=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no model")),
    )
    assert report["status"] == "blocked"
    assert len(report["models_blocked"]) == 1
    assert report["models_loaded"] == []


def test_production_cli_modified_must_be_false() -> None:
    assert build_synthetic_exp103_panel_report()["production_cli_modified"] is False


def test_l4_runtime_commit_implemented_must_be_false() -> None:
    assert build_synthetic_exp103_panel_report()["l4_runtime_commit_implemented"] is False


def test_verifier_mediated_acceptance_performed_must_be_false() -> None:
    report = build_synthetic_exp103_panel_report()
    assert report["verifier_mediated_acceptance_performed"] is False


def test_proposal_used_for_token_commit_must_be_false() -> None:
    assert build_synthetic_exp103_panel_report()["proposal_used_for_token_commit"] is False


def test_proposal_exposed_to_generator_must_be_false() -> None:
    assert build_synthetic_exp103_panel_report()["proposal_exposed_to_generator"] is False


def test_validation_passes_for_safe_synthetic_panel() -> None:
    report = build_synthetic_exp103_panel_report()
    assert report["validation_result"]["valid"] is True


def test_validation_fails_for_unsafe_synthetic_panel() -> None:
    report = build_synthetic_exp103_panel_report(unsafe=True)
    assert report["validation_result"]["valid"] is False


def test_allowed_next_phase_is_phase21c_trace_dry_run_design() -> None:
    report = build_synthetic_exp103_panel_report()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21B
    assert report["allowed_next_phase"] == "phase21c_l4_trace_only_dry_run_design"


def test_forbidden_next_phase_is_runtime_commit() -> None:
    report = build_synthetic_exp103_panel_report()
    assert report["forbidden_next_phase"] == FORBIDDEN_NEXT_PHASE_21B
    assert report["forbidden_next_phase"] == "l4_runtime_commit_implementation"


def test_no_forbidden_positive_claims_in_report() -> None:
    report = build_synthetic_exp103_panel_report()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_breakdown_summaries_present() -> None:
    report = build_synthetic_exp103_panel_report(num_cells=4)
    assert report["breakdowns_by_model"]
    assert report["breakdowns_by_compressor"]
    assert report["breakdowns_by_prompt"]
    assert report["breakdowns_by_max_new_tokens"]
