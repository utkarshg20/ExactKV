"""Tests for Experiment 102 L4 no-op opt-in scaffold (Phase 21A)."""
from __future__ import annotations

import json

import pytest

from exactkv.safety.l4_noop_opt_in_scaffold import (
    EXPERIMENT_102_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASE_21A,
    L4_OPT_IN_FLAG,
    MODE,
    RECOMMENDED_NEXT_PHASE_21A,
    STAGE,
    build_l4_noop_scaffold_trace,
    build_synthetic_exp102_report,
    default_l4_noop_opt_in_config,
    default_l4_noop_safety_gates,
    run_exp102_l4_noop_opt_in_scaffold,
    validate_exp102_report,
    validate_l4_noop_scaffold_report,
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


def test_config_defaults_disabled() -> None:
    config = default_l4_noop_opt_in_config()
    assert config.enabled is False
    assert config.stage == STAGE
    assert config.mode == MODE


def test_flag_name_is_correct() -> None:
    config = default_l4_noop_opt_in_config(enabled=True)
    assert config.flag_name == L4_OPT_IN_FLAG
    assert config.flag_name == "--experimental-l4-verifier-mediated-draft"


def test_mode_is_noop_trace_only() -> None:
    assert default_l4_noop_opt_in_config().mode == "noop_trace_only"


def test_noop_safety_gates_all_false_for_runtime_commit() -> None:
    gates = default_l4_noop_safety_gates(scaffold_enabled=True)
    assert gates.l4_runtime_commit_implemented is False
    assert gates.verifier_mediated_acceptance_performed is False
    assert gates.proposal_used_for_token_commit is False
    assert gates.proposal_exposed_to_generator is False
    assert gates.rollback_runtime_implemented is False
    assert gates.fallback_runtime_implemented is False
    assert gates.default_runtime_changed is False
    assert gates.generation_logic_changed is False
    assert gates.production_cli_modified is False
    assert gates.research_script_flag_only is True


def test_validation_passes_for_safe_synthetic_report() -> None:
    report = build_synthetic_exp102_report()
    result = validate_l4_noop_scaffold_report(report)
    assert result.valid is True
    assert validate_exp102_report(report) == []


def test_validation_fails_if_verifier_mediated_acceptance_performed() -> None:
    report = build_synthetic_exp102_report()
    report["verifier_mediated_acceptance_performed"] = True
    assert validate_l4_noop_scaffold_report(report).valid is False


def test_validation_fails_if_proposal_used_for_token_commit() -> None:
    report = build_synthetic_exp102_report()
    report["proposal_used_for_token_commit"] = True
    assert validate_l4_noop_scaffold_report(report).valid is False


def test_validation_fails_if_proposal_exposed_to_generator() -> None:
    report = build_synthetic_exp102_report()
    report["proposal_exposed_to_generator"] = True
    assert validate_l4_noop_scaffold_report(report).valid is False


def test_validation_fails_if_default_runtime_changed() -> None:
    report = build_synthetic_exp102_report()
    report["default_runtime_changed"] = True
    assert validate_l4_noop_scaffold_report(report).valid is False


def test_validation_fails_if_production_cli_modified() -> None:
    report = build_synthetic_exp102_report()
    report["production_cli_modified"] = True
    assert validate_l4_noop_scaffold_report(report).valid is False


def test_validation_fails_if_token_parity_fails() -> None:
    report = build_synthetic_exp102_report()
    report["cells"][0]["token_match"] = False
    report["cells"][0]["baseline_completed"] = True
    report["cells"][0]["noop_scaffold_completed"] = True
    assert validate_l4_noop_scaffold_report(report).valid is False


def test_report_schema_validates() -> None:
    report = run_exp102_l4_noop_opt_in_scaffold(
        prompts=[("p0", "hello")],
        compressors_requested=["noop"],
        baseline_generation_fn=_baseline_fn,
        noop_scaffold_generation_fn=_noop_fn,
    )
    assert report["experiment_id"] == EXPERIMENT_102_ID
    assert validate_exp102_report(report) == []


def test_panel_token_text_parity_with_injected_fns() -> None:
    report = run_exp102_l4_noop_opt_in_scaffold(
        prompts=[("p0", "hello"), ("p1", "world")],
        compressors_requested=["noop", "int8"],
        baseline_generation_fn=_baseline_fn,
        noop_scaffold_generation_fn=_noop_fn,
    )
    assert report["status"] == "scaffold_complete"
    assert report["token_match_cells"] == report["total_cells"]
    assert report["text_match_cells"] == report["total_cells"]


def test_allowed_next_phase_is_phase21b_panel_validation() -> None:
    report = build_synthetic_exp102_report()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21A
    assert report["allowed_next_phase"] == "phase21b_l4_noop_scaffold_panel_validation"


def test_forbidden_next_phase_is_runtime_commit() -> None:
    report = build_synthetic_exp102_report()
    assert report["forbidden_next_phase"] == FORBIDDEN_NEXT_PHASE_21A
    assert report["forbidden_next_phase"] == "l4_runtime_commit_implementation"


def test_no_forbidden_positive_claims_in_report() -> None:
    report = build_synthetic_exp102_report()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_exactkv_generator_modified_false() -> None:
    report = build_synthetic_exp102_report()
    assert report["exactkv_generator_modified"] is False


def test_l4_runtime_commit_not_implemented() -> None:
    report = build_synthetic_exp102_report()
    assert report["l4_runtime_commit_implemented"] is False
