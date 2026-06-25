"""Tests for Experiment 111 L4 verifier runtime instrumentation design (Phase 21J)."""
from __future__ import annotations

import json

from exactkv.safety.l4_verifier_runtime_instrumentation_design import (
    ARCHITECTURE_DIAGRAM,
    DATA_FLOW_STEP_IDS,
    DESIGN_OUTCOME_COMPLETE,
    EXPERIMENT_111_ID,
    FAILURE_MODE_IDS,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_21J,
    INCORRECT_ENABLEMENT_SCENARIO_IDS,
    INSTRUMENTATION_POINT_IDS,
    INTEGRATION_POINT_IDS,
    RECOMMENDED_NEXT_PHASE_21J,
    RUNTIME_HOOK_IDS,
    build_l4_data_flow_steps,
    build_l4_incorrect_enablement_scenarios,
    build_l4_instrumentation_failure_modes,
    build_l4_instrumentation_integration_points,
    build_l4_instrumentation_points,
    build_l4_runtime_hook_definitions,
    build_l4_runtime_instrumentation_design,
    build_l4_safety_boundaries,
    run_exp111_l4_verifier_runtime_instrumentation_design,
    validate_exp111_report,
)


def test_runtime_hooks_cover_required_ids() -> None:
    hooks = build_l4_runtime_hook_definitions()
    assert set(RUNTIME_HOOK_IDS) <= {h.hook_id for h in hooks}
    assert all(not h.implemented for h in hooks)


def test_instrumentation_points_cover_required_ids() -> None:
    points = build_l4_instrumentation_points()
    assert set(INSTRUMENTATION_POINT_IDS) <= {p.point_id for p in points}
    per_token = next(p for p in points if p.point_id == "per_token")
    assert "NOT IMPLEMENTED" in per_token.description or "MUST NOT" in per_token.description


def test_data_flow_steps_not_executed_at_runtime() -> None:
    steps = build_l4_data_flow_steps()
    assert set(DATA_FLOW_STEP_IDS) <= {s.step_id for s in steps}
    assert all(not s.executed_at_runtime for s in steps)


def test_safety_boundaries_present() -> None:
    boundaries = build_l4_safety_boundaries()
    assert len(boundaries) >= 5
    ids = {b.boundary_id for b in boundaries}
    assert "default_runtime_unchanged" in ids
    assert "verifier_non_authoritative_until_l4" in ids


def test_integration_points_cover_subsystems() -> None:
    integrations = build_l4_instrumentation_integration_points()
    assert set(INTEGRATION_POINT_IDS) <= {i.integration_id for i in integrations}
    assert all(not i.changes_allowed_in_phase_21j for i in integrations)


def test_failure_modes_cover_required_ids() -> None:
    failures = build_l4_instrumentation_failure_modes()
    assert set(FAILURE_MODE_IDS) <= {f.failure_id for f in failures}
    assert all(f.blocks_commit for f in failures)


def test_incorrect_enablement_scenarios_present() -> None:
    scenarios = build_l4_incorrect_enablement_scenarios()
    assert set(INCORRECT_ENABLEMENT_SCENARIO_IDS) <= {s.scenario_id for s in scenarios}


def test_architecture_diagram_references_generator() -> None:
    assert "ExactKVGenerator" in ARCHITECTURE_DIAGRAM
    assert "proposal" in ARCHITECTURE_DIAGRAM.lower()
    assert "rollback" in ARCHITECTURE_DIAGRAM.lower()


def test_design_outcome_complete() -> None:
    design = build_l4_runtime_instrumentation_design()
    assert design.decision.outcome == DESIGN_OUTCOME_COMPLETE


def test_report_schema_validates() -> None:
    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    assert report["experiment_id"] == EXPERIMENT_111_ID
    assert validate_exp111_report(report) == []


def test_design_outcome_in_report() -> None:
    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    assert report["design_outcome"] == DESIGN_OUTCOME_COMPLETE


def test_runtime_instrumentation_not_authorized() -> None:
    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    assert report["runtime_instrumentation_authorized"] is False
    assert report["runtime_instrumentation_implementation_authorized"] is False


def test_runtime_commit_authorized_false() -> None:
    assert run_exp111_l4_verifier_runtime_instrumentation_design()["runtime_commit_authorized"] is False


def test_stage_3_dry_run_design_authorized() -> None:
    decision = run_exp111_l4_verifier_runtime_instrumentation_design()["design_decision"]
    assert decision["stage_3_dry_run_design_authorized"] is True


def test_exactkv_generator_modified_false() -> None:
    assert run_exp111_l4_verifier_runtime_instrumentation_design()["exactkv_generator_modified"] is False


def test_model_experiments_run_false() -> None:
    assert run_exp111_l4_verifier_runtime_instrumentation_design()["model_experiments_run"] is False


def test_allowed_next_phase() -> None:
    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_21J
    assert "phase21k" in report["allowed_next_phase"]


def test_forbidden_next_phases() -> None:
    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    assert set(FORBIDDEN_NEXT_PHASES_21J) <= set(report["forbidden_next_phases"])


def test_no_forbidden_positive_claims() -> None:
    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative


def test_incorrect_enablement_section_in_report() -> None:
    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    section = report.get("what_happens_if_instrumentation_enabled_incorrectly") or []
    assert len(section) == len(INCORRECT_ENABLEMENT_SCENARIO_IDS)
