"""Tests for Experiment 101 L4 integration plan review (Phase 20D)."""
from __future__ import annotations

import json

from exactkv.safety.l4_integration_plan_review import (
    DECISION_READY_STAGE_1,
    EXPERIMENT_101_ID,
    FORBIDDEN_CLAIM_PHRASES,
    FORBIDDEN_NEXT_PHASES_20D,
    L4_OPT_IN_FLAG,
    RECOMMENDED_NEXT_PHASE_20D,
    REQUIRED_RISK_IDS,
    STAGE_IDS,
    build_l4_integration_plan_review,
    evaluate_l4_integration_plan_decision,
    run_exp101_l4_integration_plan_review,
    validate_exp101_report,
)


def test_plan_schema_validates() -> None:
    report = run_exp101_l4_integration_plan_review()
    assert report["experiment_id"] == EXPERIMENT_101_ID
    assert validate_exp101_report(report) == []


def test_future_change_targets_include_generator_not_modified() -> None:
    report = run_exp101_l4_integration_plan_review()
    targets = {t["path"]: t for t in report["future_change_targets"]}
    assert "exactkv/runtime/exactkv_generator.py" in targets
    assert targets["exactkv/runtime/exactkv_generator.py"]["current_status"] == "not_modified"
    assert report["exactkv_generator_modified"] is False


def test_future_interfaces_are_defined() -> None:
    report = run_exp101_l4_integration_plan_review()
    names = {i["name"] for i in report["future_interfaces"]}
    assert "L4DraftProposalProvider" in names
    assert "L4FullVerifier" in names
    assert "L4AcceptanceDecision" in names
    assert all(not i["affects_token_commit_now"] for i in report["future_interfaces"])


def test_future_opt_in_flag_plan_disabled_by_default() -> None:
    flag = run_exp101_l4_integration_plan_review()["future_flag_plan"]
    assert flag["flag_name"] == L4_OPT_IN_FLAG
    assert flag["default_enabled"] is False
    assert flag["flag_implemented"] is False
    assert "experimental" in flag["required_warnings"]


def test_implementation_stages_stage_0_through_stage_4_exist() -> None:
    report = run_exp101_l4_integration_plan_review()
    stage_ids = {s["stage_id"] for s in report["future_implementation_stages"]}
    assert set(STAGE_IDS) <= stage_ids


def test_stage_1_is_noop_only() -> None:
    stages = {
        s["stage_id"]: s
        for s in run_exp101_l4_integration_plan_review()["future_implementation_stages"]
    }
    stage_1 = stages["stage_1_noop_opt_in_scaffold"]
    assert "no-op" in stage_1["description"].lower() or any(
        "no-op" in b.lower() for b in stage_1["allowed_behavior"]
    )
    assert any("commit" in b.lower() for b in stage_1["forbidden_behavior"])


def test_stage_4_remains_blocked() -> None:
    stages = {
        s["stage_id"]: s
        for s in run_exp101_l4_integration_plan_review()["future_implementation_stages"]
    }
    assert stages["stage_4_runtime_commit_candidate"]["blocked"] is True


def test_risk_register_includes_required_risks() -> None:
    report = run_exp101_l4_integration_plan_review()
    risk_ids = {r["risk_id"] for r in report["risk_register"]}
    assert set(REQUIRED_RISK_IDS) <= risk_ids


def test_decision_recommends_ready_for_stage_1() -> None:
    review = build_l4_integration_plan_review()
    decision = evaluate_l4_integration_plan_decision(review)
    assert decision.decision == DECISION_READY_STAGE_1
    report = run_exp101_l4_integration_plan_review()
    assert report["integration_plan_decision"]["decision"] == DECISION_READY_STAGE_1


def test_l4_runtime_commit_authorized_is_false() -> None:
    report = run_exp101_l4_integration_plan_review()
    assert report["l4_runtime_commit_authorized"] is False
    assert report["integration_plan_decision"]["l4_runtime_commit_authorized"] is False


def test_exactkv_generator_modified_is_false() -> None:
    assert run_exp101_l4_integration_plan_review()["exactkv_generator_modified"] is False


def test_default_runtime_changed_is_false() -> None:
    assert run_exp101_l4_integration_plan_review()["default_runtime_changed"] is False


def test_runtime_generation_path_modified_is_false() -> None:
    assert run_exp101_l4_integration_plan_review()["runtime_generation_path_modified"] is False


def test_cli_flag_implemented_is_false() -> None:
    assert run_exp101_l4_integration_plan_review()["cli_flag_implemented"] is False


def test_allowed_next_phase_is_phase21a_noop_scaffold() -> None:
    report = run_exp101_l4_integration_plan_review()
    assert report["allowed_next_phase"] == RECOMMENDED_NEXT_PHASE_20D
    assert report["allowed_next_phase"] == "phase21a_l4_noop_opt_in_scaffold"


def test_forbidden_next_phases_include_runtime_and_benchmarks() -> None:
    report = run_exp101_l4_integration_plan_review()
    forbidden = set(report["forbidden_next_phases"])
    assert "l4_runtime_commit_implementation" in forbidden
    assert "cuda_backend" in forbidden
    assert "vllm_integration" in forbidden
    assert "lmcache_integration" in forbidden
    assert "performance_benchmark" in forbidden
    assert "memory_benchmark" in forbidden
    assert forbidden <= set(FORBIDDEN_NEXT_PHASES_20D)


def test_no_forbidden_positive_claims_in_report() -> None:
    report = run_exp101_l4_integration_plan_review()
    narrative = json.dumps(
        {
            "limitations": report.get("limitations"),
            "no_performance_claims_note": report.get("no_performance_claims_note"),
        },
    ).lower()
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        assert phrase.lower() not in narrative
