"""Tests for generation-shadow wiring review (Phase 16J)."""
from __future__ import annotations

from pathlib import Path

from exactkv.attention.generation_shadow_review import (
    DEFAULT_RUNTIME_UNCHANGED_CLAIM,
    EXPERIMENT_075_ID,
    SHADOW_FORBIDDEN_CLAIMS,
    ShadowLevelId,
    SAFETY_GATES,
    build_hook_point_review,
    inspect_codebase,
    recommend_next_level,
    run_exp075_generation_shadow_review,
    validate_exp075_report,
    _base_shadow_levels,
)


def test_shadow_level_schema() -> None:
    report = run_exp075_generation_shadow_review()
    assert validate_exp075_report(report) == []
    levels = report["shadow_levels"]
    assert len(levels) == 5
    ids = {lvl["level_id"] for lvl in levels}
    assert ids == {e.value for e in ShadowLevelId}
    for lvl in levels:
        assert "allowed_claims" in lvl
        assert "forbidden_claims" in lvl
        assert "implementation_status" in lvl


def test_l1_generation_observer_safety_gates() -> None:
    report = run_exp075_generation_shadow_review()
    gate_ids = {g["gate_id"] for g in report["safety_gates"]}
    assert "opt_in_only" in gate_ids
    assert "generated_tokens_unaffected" in gate_ids
    assert "no_streaming_token_commit" in gate_ids
    assert len(report["safety_gates"]) >= len(SAFETY_GATES)


def test_l2_l3_l4_not_implemented() -> None:
    levels = {lvl.level_id: lvl for lvl in _base_shadow_levels()}
    assert levels[ShadowLevelId.L2_DRAFT_SHADOW.value].implementation_status == "not_implemented"
    assert levels[ShadowLevelId.L3_RESTORED_VERIFIER_SHADOW.value].implementation_status == "not_implemented"
    assert levels[ShadowLevelId.L4_RUNTIME_INTEGRATION.value].implementation_status == "forbidden_for_now"


def test_forbidden_claims_present() -> None:
    report = run_exp075_generation_shadow_review()
    forbidden = set(report["forbidden_claims"])
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "active_gpu_memory_savings",
        "production_memory_savings",
    ):
        assert term in forbidden


def test_default_runtime_unchanged_claim() -> None:
    report = run_exp075_generation_shadow_review()
    assert "unchanged" in report["default_runtime_unchanged_claim"].lower()
    assert DEFAULT_RUNTIME_UNCHANGED_CLAIM in report["default_runtime_unchanged_claim"]


def test_recommended_next_level_is_l1_when_hooks_present() -> None:
    inspection = inspect_codebase()
    hook = build_hook_point_review(inspection)
    level, _phase = recommend_next_level(hook, shadow_levels=_base_shadow_levels())
    if hook["shadow_comparison_sufficient"]:
        assert level == ShadowLevelId.L1_GENERATION_OBSERVER.value
    else:
        assert level == ShadowLevelId.L0_OFFLINE_REPLAY.value


def test_recommended_next_level_l1_on_synthetic_inspection() -> None:
    hook = {
        "shadow_comparison_sufficient": True,
    }
    level, phase = recommend_next_level(hook, shadow_levels=_base_shadow_levels())
    assert level == ShadowLevelId.L1_GENERATION_OBSERVER.value
    assert "16K" in phase


def test_blocked_review_with_missing_paths(tmp_path: Path) -> None:
    report = run_exp075_generation_shadow_review(
        root=tmp_path,
        inspect_paths=["nonexistent/file.py"],
    )
    assert report["files_missing"] == ["nonexistent/file.py"]
    assert validate_exp075_report(report) == []


def test_no_forbidden_positive_claims_in_report_blob() -> None:
    report = run_exp075_generation_shadow_review()
    blob = str(report).lower()
    for term in ("tokens_per_second", "active_gpu_memory_savings", "production_memory_savings"):
        assert term in blob  # listed under forbidden_claims


def test_l0_implemented_baseline() -> None:
    levels = {lvl.level_id: lvl for lvl in _base_shadow_levels()}
    assert levels[ShadowLevelId.L0_OFFLINE_REPLAY.value].implementation_status == "implemented"
