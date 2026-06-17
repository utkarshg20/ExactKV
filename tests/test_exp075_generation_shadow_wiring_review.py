"""Tests for Experiment 075 generation-shadow wiring review (Phase 16J)."""
from __future__ import annotations

from exactkv.attention.generation_shadow_review import (
    EXPERIMENT_075_ID,
    SHADOW_FORBIDDEN_CLAIMS,
    ShadowLevelId,
    run_exp075_generation_shadow_review,
    validate_exp075_report,
)


def test_exp075_report_validates() -> None:
    report = run_exp075_generation_shadow_review()
    assert report["experiment_id"] == EXPERIMENT_075_ID
    assert validate_exp075_report(report) == []


def test_exp075_recommends_l1() -> None:
    report = run_exp075_generation_shadow_review()
    assert report["recommended_next_level"] == ShadowLevelId.L1_GENERATION_OBSERVER.value


def test_exp075_hook_point_review_fields() -> None:
    report = run_exp075_generation_shadow_review()
    hook = report["hook_point_review"]
    assert "prompt_entry_points" in hook
    assert "generation_output_points" in hook
    assert "minimal_future_cli_flag" in hook
    assert hook["minimal_future_cli_flag"] == "--generation-shadow-observer"


def test_exp075_l1_not_implemented_status() -> None:
    report = run_exp075_generation_shadow_review()
    l1 = next(l for l in report["shadow_levels"] if l["level_id"] == ShadowLevelId.L1_GENERATION_OBSERVER.value)
    assert l1["implementation_status"] == "not_implemented"
    assert l1["recommended_or_not"] == "recommended_next"


def test_exp075_forbidden_claims_listed() -> None:
    report = run_exp075_generation_shadow_review()
    assert "throughput" in report["forbidden_claims"]
    assert all(c in SHADOW_FORBIDDEN_CLAIMS for c in ("latency", "speedup"))
