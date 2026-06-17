"""Tests for Experiment 085 Phase 16 closeout summary (Phase 16T)."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.attention.phase16_closeout import (
    ALLOWED_CLAIMS,
    EXPERIMENT_085_ID,
    FORBIDDEN_CLAIMS,
    PHASE16_COMPLETED_STEPS,
    RECOMMENDED_NEXT_PHASE,
    run_exp085_phase16_closeout_summary,
    validate_exp085_report,
)


def _run_closeout(root: Path, **kwargs: object) -> dict:
    return run_exp085_phase16_closeout_summary(root=root, **kwargs)


def test_summary_schema_validates(tmp_path: Path) -> None:
    # Minimal repo layout: docs only
    for entry in (
        "docs/EXPERIMENT_066_STREAMING_QUANT_ATTENTION_FEASIBILITY.md",
        "docs/PHASE_16_CLOSEOUT.md",
        "docs/CLAIMS_AUDIT.md",
        "docs/DEFERRED_WORK_REGISTER.md",
        "docs/VERICACHE_SYSTEMS_ROADMAP.md",
        "docs/EXPERIMENT_085_PHASE16_CLOSEOUT_SUMMARY.md",
    ):
        p = tmp_path / entry
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n")

    report = _run_closeout(tmp_path)
    assert report["experiment_id"] == EXPERIMENT_085_ID
    assert validate_exp085_report(report) == []


def test_missing_reports_handled(tmp_path: Path) -> None:
    report = _run_closeout(tmp_path)
    assert len(report["reports_missing"]) == 19
    assert report["reports_found"] == []
    assert report["status"] in ("complete", "complete_with_missing_reports", "partial")


def test_found_reports_handled(tmp_path: Path) -> None:
    report_path = tmp_path / "reports/experiment_084_guarded_decode_time_shadow_panel.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({
            "experiment_id": "exp084_guarded_decode_time_shadow_panel",
            "status": "diagnostic_complete",
            "exactkv_failure_summary": {"baseline_failures": 0, "guarded_failures": 0},
            "safety_gate_summary": {"cells_all_gates_ok": 32, "cells_with_gate_failure": 0},
            "decode_time_shadow_used_for_token_commit": False,
        }),
    )
    report = _run_closeout(tmp_path)
    assert "reports/experiment_084_guarded_decode_time_shadow_panel.json" in report["reports_found"]
    assert report["exactkv_failure_summary"]["panels_with_data"] >= 1


def test_claim_freeze_includes_allowed_forbidden_future(tmp_path: Path) -> None:
    report = _run_closeout(tmp_path)
    cf = report["claim_freeze"]
    assert set(ALLOWED_CLAIMS).issubset(set(cf["allowed_claims"]))
    assert set(FORBIDDEN_CLAIMS).issubset(set(cf["forbidden_claims"]))
    assert len(cf["future_deferred_claims"]) >= 5


def test_recommended_stop_and_phase17(tmp_path: Path) -> None:
    report = _run_closeout(tmp_path)
    assert report["recommended_stop"] is True
    assert report["recommended_next_phase"] == RECOMMENDED_NEXT_PHASE
    assert report["phase16_completed_steps"] == PHASE16_COMPLETED_STEPS


def test_topk_supplementary_only_note(tmp_path: Path) -> None:
    report = _run_closeout(tmp_path)
    note = report["topk_interpretation_note"].lower()
    assert "supplementary" in note
    assert "exactness guarantee" in note


def test_no_forbidden_positive_claim_phrases(tmp_path: Path) -> None:
    report = _run_closeout(tmp_path)
    # Check evidence cells only — top-level may list forbidden claims as negations
    for section in report["evidence_summary"].values():
        dumped = json.dumps(section).lower()
        for forbidden in (
            "throughput improvement",
            "latency improvement",
            "speedup",
            "tokens_per_second",
            "runtime_seconds",
            "active_gpu_memory_savings",
            "production_memory_savings",
            "production serving supported",
            "vericache throughput reproduced",
            "vericache serving reproduced",
        ):
            assert forbidden not in dumped


def test_phase16_status_complete(tmp_path: Path) -> None:
    report = _run_closeout(tmp_path)
    assert report["phase16_status"] == "complete"
    assert len(report["experiments_covered"]) == 19
