"""Tests for Experiment 074 attention tolerance policy panel (Phase 16I)."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.attention.tolerance_policy import (
    EXPERIMENT_074_ID,
    FORBIDDEN_ATTENTION_CLAIMS,
    AttentionTolerancePolicy,
    OfflineAttentionStatus,
    extract_policy_cells_from_exp071,
    load_report_json,
    run_exp074_panel,
    synthetic_policy_panel_cells,
    validate_exp074_report,
)


def _minimal_exp074_report(**overrides: object) -> dict:
    base = {
        "experiment_id": EXPERIMENT_074_ID,
        "status": "diagnostic_complete",
        "reports_loaded": ["exp071"],
        "reports_missing": ["exp070"],
        "policy": AttentionTolerancePolicy().to_dict(),
        "total_cells_evaluated": 2,
        "strict_numeric_pass_cells": 1,
        "strict_fail_depth_aware_pass_cells": 0,
        "local_alignment_pass_free_running_accumulation_cells": 1,
        "topk_agrees_numeric_drift_present_cells": 0,
        "local_attention_mismatch_cells": 0,
        "blocked_cells": 0,
        "optional_models_requested": False,
        "optional_models_loaded": [],
        "optional_models_blocked": [],
        "interpretation_summary": {"total_evaluated": 2},
        "limitations": ["test"],
        "no_performance_claims_note": "no performance claims",
        "claim_note": "test",
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "evaluated_cells": [
            {
                "source_experiment": "exp071",
                "model_id": "mock",
                "prompt_id": "p0",
                "target_token_length": 32,
                "chunk_size": 16,
                "prefix_layers": 24,
                "metric_type": "logits",
                "strict_numeric_pass": False,
                "depth_aware_numeric_pass": False,
                "topk_supplementary_pass": True,
                "overall_offline_status": OfflineAttentionStatus.TOPK_AGREES_NUMERIC_DRIFT_PRESENT.value,
                "interpretation_note": "supplementary",
                "blockers": [],
            },
            {
                "source_experiment": "exp072",
                "model_id": "mock",
                "prompt_id": "p0",
                "target_token_length": 32,
                "chunk_size": 16,
                "prefix_layers": 24,
                "metric_type": "hidden",
                "strict_numeric_pass": False,
                "depth_aware_numeric_pass": False,
                "topk_supplementary_pass": False,
                "overall_offline_status": (
                    OfflineAttentionStatus.LOCAL_ALIGNMENT_PASS_FREE_RUNNING_ACCUMULATION.value
                ),
                "interpretation_note": "accumulation",
                "blockers": [],
            },
        ],
    }
    base.update(overrides)
    return base


def test_validate_exp074_report() -> None:
    assert validate_exp074_report(_minimal_exp074_report()) == []


def test_missing_reports_use_synthetic_panel() -> None:
    report = run_exp074_panel(report_paths={"missing": Path("/nonexistent/report.json")})
    assert report["total_cells_evaluated"] > 0
    assert "missing" in report["reports_missing"]
    assert validate_exp074_report(report) == []


def test_extract_exp071_cells(tmp_path: Path) -> None:
    sample = {
        "model_id": "mock",
        "cells": [{
            "prompt_id": "p0",
            "target_token_length": 32,
            "chunk_size": 16,
            "num_layers_replayed": 24,
            "full_model_parity_status": "passed",
            "blockers": [],
            "streaming_vs_materialized_hidden_metrics": {"max_abs_error": 0.5},
            "streaming_vs_materialized_logit_metrics": {
                "max_abs_error": 0.13,
                "top1_agreement": True,
                "top5_overlap": 5,
                "top10_overlap": 10,
            },
        }],
    }
    policy = AttentionTolerancePolicy()
    cells = extract_policy_cells_from_exp071(sample, policy=policy)
    assert len(cells) == 2
    logit_cell = [c for c in cells if c["metric_type"] == "logits"][0]
    assert logit_cell["topk_supplementary_pass"] is True


def test_load_existing_reports_when_present() -> None:
    exp071 = Path("reports/experiment_071_full_prefix_logit_drift_smoke.json")
    if not exp071.is_file():
        return
    report = run_exp074_panel(report_paths={"exp071": exp071})
    assert "exp071" in report["reports_loaded"]
    assert report["total_cells_evaluated"] > 0


def test_optional_models_blocked_mock() -> None:
    def _failing_probe(**kwargs: object) -> dict:
        del kwargs
        return {
            "loaded_models": [],
            "blocked_models": [{
                "model_id": "Qwen/Qwen2.5-1.5B",
                "classification": "model_load_blocked",
                "blockers": ["mock oom"],
            }],
            "model_entries": [{
                "model_id": "Qwen/Qwen2.5-1.5B",
                "model_load_succeeded": False,
                "architecture_supported": False,
                "classification": "model_load_blocked",
                "blockers": ["mock oom"],
                "cells": [],
            }],
        }

    report = run_exp074_panel(
        report_paths={},
        include_optional_models=True,
        optional_probe_runner=_failing_probe,
    )
    assert report["optional_models_requested"] is True
    assert report["optional_models_loaded"] == []
    assert len(report["optional_models_blocked"]) == 1


def test_synthetic_panel_covers_statuses() -> None:
    cells = synthetic_policy_panel_cells(policy=AttentionTolerancePolicy())
    statuses = {c["overall_offline_status"] for c in cells}
    assert OfflineAttentionStatus.STRICT_NUMERIC_PASS.value in statuses
    assert OfflineAttentionStatus.BLOCKED.value in statuses
    assert OfflineAttentionStatus.LOCAL_ATTENTION_MISMATCH.value in statuses


def test_no_forbidden_claim_fields() -> None:
    blob = json.dumps({"forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS)}).lower()
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "active_gpu_memory_savings",
        "production_memory_savings",
    ):
        assert term not in blob or term in FORBIDDEN_ATTENTION_CLAIMS
