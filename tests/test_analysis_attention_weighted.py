"""Tests for exactkv.analysis.attention_weighted (V7 Phase A).

Gate: proxy divergence analysis gate.

Verifies:
  * Functions work on synthetic reports.
  * Divergence and rejection counts reconcile with source data.
  * Missing attention weights are handled honestly (proxy mode).
  * compare_reports_for_divergence works across named reports.
  * No forbidden performance fields in any output.
  * Old reports missing optional fields do not crash.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from analysis_fixtures import make_report, make_result

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORBIDDEN = {"tokens_per_second", "throughput", "latency", "speedup", "runtime_seconds"}
_REPORTS = Path(__file__).parent.parent / "reports"


def _assert_no_forbidden(obj, path="root"):
    if isinstance(obj, dict):
        hits = _FORBIDDEN & obj.keys()
        assert not hits, f"Forbidden fields at {path}: {hits}"
        for k, v in obj.items():
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")


@pytest.fixture
def mixed_report():
    return make_report(
        make_result(
            prompt_id="p1", compressor_name="noop",
            lossy_exact=True, first_div_idx=None,
            total_rejected=0, total_corrections=0, acceptance_rate=1.0,
        ),
        make_result(
            prompt_id="p2", compressor_name="int8",
            lossy_exact=False, first_div_idx=3,
            total_rejected=2, total_corrections=1, acceptance_rate=0.75,
        ),
        make_result(
            prompt_id="p3", compressor_name="int8",
            lossy_exact=False, first_div_idx=1,
            total_rejected=4, total_corrections=2, acceptance_rate=0.5,
        ),
        make_result(
            prompt_id="p4", compressor_name="int4_sim",
            lossy_exact=False, first_div_idx=0,
            total_rejected=10, total_corrections=5, acceptance_rate=0.2,
            is_simulated=True, supports_real_bytes_claim=False,
        ),
    )


@pytest.fixture
def minimal_legacy_report():
    """Old-style result missing optional capability fields."""
    return {
        "results": [{
            "prompt_id": "legacy_1",
            "compressor_name": "int8",
            "lossy": {
                "token_exact_match": False,
                "first_divergence_idx": 2,
            },
            "exactkv": {
                "acceptance": {
                    "total_rejected": 1,
                    "total_corrections": 1,
                    "acceptance_rate": 0.9,
                },
            },
        }],
    }


# ---------------------------------------------------------------------------
# Attention-weight honesty
# ---------------------------------------------------------------------------

def test_has_attention_weights_false_on_synthetic(mixed_report):
    from exactkv.analysis.attention_weighted import has_attention_weights
    assert has_attention_weights(mixed_report) is False


def test_proxy_metadata_labels_proxy_mode(mixed_report):
    from exactkv.analysis.attention_weighted import proxy_analysis_metadata
    meta = proxy_analysis_metadata(mixed_report)
    assert meta["has_attention_weights"] is False
    assert meta["analysis_type"] == "proxy_divergence"
    assert "No attention weights" in meta["note"] or "proxy" in meta["note"].lower()


def test_has_attention_weights_true_when_logged():
    from exactkv.analysis.attention_weighted import has_attention_weights
    report = {
        "results": [{
            "prompt_id": "p1",
            "compressor_name": "int8",
            "attention_entropy": [0.1, 0.5],
            "lossy": {"token_exact_match": True},
            "exactkv": {"acceptance": {}},
        }],
    }
    assert has_attention_weights(report) is True


# ---------------------------------------------------------------------------
# divergence_by_compressor
# ---------------------------------------------------------------------------

def test_divergence_by_compressor_reconciles(mixed_report):
    from exactkv.analysis.attention_weighted import divergence_by_compressor
    out = divergence_by_compressor(mixed_report)
    assert out["total_runs"] == 4
    assert out["lossy_divergence_count"] == 3
    assert out["by_compressor"]["noop"]["lossy_divergence_count"] == 0
    assert out["by_compressor"]["int8"]["lossy_divergence_count"] == 2
    assert out["by_compressor"]["int4_sim"]["divergence_rate"] == 1.0
    assert out["by_compressor"]["int8"]["mean_first_divergence_idx"] == 2.0
    _assert_no_forbidden(out)


# ---------------------------------------------------------------------------
# rejection_by_compressor
# ---------------------------------------------------------------------------

def test_rejection_by_compressor_reconciles(mixed_report):
    from exactkv.analysis.attention_weighted import rejection_by_compressor
    out = rejection_by_compressor(mixed_report)
    assert out["aggregate_rejected"] == 16  # 0+2+4+10
    assert out["aggregate_corrections"] == 8  # 0+1+2+5
    assert out["by_compressor"]["int8"]["total_rejected"] == 6
    assert out["by_compressor"]["int4_sim"]["total_corrections"] == 5
    _assert_no_forbidden(out)


# ---------------------------------------------------------------------------
# divergence_position_table
# ---------------------------------------------------------------------------

def test_divergence_position_table_buckets(mixed_report):
    from exactkv.analysis.attention_weighted import divergence_position_table
    out = divergence_position_table(mixed_report)
    noop_row = out["table"]["noop"]
    assert noop_row["no_divergence"] == 1
    int8_row = out["table"]["int8"]
    assert sum(int8_row.values()) == 2
    _assert_no_forbidden(out)


# ---------------------------------------------------------------------------
# acceptance_vs_divergence_summary
# ---------------------------------------------------------------------------

def test_acceptance_vs_divergence_summary_rows(mixed_report):
    from exactkv.analysis.attention_weighted import acceptance_vs_divergence_summary
    out = acceptance_vs_divergence_summary(mixed_report)
    rows = {r["compressor_name"]: r for r in out["rows"]}
    assert rows["noop"]["divergence_rate"] == 0.0
    assert rows["noop"]["mean_acceptance_rate"] == 1.0
    assert rows["int4_sim"]["divergence_rate"] == 1.0
    assert "causal attention importance" in out["note"]
    _assert_no_forbidden(out)


# ---------------------------------------------------------------------------
# compare_reports_for_divergence
# ---------------------------------------------------------------------------

def test_compare_reports_for_divergence_overlap(mixed_report):
    from exactkv.analysis.attention_weighted import compare_reports_for_divergence
    report_b = make_report(
        make_result(compressor_name="int8", lossy_exact=False, first_div_idx=5),
        make_result(compressor_name="noop", lossy_exact=True, first_div_idx=None),
    )
    out = compare_reports_for_divergence({
        "report_a": mixed_report,
        "report_b": report_b,
    })
    assert "report_a" in out["per_report"]
    assert "report_b" in out["per_report"]
    overlap = {o["compressor_name"]: o for o in out["overlapping_compressors"]}
    assert "int8" in overlap
    assert "noop" in overlap
    assert len(overlap["int8"]["divergence_rate_by_report"]) == 2
    _assert_no_forbidden(out)


# ---------------------------------------------------------------------------
# Legacy / real reports
# ---------------------------------------------------------------------------

def test_minimal_legacy_report_does_not_crash(minimal_legacy_report):
    from exactkv.analysis.attention_weighted import (
        acceptance_vs_divergence_summary,
        divergence_by_compressor,
        rejection_by_compressor,
    )
    divergence_by_compressor(minimal_legacy_report)
    rejection_by_compressor(minimal_legacy_report)
    acceptance_vs_divergence_summary(minimal_legacy_report)


@pytest.mark.skipif(
    not (_REPORTS / "experiment_003_asymmetric_kv_sweep.json").exists(),
    reason="experiment 003 report not present locally",
)
def test_real_experiment_003_no_attention_weights():
    from exactkv.analysis.attention_weighted import (
        divergence_by_compressor,
        has_attention_weights,
    )
    report = json.loads(
        (_REPORTS / "experiment_003_asymmetric_kv_sweep.json").read_text()
    )
    assert has_attention_weights(report) is False
    out = divergence_by_compressor(report)
    assert out["total_runs"] == 612
    assert out["lossy_divergence_count"] > 0
    _assert_no_forbidden(out)
