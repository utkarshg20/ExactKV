"""Tests for ExactKV canonical leaderboard platform (Phase B)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.benchmarks.leaderboard_platform import (
    CANONICAL_COMPRESSORS,
    LEADERBOARD_ID,
    compute_leaderboard_score,
    generate_insights,
    load_phase_a_report,
    normalize_first_divergence,
    normalize_model_compressor_metrics,
    rank_leaderboard_rows,
    render_leaderboard_markdown,
    run_leaderboard_platform,
    validate_leaderboard_report,
    write_leaderboard_outputs,
)


@pytest.fixture
def phase_a_report() -> dict:
    return load_phase_a_report(Path("reports/phaseA_benchmark.json"))


def test_load_phase_a(phase_a_report: dict) -> None:
    assert phase_a_report["phase_id"] == "phaseA_scale_benchmark"
    assert phase_a_report["total_cells"] > 0


def test_normalize_all_model_compressor_pairs(phase_a_report: dict) -> None:
    rows = normalize_model_compressor_metrics(phase_a_report)
    models = phase_a_report["models_evaluated"]
    assert len(rows) == len(models) * len(CANONICAL_COMPRESSORS)
    available = [r for r in rows if r["availability"] != "unavailable"]
    assert len(available) == len(rows)


def test_scoring_function_bounds() -> None:
    score = compute_leaderboard_score(
        acceptance_rate=1.0,
        verifier_agreement=1.0,
        first_divergence_normalized=1.0,
        exactkv_failure_rate=0.0,
        stability_score=1.0,
    )
    assert score == pytest.approx(1.0)
    low = compute_leaderboard_score(
        acceptance_rate=0.0,
        verifier_agreement=0.0,
        first_divergence_normalized=0.0,
        exactkv_failure_rate=1.0,
        stability_score=0.0,
    )
    assert low == pytest.approx(0.0)


def test_first_divergence_normalization() -> None:
    assert normalize_first_divergence(None) == 1.0
    assert normalize_first_divergence(8.0, max_new_tokens=16) == pytest.approx(0.5)


def test_run_leaderboard_platform(phase_a_report: dict, tmp_path: Path) -> None:
    report = run_leaderboard_platform(
        phase_a_path=Path("reports/phaseA_benchmark.json"),
    )
    assert report["leaderboard_id"] == LEADERBOARD_ID
    assert validate_leaderboard_report(report).valid
    assert len(report["insights"]) >= 1
    ranked = [e for e in report["entries"] if e.get("rank") is not None]
    assert len(ranked) == len(phase_a_report["models_evaluated"]) * len(CANONICAL_COMPRESSORS)


def test_filter_model(phase_a_report: dict) -> None:
    report = run_leaderboard_platform(
        phase_a_path=Path("reports/phaseA_benchmark.json"),
        filter_model="0.5B-Instruct",
    )
    assert all("Instruct" in e["model"] for e in report["entries"])


def test_filter_compressor() -> None:
    report = run_leaderboard_platform(
        phase_a_path=Path("reports/phaseA_benchmark.json"),
        filter_compressor="int8",
    )
    assert all(e["compressor"] == "int8" for e in report["entries"])


def test_write_outputs(tmp_path: Path) -> None:
    report = run_leaderboard_platform(phase_a_path=Path("reports/phaseA_benchmark.json"))
    json_path = tmp_path / "leaderboard.json"
    md_path = tmp_path / "leaderboard.md"
    write_leaderboard_outputs(report, json_path=json_path, markdown_path=md_path)
    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["leaderboard_id"] == LEADERBOARD_ID
    assert "# ExactKV Canonical Leaderboard" in md_path.read_text()


def test_insights_count(phase_a_report: dict) -> None:
    rows = normalize_model_compressor_metrics(phase_a_report)
    ranked = rank_leaderboard_rows(rows)
    from exactkv.benchmarks.leaderboard_platform import aggregate_global_compressor_rankings

    insights = generate_insights(ranked, global_rankings=aggregate_global_compressor_rankings(ranked))
    assert 1 <= len(insights) <= 5


def test_no_runtime_flags() -> None:
    report = run_leaderboard_platform(phase_a_path=Path("reports/phaseA_benchmark.json"))
    assert report["exactkv_generator_modified"] is False
    assert report["runtime_commit_authorized"] is False
    assert report["trace_only"] is True
