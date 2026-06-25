"""Tests for Phase A scale benchmarking."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exactkv.benchmarks.phase_a_scale_benchmark import (
    PHASE_A_ALL_COMPRESSORS,
    PHASE_A_ID,
    PHASE_A_MAX_NEW_TOKENS,
    PHASE_A_MODELS,
    build_compressor_rankings,
    build_deterministic_phase_a_cell,
    default_phase_a_prompts,
    render_phase_a_markdown_summary,
    run_phase_a_scale_benchmark,
    validate_phase_a_report,
    write_phase_a_outputs,
)


def test_deterministic_benchmark_produces_valid_report(tmp_path: Path) -> None:
    report = run_phase_a_scale_benchmark(deterministic_mode=True)
    validation = validate_phase_a_report(report)
    assert validation.valid, validation.errors
    assert report["phase_id"] == PHASE_A_ID
    assert report["exactkv_generator_modified"] is False
    assert report["runtime_commit_authorized"] is False


def test_expected_cell_count_deterministic() -> None:
    report = run_phase_a_scale_benchmark(deterministic_mode=True)
    expected = (
        len(PHASE_A_MODELS)
        * len(default_phase_a_prompts())
        * len(PHASE_A_ALL_COMPRESSORS)
        * len(PHASE_A_MAX_NEW_TOKENS)
    )
    assert report["total_cells"] == expected
    assert report["expected_cells"] == expected


def test_all_compressors_in_summary() -> None:
    report = run_phase_a_scale_benchmark(deterministic_mode=True)
    for comp in PHASE_A_ALL_COMPRESSORS:
        assert comp in report["compressor_summary"]
        assert report["compressor_summary"][comp]["num_cells"] > 0


def test_rankings_present() -> None:
    report = run_phase_a_scale_benchmark(deterministic_mode=True)
    rankings = report["compressor_rankings"]
    for key in ("by_acceptance_rate", "by_divergence_stability", "by_failure_rate"):
        assert key in rankings
        assert len(rankings[key]) == len(PHASE_A_ALL_COMPRESSORS)


def test_per_model_tables() -> None:
    report = run_phase_a_scale_benchmark(deterministic_mode=True)
    assert len(report["per_model_tables"]) == len(PHASE_A_MODELS)
    for model in PHASE_A_MODELS:
        assert model in report["per_model_tables"]


def test_cell_metrics_extracted() -> None:
    cell = build_deterministic_phase_a_cell(
        model_name=PHASE_A_MODELS[0],
        prompt_entry=default_phase_a_prompts()[0],
        compressor_name="int4_sim",
        max_new_tokens=8,
    )
    metrics = cell["metrics"]
    assert "acceptance_rate" in metrics
    assert "first_divergence_index" in metrics or metrics["token_level_divergence"] is False
    assert "verifier_agreement_score" in metrics
    assert "exactkv_failure" in metrics


def test_write_outputs(tmp_path: Path) -> None:
    report = run_phase_a_scale_benchmark(deterministic_mode=True)
    json_path = tmp_path / "phaseA_benchmark.json"
    md_path = tmp_path / "phaseA_benchmark.md"
    write_phase_a_outputs(report, json_path=json_path, markdown_path=md_path)
    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["phase_id"] == PHASE_A_ID
    assert "# Phase A Scale Benchmark Summary" in md_path.read_text()


def test_ranking_ordering() -> None:
    summary = {
        "noop": {"mean_acceptance_rate": 0.9, "divergence_stability_score": 0.95, "exactkv_failure_rate": 0.0},
        "int4_sim": {"mean_acceptance_rate": 0.5, "divergence_stability_score": 0.4, "exactkv_failure_rate": 0.1},
    }
    rankings = build_compressor_rankings(summary)
    assert rankings["by_acceptance_rate"][0]["compressor"] == "noop"
    assert rankings["by_failure_rate"][0]["compressor"] == "noop"


def test_markdown_renders() -> None:
    report = run_phase_a_scale_benchmark(deterministic_mode=True)
    md = render_phase_a_markdown_summary(report)
    assert "noop" in md
    assert "Reproducibility" in md
