"""Tests for Phase G unified truth + divergence authority engine."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.engine.unified_truth_engine import (
    DIVERGENCE_TYPE_LENGTH_DRIFT,
    DIVERGENCE_TYPE_NONE,
    DIVERGENCE_TYPE_TOKEN_MISMATCH,
    FAILURE_REGIME_COMPRESSION_BREAK,
    FAILURE_REGIME_STABLE,
    FirstDivergenceAuthority,
    build_unified_truth_records,
    run_phase_g_unified_truth_engine,
    validate_kernel_consistency,
    validate_phase_g_report,
    write_phase_g_outputs,
)


def test_first_divergence_authority_token_mismatch() -> None:
    auth = FirstDivergenceAuthority()
    result = auth.compute(
        {"token_ids": [1, 2, 3, 4]},
        {"token_ids": [1, 9, 3, 4]},
    )
    assert result.canonical_first_divergence_index == 1
    assert result.divergence_type == DIVERGENCE_TYPE_TOKEN_MISMATCH
    assert result.token_exact_match is False


def test_first_divergence_authority_length_drift() -> None:
    auth = FirstDivergenceAuthority()
    result = auth.compute(
        {"token_ids": [1, 2, 3]},
        {"token_ids": [1, 2]},
    )
    assert result.canonical_first_divergence_index == 2
    assert result.divergence_type == DIVERGENCE_TYPE_LENGTH_DRIFT


def test_first_divergence_authority_no_divergence() -> None:
    auth = FirstDivergenceAuthority()
    result = auth.compute(
        {"token_ids": [10, 20]},
        {"token_ids": [10, 20]},
    )
    assert result.canonical_first_divergence_index is None
    assert result.divergence_type == DIVERGENCE_TYPE_NONE
    assert result.token_exact_match is True


def test_kernel_consistency_from_phase_f_fixture() -> None:
    phase_f = {
        "phase_e_id": "phaseE_kv_compression_kernel",
        "phase_f_id": "phaseF_triton_kv_compression_kernel",
        "kv_shape": [1, 8, 512, 64],
        "benchmarks": [
            {
                "backend": "torch",
                "mode": "int8",
                "status": "ok",
                "latency_ms": 0.1,
                "compression_ratio": 0.265625,
                "memory_before": 100,
                "memory_after": 26,
                "execution_backend": "torch",
            },
            {
                "backend": "triton",
                "mode": "int8",
                "status": "ok",
                "latency_ms": 0.07,
                "compression_ratio": 0.265625,
                "memory_before": 100,
                "memory_after": 26,
                "execution_backend": "triton",
            },
        ],
    }
    report = validate_kernel_consistency(phase_f, phase_f)
    assert report.overall_consistent is True
    assert report.modes[0]["token_equivalence_ratio"] == 1.0
    assert report.modes[0]["memory_consistent"] is True


def test_kernel_consistency_detects_memory_mismatch() -> None:
    phase_f = {
        "kv_shape": [1, 8, 128, 64],
        "benchmarks": [
            {
                "backend": "torch",
                "mode": "int4",
                "status": "ok",
                "compression_ratio": 0.25,
                "memory_before": 100,
                "memory_after": 25,
            },
            {
                "backend": "triton",
                "mode": "int4",
                "status": "ok",
                "compression_ratio": 0.30,
                "memory_before": 100,
                "memory_after": 30,
            },
        ],
    }
    report = validate_kernel_consistency(phase_f, phase_f)
    assert report.overall_consistent is False
    assert report.modes[0]["consistent"] is False


def test_deterministic_unified_records_from_disk() -> None:
    phase_a_path = Path("reports/phaseA_benchmark.json")
    if not phase_a_path.exists():
        return
    report_a = run_phase_g_unified_truth_engine()
    report_b = run_phase_g_unified_truth_engine()
    assert report_a["source_totals"] == report_b["source_totals"]
    assert report_a["failure_regime_counts"] == report_b["failure_regime_counts"]
    ids_a = [r["record_id"] for r in report_a["records"]]
    ids_b = [r["record_id"] for r in report_b["records"]]
    assert ids_a == ids_b


def test_run_phase_g_pipeline(tmp_path: Path) -> None:
    phase_a_path = Path("reports/phaseA_benchmark.json")
    if not phase_a_path.exists():
        return
    report = run_phase_g_unified_truth_engine(
        divergence_map_path=tmp_path / "phaseG_divergence_map.png",
    )
    assert report["status"] == "unified_truth_complete"
    assert report["source_totals"]["unified_records"] == 336
    assert report["kernel_consistency"]["overall_consistent"] is True
    assert validate_phase_g_report(report).valid


def test_no_hallucinated_metrics_in_records() -> None:
    phase_a_path = Path("reports/phaseA_benchmark.json")
    if not phase_a_path.exists():
        return
    report = run_phase_g_unified_truth_engine()
    for rec in report["records"]:
        da = rec["divergence_authority"]
        assert "canonical_first_divergence_index" in da
        assert da["divergence_type"] in {
            "none",
            "token_mismatch",
            "length_drift",
            "kernel_inconsistency",
            "verifier_disagreement",
        }
        pa = rec["phase_a"]
        assert "metrics" in pa
        if rec["phase_d"] is not None:
            assert "divergence_metrics" in rec["phase_d"]


def test_write_phase_g_outputs(tmp_path: Path) -> None:
    phase_a_path = Path("reports/phaseA_benchmark.json")
    if not phase_a_path.exists():
        return
    report = run_phase_g_unified_truth_engine(
        divergence_map_path=tmp_path / "map.png",
    )
    paths = write_phase_g_outputs(
        report,
        truth_path=tmp_path / "truth.json",
        kernel_path=tmp_path / "kernel.json",
    )
    assert paths["phaseG_unified_truth"].exists()
    assert paths["phaseG_kernel_consistency"].exists()
    kernel = json.loads(paths["phaseG_kernel_consistency"].read_text())
    assert "modes" in kernel
    assert kernel["overall_consistent"] is True


def test_build_unified_truth_records_regime_classification() -> None:
    phase_a = {
        "cells": [
            {
                "model_name": "m",
                "compressor_name": "noop",
                "prompt_id": "p0",
                "max_new_tokens": 4,
                "exactkv_failure": False,
                "full": {"output_ids": [1, 2, 3]},
                "exactkv": {"output_ids": [1, 2, 3]},
                "metrics": {
                    "token_level_divergence": False,
                    "first_divergence_index": None,
                    "acceptance_rate": 1.0,
                },
            },
            {
                "model_name": "m",
                "compressor_name": "int4_sim",
                "prompt_id": "p0",
                "max_new_tokens": 4,
                "exactkv_failure": False,
                "full": {"output_ids": [1, 2, 3]},
                "exactkv": {"output_ids": [1, 9, 3]},
                "metrics": {
                    "token_level_divergence": True,
                    "first_divergence_index": 1,
                    "acceptance_rate": 0.5,
                },
            },
        ],
    }
    phase_d = {"cells": []}
    phase_f = {
        "kv_shape": [1, 8, 128, 64],
        "benchmarks": [
            {
                "backend": "torch",
                "mode": "int4",
                "status": "ok",
                "compression_ratio": 0.25,
                "memory_before": 10,
                "memory_after": 2,
            },
            {
                "backend": "triton",
                "mode": "int4",
                "status": "ok",
                "compression_ratio": 0.25,
                "memory_before": 10,
                "memory_after": 2,
                "execution_backend": "triton",
            },
        ],
    }
    leaderboard = {"entries": []}
    records = build_unified_truth_records(phase_a, phase_d, phase_f, leaderboard)
    regimes = {r.compressor_name: r.failure_regime for r in records}
    assert regimes["noop"] == FAILURE_REGIME_STABLE
    assert regimes["int4_sim"] == FAILURE_REGIME_COMPRESSION_BREAK
