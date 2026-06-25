"""Tests for Gate R0 release evidence integrity."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.platform.evidence_integrity import (
    EXPECTED_CELL_COUNT,
    validate_release_evidence,
)


def test_scale_artifact_exists_with_1500_cells() -> None:
    raw_path = Path("reports/scale_7b/raw.json")
    if not raw_path.is_file():
        return
    raw = json.loads(raw_path.read_text())
    assert len(raw.get("cells") or []) == EXPECTED_CELL_COUNT


def test_release_evidence_validator_passes_when_artifacts_present() -> None:
    if not Path("reports/scale_7b/raw.json").is_file():
        return
    report = validate_release_evidence(Path("."))
    assert report.checks, "expected checks to run"
    names = {c.name for c in report.checks}
    assert "scale_raw_exists" in names
    assert "scale_cell_count" in names
    assert "phase_f_exists" in names
    failed = [c for c in report.checks if not c.passed and c.severity == "error"]
    assert not failed, f"failed checks: {failed}"


def test_both_models_present_in_scale_raw() -> None:
    raw_path = Path("reports/scale_7b/raw.json")
    if not raw_path.is_file():
        return
    raw = json.loads(raw_path.read_text())
    models = set(raw.get("models_evaluated") or [])
    for cell in raw.get("cells") or []:
        if cell.get("model_name"):
            models.add(cell["model_name"])
    assert "meta-llama/Llama-3.1-8B" in models
    assert "mistralai/Mistral-7B-Instruct-v0.3" in models


def test_scale_not_deterministic() -> None:
    summary_path = Path("reports/scale_7b/scale_summary.json")
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text())
    assert summary.get("deterministic_mode") is False


def test_public_release_leaderboard_exists() -> None:
    path = Path("reports/public_release/leaderboard_final.json")
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    assert data.get("entries")


def test_phase_f_has_int8_int4_speedups() -> None:
    path = Path("reports/phaseF_kernel_benchmark.json")
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    speedups = {s["mode"]: s for s in data.get("speedups") or []}
    assert isinstance(speedups["int8"]["speedup_x"], (int, float))
    assert isinstance(speedups["int4"]["speedup_x"], (int, float))


def test_block_sparse_not_triton_accelerated() -> None:
    path = Path("reports/phaseF_kernel_benchmark.json")
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    triton_bs = next(
        (b for b in data.get("benchmarks") or [] if b.get("mode") == "block_sparse" and b.get("backend") == "triton"),
        None,
    )
    assert triton_bs is not None
    assert triton_bs.get("execution_backend") != "triton"


def test_adapter_honesty_discloses_fallback_and_probe() -> None:
    report = validate_release_evidence(Path("."))
    sq = report.adapter_honesty.get("spectralquant_real") or {}
    shard = report.adapter_honesty.get("shard_real") or {}
    if not sq.get("spectralquant_available"):
        assert sq.get("mode") == "int4_sim_scaling_fallback"
    assert shard.get("probe_only") is True


def test_check_release_evidence_script_writes_outputs() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/check_release_evidence.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    if not Path("reports/scale_7b/raw.json").is_file():
        return
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert Path("docs/RELEASE_EVIDENCE_STATUS.md").is_file()
    assert Path("reports/release_evidence_status.json").is_file()
