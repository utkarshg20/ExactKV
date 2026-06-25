"""Public release artifact tests (Phase J)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ensure_public_release() -> None:
    subprocess.run(
        [sys.executable, "scripts/exactkv_repro.py", "--reports-only"],
        cwd=_ROOT,
        check=True,
    )


def test_public_release_files_exist(ensure_public_release: None) -> None:
    release = _ROOT / "reports/public_release"
    for name in (
        "README_PUBLIC.md",
        "benchmark_summary.md",
        "methodology.md",
        "leaderboard_final.json",
        "release_manifest.json",
        "demo_cards.json",
        "demo_cards.md",
        "launch_manifest.json",
    ):
        assert (release / name).is_file(), name


def test_readme_uses_1500_cell_scale_evidence(ensure_public_release: None) -> None:
    text = (_ROOT / "reports/public_release/README_PUBLIC.md").read_text(encoding="utf-8")
    assert "1500" in text
    assert "scale_7b" in text.lower() or "phase h+" in text.lower()
    assert "336" not in text.split("Historical")[0] if "Historical" in text else "1500" in text


def test_benchmark_summary_1500_cells(ensure_public_release: None) -> None:
    text = (_ROOT / "reports/public_release/benchmark_summary.md").read_text(encoding="utf-8").lower()
    assert "1500" in text
    assert "**cells:** 336" not in text


def test_manifest_source_artifacts(ensure_public_release: None) -> None:
    manifest = json.loads(
        (_ROOT / "reports/public_release/release_manifest.json").read_text(encoding="utf-8")
    )
    sources = manifest.get("source_artifacts") or []
    required = (
        "reports/scale_7b/raw.json",
        "reports/scale_7b/leaderboard.json",
        "reports/scale_7b/scale_summary.json",
        "reports/novelty_audit.json",
    )
    for src in required:
        assert src in sources, f"missing {src}"


def test_public_release_validator_passes(ensure_public_release: None) -> None:
    from exactkv.platform.public_release_validator import validate_public_release

    report = validate_public_release(_ROOT)
    errors = [i for i in report.issues if i.severity == "error"]
    assert report.valid, [f"{e.check}: {e.detail}" for e in errors]


def test_check_public_release_script() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/check_public_release.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_raw_contains_mistral_cells() -> None:
    raw_path = _ROOT / "reports/scale_7b/raw.json"
    if not raw_path.is_file():
        pytest.skip("scale raw missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    mistral = [c for c in raw.get("cells") or [] if "mistral" in str(c.get("model_name", "")).lower()]
    assert len(mistral) >= 1
    assert len(mistral) == 750


def test_public_leaderboard_mistral_numeric_scores(ensure_public_release: None) -> None:
    lb = json.loads(
        (_ROOT / "reports/public_release/leaderboard_final.json").read_text(encoding="utf-8")
    )
    mistral_rows = [
        e for e in lb.get("entries") or []
        if "mistral" in str(e.get("model", "")).lower()
    ]
    assert mistral_rows, "expected Mistral rows in public leaderboard"
    scored = [r for r in mistral_rows if r.get("score") is not None]
    assert scored, "expected numeric Mistral scores"
    unavailable = [r for r in mistral_rows if r.get("availability") == "unavailable"]
    assert not unavailable, "Mistral must not be unavailable when raw cells exist"


def test_both_models_in_public_leaderboard(ensure_public_release: None) -> None:
    lb = json.loads(
        (_ROOT / "reports/public_release/leaderboard_final.json").read_text(encoding="utf-8")
    )
    models = {e.get("model") for e in lb.get("entries") or [] if e.get("score") is not None}
    assert "meta-llama/Llama-3.1-8B" in models
    assert "mistralai/Mistral-7B-Instruct-v0.3" in models


def test_validator_fails_synthetic_mistral_unavailable() -> None:
    from exactkv.platform.leaderboard_aggregates import validate_leaderboard_against_raw

    phase_a = {
        "cells": [{"model_name": "mistralai/Mistral-7B-Instruct-v0.3", "compressor_name": "noop"}],
    }
    leaderboard = {
        "entries": [
            {
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "availability": "unavailable",
                "score": None,
            },
        ],
    }
    errors = validate_leaderboard_against_raw(phase_a, leaderboard)
    assert errors


def test_per_model_tables_repair_includes_mistral() -> None:
    from exactkv.platform.leaderboard_aggregates import repair_phase_a_report_aggregates

    raw_path = _ROOT / "reports/scale_7b/raw.json"
    if not raw_path.is_file():
        pytest.skip("scale raw missing")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    repaired = repair_phase_a_report_aggregates(raw)
    tables = repaired.get("per_model_tables") or {}
    assert "mistralai/Mistral-7B-Instruct-v0.3" in tables
    assert tables["mistralai/Mistral-7B-Instruct-v0.3"]["total_cells"] == 750
