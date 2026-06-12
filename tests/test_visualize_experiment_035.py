"""Tests for Experiment 035 visualization script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "visualize_experiment_035.py"
_ASSETS = _ROOT / "docs" / "assets"

PNG_NAMES = [
    "exp035_exactness_summary.png",
    "exp035_acceptance_by_compressor.png",
    "exp035_acceptance_by_model.png",
    "exp035_category_heatmap.png",
    "exp035_first_divergence_histogram.png",
    "exp035_timing_diagnostic.png",
    "exp035_memory_diagnostic.png",
    "exp035_killer_demo_card.png",
]


@pytest.fixture(scope="module")
def run_visualize() -> None:
    subprocess.run([sys.executable, str(_SCRIPT)], cwd=_ROOT, check=True)


def test_visualize_script_exits_zero(run_visualize: None) -> None:
    assert _SCRIPT.is_file()


@pytest.mark.parametrize("name", PNG_NAMES)
def test_png_assets_exist(run_visualize: None, name: str) -> None:
    path = _ASSETS / name
    assert path.is_file(), f"missing {path}"
    assert path.stat().st_size > 500


def test_report_and_leaderboard_exist(run_visualize: None) -> None:
    report = _ROOT / "docs" / "EXPERIMENT_035_VISUAL_PLOTS_AND_LEADERBOARD.md"
    leaderboard = _ROOT / "docs" / "leaderboard.md"
    assert report.is_file()
    assert leaderboard.is_file()
    text = report.read_text(encoding="utf-8")
    assert "not a new benchmark" in text
    assert "ExactKV tells you when they start lying" in text
    lb = leaderboard.read_text(encoding="utf-8")
    assert "snapkv_experimental" in lb
    assert "not ExactKV results" in lb
