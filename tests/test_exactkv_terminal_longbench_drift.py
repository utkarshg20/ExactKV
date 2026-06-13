"""Tests for ExactKV LongBench-style drift terminal demo (Phase 10A)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "exactkv_terminal_longbench_drift.py"


def _run_demo(*extra: str) -> str:
    cmd = [sys.executable, str(_SCRIPT), "--no-delay", "--plain", *extra]
    result = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, check=True)
    return result.stdout


@pytest.mark.parametrize(
    "needle",
    [
        "Outcome",
        "compressed cache can still drift",
        "DRIFT DETECTED",
        "exact words changed",
        "reject",
        "commit",
        "ExactKV failures:                0",
        "Final output match:              true",
    ],
)
def test_terminal_longbench_drift_required_strings(needle: str) -> None:
    out = _run_demo()
    assert needle in out


def test_terminal_demo_uses_lb_md_fixture_by_default() -> None:
    out = _run_demo()
    assert "lb_md_001" in out or "Maya" in out
    assert "billing" in out.lower()
    assert "answer" in out.lower()


def test_terminal_demo_source_json() -> None:
    report = _ROOT / "reports" / "experiment_037_longbench_style_drift_candidates.json"
    if not report.is_file():
        pytest.skip("Exp 037 JSON not present locally")
    out = _run_demo("--source-json", str(report))
    assert "OUTCOME" in out
    assert "DRIFT DETECTED" in out
