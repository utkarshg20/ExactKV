"""Tests for the ExactKV live correction terminal demo (Phase 7b)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "demo_exactkv_live_correction.py"


def _run_demo(*extra: str) -> str:
    cmd = [sys.executable, str(_SCRIPT), "--no-delay", "--plain", *extra]
    result = subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.mark.parametrize(
    "needle",
    [
        "EXACTKV LIVE KV CRASH TEST",
        "REJECT draft token",
        "COMMIT verifier token",
        "ExactKV failures: 0",
        "Final output match: true",
        "ExactKV tells you when they start lying",
    ],
)
def test_live_demo_output_contains_required_strings(needle: str) -> None:
    out = _run_demo()
    assert needle in out


def test_live_demo_uses_exp034_trace_fields() -> None:
    out = _run_demo()
    assert "int4_sim" in out
    assert "tool_json" in out
    assert "REJECT draft token: }}" in out
    assert "COMMIT verifier token: metric" in out
    assert "Full-KV verifier expects:" in out


def test_record_script_writes_asset() -> None:
    subprocess.run(
        [sys.executable, str(_SCRIPT), "--record-script"],
        cwd=_ROOT,
        check=True,
    )
    path = _ROOT / "docs" / "assets" / "demo_exactkv_live_correction_script.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "asciinema" in text
    assert "--no-delay --plain" in text


def test_load_demo_trace_from_fixture_when_json_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    out = _run_demo("--trace-json", str(missing))
    assert "ExactKV failures: 0" in out
