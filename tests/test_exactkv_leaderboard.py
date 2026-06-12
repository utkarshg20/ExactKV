"""Tests for ExactKV crash-test leaderboard (terminal + HTML)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "exactkv_leaderboard.py"


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *extra],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture(scope="module")
def generated() -> None:
    _run("--all", "--plain")


def test_leaderboard_script_exits_zero(generated: None) -> None:
    assert _SCRIPT.is_file()


@pytest.mark.parametrize(
    "needle",
    [
        "EXACTKV CRASH-TEST LEADERBOARD",
        "FULL PANEL RESULTS",
        "RESTRICTED BACKENDS",
        "SMOKE ONLY",
        "FUTURE CANDIDATES",
        "Compressor",
        "Acceptance",
        "Failures",
    ],
)
def test_terminal_output(generated: None, needle: str) -> None:
    out = _run("--terminal", "--plain").stdout
    assert needle in out


def test_leaderboard_md_exists(generated: None) -> None:
    path = _ROOT / "docs" / "leaderboard.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "KV Compression Crash-Test Leaderboard" in text
    assert "Full-suite integrated" in text
    assert "Restricted backends" in text
    assert "Smoke-only adapters" in text
    assert "Future candidates" in text
    assert "FULL PANEL" in text
    assert "TurboQuant" in text


def test_leaderboard_html_exists(generated: None) -> None:
    path = _ROOT / "docs" / "leaderboard.html"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "KV Compression Crash-Test Leaderboard" in text
    assert 'data-tab="full-suite"' in text
    assert 'data-tab="restricted"' in text
    assert 'data-tab="smoke"' in text
    assert 'data-tab="future"' in text
    assert "badge tier-full" in text
    assert "badge tier-restricted" in text
    assert "SMOKE ONLY" in text
    assert "SIMULATED" in text


def test_html_has_no_speedup_claims(generated: None) -> None:
    text = (_ROOT / "docs" / "leaderboard.html").read_text(encoding="utf-8")
    assert "speedup" not in text.lower() or "no speedup" in text.lower()
