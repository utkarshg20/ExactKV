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
    _run("--md", "--html", "--plain")


def test_leaderboard_script_exits_zero(generated: None) -> None:
    assert _SCRIPT.is_file()


@pytest.mark.parametrize(
    "needle",
    [
        "EXACTKV CRASH-TEST LEADERBOARD",
        "FULL PANEL",
        "REPAIR POLICY",
        "RESTRICTED BACKEND",
        "SMOKE ONLY",
        "FUTURE CANDIDATE",
        "Compressors ranked by acceptance and exactness",
        "No speedup, memory savings, or serving claims",
        "FULL · REAL-BYTE",
        "RESTRICTED ·",
        "EXTERNAL DRAFTER",
        "Shard external-drafter",
        "SMOKE]",
    ],
)
def test_terminal_output(generated: None, needle: str) -> None:
    out = _run("--terminal", "--plain").stdout
    assert needle in out


def test_plain_mode_no_box_drawing(generated: None) -> None:
    out = _run("--terminal", "--plain").stdout
    assert "╔" not in out
    assert "╚" not in out


def test_summary_mode(generated: None) -> None:
    out = _run("--summary").stdout
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    assert 8 <= len(lines) <= 12
    assert "EXACTKV LEADERBOARD — LAUNCH SUMMARY" in out
    assert "Best full-panel:" in out
    assert "Restricted backends:" in out
    assert "Smoke-only adapters:" in out
    assert "Future candidates:" in out
    assert "Top caveat:" in out
    assert "No speedup, memory savings, or serving claims" in out


def test_watch_once_plain(generated: None) -> None:
    out = _run("--watch", "--once", "--plain").stdout
    assert "EXACTKV CRASH-TEST LEADERBOARD" in out
    assert "Last refresh:" in out
    assert "FULL PANEL" in out


def test_leaderboard_md_exists(generated: None) -> None:
    path = _ROOT / "docs" / "leaderboard.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "KV Compression Crash-Test Leaderboard" in text
    assert "Full-suite integrated" in text
    assert "Restricted backends" in text
    assert "Smoke-only adapters" in text
    assert "Future candidates" in text
    assert "Ranking policy" in text
    assert "Full-panel results" in text
    assert "not ranked against full-panel compressors" in text
    assert "no ExactKV panel metrics yet" in text
    assert "FULL" in text
    assert "TurboQuant" in text
    assert "No speedup, memory savings, or serving claims" in text


def test_leaderboard_html_exists(generated: None) -> None:
    path = _ROOT / "docs" / "leaderboard.html"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "KV Compression Crash-Test Leaderboard" in text
    assert 'data-tab="full-suite"' in text
    assert 'data-tab="repair"' in text
    assert 'data-tab="restricted"' in text
    assert 'data-tab="smoke"' in text
    assert 'data-tab="future"' in text
    assert "sticky-header" in text
    assert "not-apples" in text
    assert "badge tier-full" in text
    assert "badge tier-restricted" in text
    assert "badge tier-smoke" in text
    assert "SMOKE" in text
    assert "SIMULATED" in text
    assert "REAL-BYTE" in text
    assert "No speedup, memory savings, or serving claims" in text


def test_leaderboard_md_shard_restricted_not_future(generated: None) -> None:
    text = (_ROOT / "docs" / "leaderboard.md").read_text(encoding="utf-8")
    assert "Shard external-drafter probe" in text
    assert "RESTRICTED BACKEND" in text or "Restricted backends" in text
    assert "EXTERNAL DRAFTER" in text or "external drafter probe" in text
    assert "58.22/64" in text or "0.911" in text
    assert "18.75%" in text or "6 draft divergences" in text
    assert "31.25%" in text
    assert "56.25%" in text or "Exp 041" in text
    # Shard must not appear under Future candidates table
    future_idx = text.find("## Future candidates")
    restricted_idx = text.find("## Restricted backends")
    shard_in_restricted = text[restricted_idx:future_idx].find("Shard external-drafter probe") >= 0
    assert shard_in_restricted
    if future_idx >= 0:
        future_section = text[future_idx:]
        assert "Shard external-drafter probe" not in future_section
    assert "SpectralQuant" in text
    assert "FUTURE" in text


def test_leaderboard_shard_caveat_language(generated: None) -> None:
    text = (_ROOT / "docs" / "leaderboard.md").read_text(encoding="utf-8").lower()
    assert "not full-panel compressor acceptance" in text
    assert "external shard readme not exactkv" in text
    forbidden = [
        "shard is integrated",
        "shard improves memory",
        "shard improves speed",
        "shard is production-ready",
        "shard is an exactkv compressor",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_leaderboard_html_shard_restricted(generated: None) -> None:
    text = (_ROOT / "docs" / "leaderboard.html").read_text(encoding="utf-8")
    assert "Shard external-drafter probe" in text
    assert 'data-tab="restricted"' in text
    assert "EXTERNAL DRAFTER" in text
    assert "SpectralQuant" in text
    assert 'data-tab="future"' in text
    # Future tab should not list Shard probe row
    future_start = text.find('id="tab-future"')
    assert future_start >= 0
    future_panel = text[future_start : future_start + 2500]
    assert "Shard external-drafter probe" not in future_panel


def test_html_has_no_positive_speedup_claims(generated: None) -> None:
    text = (_ROOT / "docs" / "leaderboard.html").read_text(encoding="utf-8")
    lower = text.lower()
    if "speedup" in lower:
        assert "no speedup" in lower


def test_terminal_has_no_positive_speedup_claims(generated: None) -> None:
    out = _run("--terminal", "--plain").stdout.lower()
    if "speedup" in out:
        assert "no speedup" in out
