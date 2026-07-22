"""Tests for the ExactKV live terminal demo."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from exactkv.demo.live_terminal import (
    TerminalStyle,
    center_visible,
    ljust_visible,
    strip_ansi,
    visible_len,
)
from exactkv.demo.streaming_demo import (
    _box_line,
    _comparison_columns,
    _drift_alert,
    _intro_frame,
    _top_rail,
    _verifier_card,
    _wrap_snippet,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "exactkv_live_demo.py"
_JSON = _ROOT / "site" / "data" / "case_studies.json"


def test_wrap_snippet_fills_and_scrolls() -> None:
    assert _wrap_snippet('{"a":1}', 10) == ['{"a":1}', "", "", "", ""]
    wrapped = _wrap_snippet("units imperial metric", 12)
    assert any("metric" in line for line in wrapped)
    assert not any(line.endswith("m") and "etric" in wrapped for line in wrapped)


def test_wrap_snippet_keeps_words_intact() -> None:
    wrapped = _wrap_snippet('{"units":"metric","temp":18}', 14)
    joined = " ".join(wrapped).replace(" ", "")
    assert "metric" in joined
    for line in wrapped:
        if line and line[-1].isalpha():
            assert not (len(line) < len("metric") and line in "metric")


def test_visible_padding_ignores_ansi() -> None:
    style = TerminalStyle(plain=False)
    colored = style.red(style.bold("DRIFT"))
    assert visible_len(colored) == 5
    assert visible_len(ljust_visible(colored, 12)) == 12
    assert visible_len(center_visible(colored, 11)) == 11
    assert strip_ansi(ljust_visible(colored, 12)).endswith("       ")


def test_box_line_stable_with_ansi() -> None:
    style = TerminalStyle(plain=False)
    width = 40
    plain = _box_line("hello", width)
    colored = _box_line(style.red(style.bold("hello")), width)
    assert visible_len(plain) == visible_len(colored) == width + 4
    assert plain.startswith("┃ ") and plain.endswith(" ┃")
    assert colored.startswith("┃ ") and colored.endswith(" ┃")


def test_boxed_frames_keep_right_border_aligned() -> None:
    style = TerminalStyle(plain=False)
    frames = [
        _intro_frame(style, step=2),
        _top_rail(style),
        _drift_alert(style, note="note", wrong="imperial", right="metric", flash=True),
        _verifier_card(style, wrong="22", right="18", drift_num=2),
        _comparison_columns(
            style=style,
            full_vis='{"units":"metric"}',
            lossy_vis='{"units":"imperial"}',
            exactkv_vis='{"units":"',
            active=1,
            lossy_flag="DRIFT",
            exactkv_flag="HOLD",
            cursor_path=2,
            hot_lossy=True,
        ),
    ]
    for lines in frames:
        boxed = [ln for ln in lines if ln[:1] in {"┏", "┗", "┃", "║", "┌", "└", "├", "│"}]
        if not boxed:
            continue
        widths = {visible_len(ln) for ln in boxed}
        assert len(widths) == 1, f"misaligned borders: {widths}\n" + "\n".join(
            repr(strip_ansi(ln)) for ln in boxed
        )

def test_streaming_demo_wraps_in_output() -> None:
    out = _run_demo()
    assert "SIDE-BY-SIDE TOKEN PATHS" in out


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
        "EXACTKV",
        "LOSSY DRAFT",
        "EXACTKV OUT",
        "VERIFIER",
        "REJECT",
        "COMMIT",
        "EXACTKV MATCH",
        "exactkv_failures: 0",
        "KV compression should not be trusted",
        "imperial",
        "metric",
        "22",
        "overcast",
        "45",
        "clear skies",
        "drifts caught & corrected: 4",
        "PROBLEM",
        "PANELS",
        "8,132",
        "first-divergence",
        "France",
        "WITHOUT EXACTKV",
    ],
)
def test_streaming_demo_required_strings(needle: str) -> None:
    out = _run_demo()
    assert needle in out


def test_streaming_is_default_mode() -> None:
    out = _run_demo()
    assert "SIDE-BY-SIDE TOKEN PATHS" in out
    assert "WITHOUT EXACTKV" in out
    assert "ACT 1" not in out


def test_hero_scenario_semantic_and_scale_punch() -> None:
    out = _run_demo("--speed", "hero")
    assert "dropoff" in out
    assert "pickup" in out
    assert "REJECT" in out
    assert "COMMIT" in out
    assert "TASK TYPE DOMINATES DRIFT" in out
    assert "~6%" in out or "6%" in out
    assert "~90%" in out or "90%" in out
    assert "8,132" in out
    assert "imperial" not in out  # weather scenario not active


@pytest.mark.parametrize(
    "needle",
    [
        "EXACTKV LIVE CASE STUDIES",
        "Full KV",
        "Lossy draft",
        "ExactKV out",
        "DRIFT DETECTED",
        "EXACTKV MATCH",
        "exactkv_failures: 0",
    ],
)
def test_cases_mode_required_strings(needle: str) -> None:
    out = _run_demo("--mode", "cases", "--case", "p02_p2_json_tool")
    assert needle in out


def test_cases_carousel_four() -> None:
    out = _run_demo("--mode", "cases")
    assert "CASE 1/4" in out
    assert "CASE 4/4" in out


def test_list_cases() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--mode", "cases", "--list-cases"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "p02_p2_json_tool" in result.stdout


def test_kivi_catastrophic_case(tmp_path: Path) -> None:
    data = json.loads(_JSON.read_text(encoding="utf-8"))
    subset = {
        "case_studies": [
            c
            for c in data["case_studies"]
            if c["prompt_id"] == "lb_narrativeqa_000_ctx2048"
            and c["compressor_name"] == "kivi_offline_r32"
        ]
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(subset), encoding="utf-8")
    out = _run_demo(
        "--mode", "cases",
        "--json", str(path),
        "--case", "lb_narrativeqa_000_ctx2048",
        "--compressor", "kivi_offline_r32",
    )
    assert "!!!!" in out
