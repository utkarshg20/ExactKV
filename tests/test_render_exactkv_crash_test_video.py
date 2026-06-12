"""Tests for cinematic crash-test video renderer (Phase 8c)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "render_exactkv_crash_test_video.py"
_ASSETS = _ROOT / "docs" / "assets"
_STORYBOARD = _ASSETS / "exactkv_crash_test_storyboard.md"
_DOC = _ROOT / "docs" / "EXACTKV_CRASH_TEST_VIDEO.md"

REQUIRED_STRINGS = [
    "Everyone is racing to shrink KV caches",
    "ExactKV tells you when they start lying",
    "Compressed KV drafts",
    "Full KV verifies",
    "Wrong token rejected",
    "EXACT MATCH",
    "KV compression should not be trusted",
]


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *extra],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_fast_no_video_generates_storyboard() -> None:
    _run("--fast", "--no-video")
    assert _STORYBOARD.is_file()
    text = _STORYBOARD.read_text(encoding="utf-8")
    for needle in REQUIRED_STRINGS:
        assert needle in text, needle
    assert "Exp 034" in text or "tj_002" in text
    assert _DOC.is_file()


def test_fast_generates_watchable_artifact() -> None:
    _run("--fast")
    story = _STORYBOARD.read_text(encoding="utf-8")
    for needle in REQUIRED_STRINGS:
        assert needle in story
    html = _ASSETS / "exactkv_crash_test_demo.html"
    mp4 = _ASSETS / "exactkv_crash_test_demo.mp4"
    gif = _ASSETS / "exactkv_crash_test_demo.gif"
    frames = _ASSETS / "exactkv_crash_test_frames"
    assert html.is_file()
    assert frames.is_dir() and len(list(frames.glob("frame_*.png"))) > 10
    assert mp4.is_file() or gif.is_file(), "need MP4 or GIF watchable artifact"
    if mp4.is_file():
        assert mp4.stat().st_size > 50_000
    if gif.is_file():
        assert gif.stat().st_size > 50_000


def test_html_references_source() -> None:
    html = (_ASSETS / "exactkv_crash_test_demo.html").read_text(encoding="utf-8")
    assert "Exp 034" in html or "tj_002" in html
    assert "not" in html.lower() or "benchmark" in html.lower()
