"""Tests for public visual polish package (Exp 036)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "render_public_visuals_036.py"
_ASSETS = _ROOT / "docs" / "assets"

PUBLIC_ASSETS = [
    "public_exactkv_hero_card.png",
    "public_killer_correction_card.png",
    "public_exactness_wall.png",
    "public_leaderboard.png",
    "public_timing_truth_card.png",
    "public_memory_truth_card.png",
    "public_exactkv_one_page_summary.png",
]


@pytest.fixture(scope="module")
def run_render() -> None:
    subprocess.run([sys.executable, str(_SCRIPT)], cwd=_ROOT, check=True)


@pytest.mark.parametrize("name", PUBLIC_ASSETS)
def test_public_png_exists(run_render: None, name: str) -> None:
    path = _ASSETS / name
    assert path.is_file(), name
    assert path.stat().st_size > 2000


def test_thread_cards_exist(run_render: None) -> None:
    thread_dir = _ASSETS / "public_exactkv_launch_thread_cards"
    assert thread_dir.is_dir()
    assert len(list(thread_dir.glob("thread_*.png"))) >= 6


def test_public_visual_package_doc(run_render: None) -> None:
    doc = _ROOT / "docs" / "PUBLIC_VISUAL_PACKAGE.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "not a new benchmark" in text
    assert "Internal-only" in text
    assert "Shard" in text
    lb = (_ROOT / "docs" / "leaderboard.md").read_text(encoding="utf-8")
    assert "RESTRICTED BACKEND" in lb or "## Restricted backends" in lb
    assert "TurboQuant" in lb
