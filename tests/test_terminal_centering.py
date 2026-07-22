"""Tests for terminal centering helpers used by the live demo."""
from __future__ import annotations

from exactkv.demo.live_terminal import center_on_terminal, strip_ansi, visible_len


def test_center_on_terminal_pads_visible_width() -> None:
    colored = "\033[31mEXACTKV\033[0m"
    centered = center_on_terminal(colored, cols=21)
    assert centered.startswith(" " * 7)
    assert strip_ansi(centered).strip() == "EXACTKV"
    assert visible_len(centered) == 14  # left pad + text


def test_center_on_terminal_leaves_wide_lines_alone() -> None:
    line = "x" * 40
    assert center_on_terminal(line, cols=20) == line
