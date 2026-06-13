"""Tests for scripts/check_report_hygiene.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_report_hygiene.py"


def test_hygiene_script_exists() -> None:
    assert _SCRIPT.is_file()


def test_hygiene_passes_on_repo() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--require-public"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gitignore_lists_reports() -> None:
    gi = (_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "reports/*.json" in gi
    assert "reports/*.csv" in gi
