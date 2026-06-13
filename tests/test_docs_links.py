"""Tests for scripts/check_docs_links.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_docs_links.py"


def test_link_checker_exists() -> None:
    assert _SCRIPT.is_file()


def test_links_pass_on_repo() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_detects_missing_local_link(tmp_path: Path) -> None:
    md = tmp_path / "test.md"
    md.write_text("[missing](does_not_exist.md)\n", encoding="utf-8")
    from scripts.check_docs_links import scan_file

    hits = scan_file(md, tmp_path)
    assert hits
