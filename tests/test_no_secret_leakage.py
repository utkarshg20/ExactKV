"""Tests for secret leakage scanner (Gate R0)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_no_secret_leakage_scan_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/check_no_secrets.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED" in proc.stdout


def test_secret_scanner_detects_hf_token_pattern() -> None:
    import tempfile

    from scripts.check_no_secrets import scan_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
        fh.write('HF_TOKEN="hf_abcdefghijklmnopqrstuvwxyz123456"\n')
        tmp_path = Path(fh.name)
    try:
        hits = scan_file(tmp_path)
        assert hits
    finally:
        tmp_path.unlink(missing_ok=True)


def test_secret_scanner_allows_placeholder() -> None:
    import tempfile

    from scripts.check_no_secrets import scan_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
        fh.write("export HF_TOKEN=...\n# PASTE_YOUR_TOKEN_HERE\n")
        tmp_path = Path(fh.name)
    try:
        hits = scan_file(tmp_path)
        assert not hits
    finally:
        tmp_path.unlink(missing_ok=True)
