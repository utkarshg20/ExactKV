"""Tests for public claim safety audit (Gate R0)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_audit_public_claims_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/audit_public_claims.py"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED" in proc.stdout


def test_forbidden_claim_detector() -> None:
    import tempfile

    from scripts.audit_public_claims import scan_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as fh:
        fh.write("ExactKV is production ready for serving system deployment.\n")
        tmp_path = Path(fh.name)
    try:
        hits = scan_file(tmp_path)
        assert hits
    finally:
        tmp_path.unlink(missing_ok=True)
