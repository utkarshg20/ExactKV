"""Tests for scripts/audit_public_claims.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "audit_public_claims.py"


def test_audit_script_exists() -> None:
    assert _SCRIPT.is_file()


def test_audit_passes_on_repo() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_detects_forbidden_claim_in_temp_file(tmp_path: Path) -> None:
    bad = tmp_path / "BAD.md"
    bad.write_text("ExactKV delivers 2x speedup over full KV.\n", encoding="utf-8")
    # Scan only temp README-like file by invoking module logic
    from scripts.audit_public_claims import scan_file

    hits = scan_file(bad)
    assert hits
    assert any("speedup" in label.lower() for _, label, _ in hits)


def test_allowlists_negated_claim(tmp_path: Path) -> None:
    ok = tmp_path / "OK.md"
    ok.write_text("ExactKV does not claim speedup or throughput.\n", encoding="utf-8")
    from scripts.audit_public_claims import scan_file

    assert not scan_file(ok)
