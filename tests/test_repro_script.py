"""Tests for exactkv_repro.py (Phase J)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_full_without_confirm_expensive_refuses() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/exactkv_repro.py", "--full"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "confirm-expensive" in proc.stderr.lower() or "confirm-expensive" in proc.stdout.lower()


def test_release_check_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/exactkv_repro.py", "--release-check"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_reports_only_writes_manifest() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/exactkv_repro.py", "--reports-only"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest_path = _ROOT / "reports/repro_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("claim_boundary_note")
    assert manifest.get("commands_run")
