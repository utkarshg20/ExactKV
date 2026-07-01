"""Tests for the public ExactKV release package (GitHub-facing)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "paper/ExactKV_Technical_Report.md",
    "paper/ExactKV_Technical_Report.tex",
    "paper/references.bib",
    "paper/export_status.json",
    "site/index.html",
    "site/claim_safe_copy.json",
    "reports/public_release/leaderboard_final.json",
    "reports/scale_7b/raw.json",
    "RELEASE.md",
    "README.md",
    "docs/CLAIM_BOUNDARIES.md",
]


@pytest.mark.parametrize("rel", REQUIRED)
def test_required_artifact_exists(rel):
    assert (ROOT / rel).is_file(), f"missing {rel}"


def test_leaderboard_well_formed():
    lb = json.loads((ROOT / "reports/public_release/leaderboard_final.json").read_text())
    assert len(lb.get("entries") or []) >= 4
    assert lb.get("validation_result", {}).get("valid") is not False


def test_export_status_explains_pdf():
    es = json.loads((ROOT / "paper/export_status.json").read_text())
    if not es.get("pdf_generated"):
        assert es.get("pdf_reason")
        assert es.get("pdf_build_commands")


def test_validator_script_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_final_release_package.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
