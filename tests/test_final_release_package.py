"""Tests for the ExactKV final release synthesis package (Part 9)."""
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
    "launch/x_thread.md",
    "launch/linkedin_post.md",
    "launch/short_announcement.md",
    "launch/launch_manifest.json",
    "release_synthesis/artifact_inventory.md",
    "release_synthesis/artifact_inventory.json",
    "release_synthesis/artifact_inventory.csv",
    "release_synthesis/evidence_ledger.md",
    "release_synthesis/evidence_ledger.json",
    "release_synthesis/claim_decision_table.md",
    "release_synthesis/claim_decision_table.json",
    "release_synthesis/source_of_truth_map.md",
    "release_synthesis/project_lineage.md",
    "release_synthesis/version_lineage.md",
    "release_synthesis/phase_lineage.md",
    "release_synthesis/related_work_audit.md",
    "release_synthesis/references.bib",
    "RELEASE.md",
]


@pytest.mark.parametrize("rel", REQUIRED)
def test_required_artifact_exists(rel):
    assert (ROOT / rel).is_file(), f"missing {rel}"


def test_source_of_truth_present():
    assert (ROOT / "reports/scale_7b/raw.json").is_file()


def test_claim_decision_table_well_formed():
    cdt = json.loads((ROOT / "release_synthesis/claim_decision_table.json").read_text())
    claims = cdt["claims"]
    assert len(claims) >= 15
    for c in claims:
        assert c["evidence_artifact"]
        assert c["decision"] in {"allowed", "allowed_with_qualification", "forbidden"}
    forbidden = {c["claim"].lower() for c in claims if c["decision"] == "forbidden"}
    assert any("end-to-end speedup" in f for f in forbidden)
    assert any("active gpu memory" in f for f in forbidden)
    assert any("reproduces vericache" in f for f in forbidden)
    assert any("production ready" in f for f in forbidden)


def test_evidence_ledger_headline():
    led = json.loads((ROOT / "release_synthesis/evidence_ledger.json").read_text())
    assert led["benchmark_source_of_truth"] == "reports/scale_7b/raw.json"
    assert led["headline_facts"]["exactkv_failures"] == 0
    assert led["headline_facts"]["total_cells"] == 1500


def test_export_status_explains_pdf():
    es = json.loads((ROOT / "paper/export_status.json").read_text())
    if not es.get("pdf_generated"):
        assert es.get("pdf_reason")
        assert es.get("pdf_build_commands")


def test_references_bib_has_verified_entries():
    bib = (ROOT / "release_synthesis/references.bib").read_text()
    for key in ("leviathan2023speculative", "kwon2023pagedattention",
                "hooper2024kvquant", "liu2024kivi", "li2024snapkv",
                "liu2024cachegen", "chen2024magicdec"):
        assert key in bib, f"missing bib entry {key}"


def test_inventory_covers_all_tracked_files():
    inv = json.loads((ROOT / "release_synthesis/artifact_inventory.json").read_text())
    assert inv["total_artifacts"] > 1000
    paths = {a["path"] for a in inv["artifacts"]}
    assert "reports/scale_7b/raw.json" in paths


def test_validator_script_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_final_release_package.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
