"""Release documentation consistency tests (Phase J)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DOCS = (
    "docs/QUICKSTART.md",
    "docs/REPRODUCIBILITY.md",
    "docs/METRIC_DEFINITIONS.md",
    "docs/CLAIM_BOUNDARIES.md",
    "docs/RESULTS_SUMMARY.md",
    "docs/ARTIFACT_INDEX.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/PROJECT_LINEAGE.md",
    "docs/VERSION_LINEAGE.md",
    "docs/HISTORICAL_ARTIFACT_INVENTORY.md",
    "docs/EXACTKV_TECHNICAL_REPORT.md",
    "docs/launch_blog_final.md",
    "docs/launch_x_thread_final.md",
    "docs/launch_linkedin_final.md",
)


def test_required_docs_exist() -> None:
    for rel in REQUIRED_DOCS:
        assert (_ROOT / rel).is_file(), rel


def test_claim_boundaries_sections() -> None:
    text = (_ROOT / "docs/CLAIM_BOUNDARIES.md").read_text(encoding="utf-8").lower()
    assert "## allowed" in text
    assert "## qualified" in text
    assert "## forbidden" in text


def test_results_summary_1500_cells() -> None:
    text = (_ROOT / "docs/RESULTS_SUMMARY.md").read_text(encoding="utf-8")
    assert "1500" in text
    assert "exactkv_failures" in text.lower() or "ExactKV failures" in text
    assert "unavailable" not in text.lower() or "mock_fallback" in text.lower()


def test_artifact_index_required_entries() -> None:
    text = (_ROOT / "docs/ARTIFACT_INDEX.md").read_text(encoding="utf-8")
    for artifact in (
        "reports/scale_7b/raw.json",
        "reports/novelty_audit.json",
        "reports/phaseF_kernel_benchmark.json",
    ):
        assert artifact in text


def test_technical_report_has_lineage() -> None:
    text = (_ROOT / "docs/EXACTKV_TECHNICAL_REPORT.md").read_text(encoding="utf-8").lower()
    assert "project lineage" in text
    assert "did not start at phase a" in text
    assert "v1" in text and "v21" in text


def test_version_lineage_doc_exists() -> None:
    text = (_ROOT / "docs/VERSION_LINEAGE.md").read_text(encoding="utf-8").lower()
    assert "v1" in text and "v21" in text


def test_release_checklist_final_signoff() -> None:
    text = (_ROOT / "docs/RELEASE_CHECKLIST.md").read_text(encoding="utf-8").lower()
    assert "final launch sign-off" in text
    assert "check_launch_pack" in text


def test_audit_and_evidence_pass() -> None:
    for script in (
        "scripts/audit_public_claims.py",
        "scripts/check_release_evidence.py",
        "scripts/check_no_secrets.py",
    ):
        proc = subprocess.run([sys.executable, script], cwd=_ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, f"{script}: {proc.stdout}{proc.stderr}"
