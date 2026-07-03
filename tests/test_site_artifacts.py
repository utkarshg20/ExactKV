"""Tests for the ExactKV landing page artifacts (site/)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_site_files_exist():
    for f in (
        "index.html",
        "styles.css",
        "main.js",
        "content_manifest.json",
        "claim_safe_copy.json",
        "README.md",
        "data/leaderboard.json",
        "data/case_studies.json",
        "assets/public_exactkv_one_page_summary.png",
        "assets/exp035_first_divergence_histogram.png",
        "assets/exp035_category_heatmap.png",
    ):
        assert (SITE / f).is_file(), f"missing site/{f}"


def test_hero_and_leaderboard_present():
    html = (SITE / "index.html").read_text(encoding="utf-8").lower()
    assert "lying" in html, "hero headline missing"
    assert "reviewer tl;dr" in html, "reviewer TLDR missing"
    assert "executive summary" in html, "executive summary missing"
    assert "read the evidence" in html, "reviewer action card missing"
    assert "read pdf" in html, "hero PDF CTA missing"
    assert "leaderboard" in html
    assert "1,500" in html, "1500-cell headline missing"
    assert "exactkv failures" in html
    assert "loading case studies" not in html, "case studies should not show loading placeholder"
    assert "loading leaderboard" not in html, "leaderboard should not show loading placeholder"


def test_required_caveats_present():
    html = (SITE / "index.html").read_text(encoding="utf-8").lower()
    for caveat in (
        "kernel microbenchmark",
        "stored tensor byte ratio",
        "fallback/proxy",
        "probe-first",
        "does not reproduce vericache",
        "not a production serving system",
    ):
        assert caveat in html, f"missing caveat: {caveat}"


def test_no_forbidden_terms_in_rendered_page():
    html = (SITE / "index.html").read_text(encoding="utf-8").lower()
    for term in (
        "first ever",
        "first and only",
        "nothing like this exists",
        "production-ready",
        "reproduces vericache",
        "beats vericache",
        "fastest",
        "10x compression",
    ):
        assert term not in html, f"forbidden term present: {term}"


def test_leaderboard_rows_match_scale_evidence():
    manifest = json.loads((SITE / "content_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["leaderboard_rows"]
    assert len(rows) >= 10
    top = rows[0]
    assert top["compressor"] == "noop"
    assert top["acceptance"] == 1.0
    avail = {(r["compressor"], r["availability"]) for r in rows}
    assert any(c == "spectralquant" and a == "mock_fallback" for c, a in avail)
    assert any(c == "shard" and a == "probe_only" for c, a in avail)


def test_content_manifest_matches_page_structure():
    manifest = json.loads((SITE / "content_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("schema", "").startswith("exactkv.site.content_manifest.v2")
    section_ids = {s["id"] for s in manifest.get("sections", [])}
    assert "reviewer-tldr" in section_ids
    assert "summary" in section_ids
    assert "case-studies" in section_ids
    assert "leaderboard" in section_ids
    assert "demo_cards" not in manifest
    assert manifest.get("deployment", {}).get("public_url", "").startswith("https://")
    policy = manifest.get("case_study_policy") or {}
    assert "longbench_pilot" in (policy.get("excludes") or [])
    assert len(manifest.get("leaderboard_rows") or []) >= 10


def test_claim_safe_copy_lists_caveats_and_forbidden():
    copy = json.loads((SITE / "claim_safe_copy.json").read_text(encoding="utf-8"))
    assert len(copy["required_caveats"]) >= 6
    assert len(copy["forbidden_terms"]) >= 8
    assert "technical_report_pdf" in copy.get("urls", {})


def test_release_metadata_in_manifest():
    manifest = json.loads((SITE / "content_manifest.json").read_text(encoding="utf-8"))
    rel = manifest.get("release") or {}
    assert rel.get("git_tag") == "v-release"
    assert rel.get("public_name") == "research release"


def test_release_metadata_in_claim_copy():
    copy = json.loads((SITE / "claim_safe_copy.json").read_text(encoding="utf-8"))
    rel = copy.get("release") or {}
    assert rel.get("git_tag") == "v-release"
    assert "github_release" in copy.get("urls", {})


def test_site_claims_script_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_site_claims.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_site_case_studies_use_headline_panels():
    path = SITE / "data" / "case_studies.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("case_studies") or []
    assert cases, "expected curated case studies"
    pilot_only = all(c.get("panel") == "longbench_pilot" for c in cases)
    assert not pilot_only, "case studies should not all come from synthetic longbench_pilot"
    for c in cases:
        panel = c.get("panel") or ""
        assert panel in {
            "core_scale",
            "hf_longbench_v26",
            "bfcl_validity_v27",
            "bfcl_export_50",
            "faithful_wave1_longbench",
        }, f"unexpected panel source: {panel}"
        blob = " ".join(
            str(c.get(k) or "")
            for k in ("full_snippet", "lossy_snippet", "exactkv_snippet")
        )
        assert "deterministic filler" not in blob, f"pilot filler leaked into {c.get('prompt_id')}"
        if panel != "faithful_wave1_longbench":
            assert "segment_" not in blob, f"padding segment leaked into {c.get('prompt_id')}"
