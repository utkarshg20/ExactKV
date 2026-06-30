"""Tests for the ExactKV landing page artifacts (site/)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_site_files_exist():
    for f in ("index.html", "styles.css", "main.js", "content_manifest.json",
              "claim_safe_copy.json", "README.md"):
        assert (SITE / f).is_file(), f"missing site/{f}"


def test_hero_and_leaderboard_present():
    html = (SITE / "index.html").read_text(encoding="utf-8").lower()
    assert "lying" in html, "hero headline missing"
    assert "leaderboard" in html
    assert "1,500" in html, "1500-cell headline missing"
    assert "exactkv failures" in html


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
    # Phrases below are positive claims that never appear even in caveat/negation
    # form on the page. (Phrases like "end-to-end speedup" or "active gpu memory
    # savings" DO appear, but only negated as caveats; the negation-aware
    # check_site_claims.py validator covers those — see test below.)
    for term in ("first ever", "first and only", "nothing like this exists",
                 "production-ready", "reproduces vericache", "beats vericache",
                 "fastest", "10x compression"):
        assert term not in html, f"forbidden term present: {term}"


def test_leaderboard_rows_match_scale_evidence():
    manifest = json.loads((SITE / "content_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["leaderboard_rows"]
    assert len(rows) >= 10
    # noop / int8 top the board with full acceptance and zero divergence.
    top = rows[0]
    assert top["compressor"] == "noop"
    assert top["acceptance"] == 1.0
    # SpectralQuant disclosed as fallback/proxy; Shard as probe_only.
    avail = {(r["compressor"], r["availability"]) for r in rows}
    assert any(c == "spectralquant" and a == "mock_fallback" for c, a in avail)
    assert any(c == "shard" and a == "probe_only" for c, a in avail)


def test_claim_safe_copy_lists_caveats_and_forbidden():
    copy = json.loads((SITE / "claim_safe_copy.json").read_text(encoding="utf-8"))
    assert len(copy["required_caveats"]) >= 6
    assert len(copy["forbidden_terms"]) >= 8


def test_site_claims_script_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_site_claims.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
