#!/usr/bin/env python3
"""Validate the ExactKV landing page (site/) for claim safety and completeness.

Checks:
  - site/index.html, styles.css, content_manifest.json, claim_safe_copy.json exist
  - hero headline present
  - leaderboard table present
  - required caveats present (Phase F microbenchmark, stored byte ratio,
    SpectralQuant fallback, Shard probe-first, VeriCache, not production)
  - no forbidden positive claims (production-ready, first-ever, end-to-end
    speedup, active GPU memory savings, real SpectralQuant/Shard, beats X)
  - no obvious secrets/tokens

Exit 0 if clean, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

REQUIRED_FILES = [
    "index.html",
    "styles.css",
    "main.js",
    "content_manifest.json",
    "claim_safe_copy.json",
    "README.md",
    "data/leaderboard.json",
    "data/case_studies.json",
]

REQUIRED_ASSETS = [
    "assets/og_card.png",
    "assets/exactkv_logo.png",
    "assets/exactkv_icon.png",
    "assets/public_exactkv_one_page_summary.png",
    "assets/exp035_first_divergence_histogram.png",
    "assets/exp035_category_heatmap.png",
]

# (substring that must appear somewhere in index.html, human label)
REQUIRED_CONTENT = [
    ("start", "hero headline (\"start ... lying\")"),
    ("leaderboard", "leaderboard section"),
    ("acceptance", "acceptance metric"),
    ("first-divergence", "first-divergence framing"),
    ("kernel microbenchmark", "Phase F kernel microbenchmark caveat"),
    ("stored tensor byte ratio", "compression ratio caveat"),
    ("fallback/proxy", "SpectralQuant fallback/proxy caveat"),
    ("probe-first", "Shard probe-first caveat"),
    ("does not", "VeriCache / production negation caveat"),
    ("1,500", "1500-cell headline"),
    ("executive summary", "executive summary section"),
    ("read pdf", "hero PDF CTA"),
    ("v-release", "public git tag v-release"),
    ("research release", "public release name"),
]

# Lines containing any of these are treated as caveat/negation context (allowed).
ALLOW = re.compile(
    r"\b(not|no|without|does not|don't|never|isn't|fallback|proxy|probe|"
    r"microbenchmark|stored tensor|not a|not real|inspired by|forbidden)\b",
    re.I,
)

# Inherently-positive forbidden claims.
FORBIDDEN = [
    (re.compile(r"\bfirst ever\b", re.I), "first ever"),
    (re.compile(r"\bfirst and only\b", re.I), "first and only"),
    (re.compile(r"\bnothing like this exists\b", re.I), "nothing like this exists"),
    (re.compile(r"\bproduction[- ]ready\b", re.I), "production ready"),
    (re.compile(r"\bproduction serving system\b", re.I), "production serving system"),
    (re.compile(r"\bactive gpu memory savings\b", re.I), "active GPU memory savings"),
    (re.compile(r"\bvram savings\b", re.I), "VRAM savings"),
    (re.compile(r"\bend[- ]to[- ]end speedup\b", re.I), "end-to-end speedup"),
    (re.compile(r"\bfaster inference\b", re.I), "faster inference"),
    (re.compile(r"\breproduces vericache\b", re.I), "reproduces VeriCache"),
    (re.compile(r"\bbeats? (vericache|turboquant|shard|deepmind|google)\b", re.I), "beats <system>"),
    (re.compile(r"\breal spectralquant\b", re.I), "real SpectralQuant"),
    (re.compile(r"\breal shard\b", re.I), "real Shard"),
    (re.compile(r"\bfastest\b", re.I), "fastest"),
    (re.compile(r"\bsota\b", re.I), "SOTA"),
    (re.compile(r"\b\d+(\.\d+)?x\s+speedup\b", re.I), "Nx speedup (unqualified)"),
]

SECRETS = [
    (re.compile(r"\b(sk|pk)-[A-Za-z0-9]{20,}\b"), "API key"),
    (re.compile(r"\bgh[posu]_[A-Za-z0-9]{20,}\b"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "HuggingFace token"),
]


def main() -> int:
    errors: list[str] = []

    if not SITE.is_dir():
        print("FAIL: site/ directory missing")
        return 1

    for f in REQUIRED_FILES:
        if not (SITE / f).is_file():
            errors.append(f"missing required file: site/{f}")

    for f in REQUIRED_ASSETS:
        if not (SITE / f).is_file():
            errors.append(f"missing required asset: site/{f} (run scripts/sync_site_data.sh)")

    html_path = SITE / "index.html"
    html = html_path.read_text(encoding="utf-8") if html_path.is_file() else ""
    html_l = html.lower()

    for needle, label in REQUIRED_CONTENT:
        if needle.lower() not in html_l:
            errors.append(f"missing required content: {label} (looked for '{needle}')")

    # Secret scan runs across all site files.
    for f in REQUIRED_FILES:
        p = SITE / f
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, label in SECRETS:
                if pat.search(line):
                    errors.append(f"{f}:L{lineno} possible secret [{label}]")

    # Forbidden positive-claim scan runs on the RENDERED public page only.
    # claim_safe_copy.json (which lists forbidden terms by design) and README.md
    # (which documents the check) are intentionally excluded from this scan.
    for lineno, line in enumerate(html.splitlines(), 1):
        if ALLOW.search(line):
            continue
        for pat, label in FORBIDDEN:
            if pat.search(line):
                errors.append(f"index.html:L{lineno} forbidden claim [{label}]: {line.strip()[:90]}")

    # Case-study gallery must come from headline panels, not synthetic pilots.
    case_path = SITE / "data" / "case_studies.json"
    if case_path.is_file():
        import json

        case_data = json.loads(case_path.read_text(encoding="utf-8"))
        cases = case_data.get("case_studies") or []
        allowed_panels = {
            "core_scale",
            "hf_longbench_v26",
            "bfcl_validity_v27",
            "bfcl_export_50",
            "faithful_wave1_longbench",
        }
        for c in cases:
            panel = c.get("panel") or ""
            if panel not in allowed_panels:
                errors.append(f"case_studies.json: unexpected panel source {panel!r}")
            blob = " ".join(
                str(c.get(k) or "")
                for k in ("full_snippet", "lossy_snippet", "exactkv_snippet")
            )
            if "deterministic filler" in blob:
                errors.append(
                    f"case_studies.json: pilot padding in {c.get('prompt_id')!r}"
                )

    if ", </td>" in html or "<td class=\"num\">, </td>" in html:
        errors.append("index.html: broken empty table cell (`, </td>`)")

    if errors:
        print("ExactKV site claims check: FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("ExactKV site claims check: PASSED")
    print(f"  checked {len(REQUIRED_FILES)} files; hero + leaderboard + caveats present; no forbidden claims; no secrets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
