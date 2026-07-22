#!/usr/bin/env python3
"""Validate the public ExactKV release package on GitHub.

Checks presence of paper, site, benchmark artifacts, claim safety of public copy,
and secret hygiene. Internal launch copy (`launch/`) and release synthesis
(`release_synthesis/`) are local-only and not required here.

Exit 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "paper/ExactKV_Technical_Report.md",
    "paper/ExactKV_Technical_Report.tex",
    "paper/references.bib",
    "paper/export_status.json",
    "site/index.html",
    "site/styles.css",
    "site/main.js",
    "site/README.md",
    "site/content_manifest.json",
    "site/claim_safe_copy.json",
    "site/data/leaderboard.json",
    "site/data/case_studies.json",
    "site/assets/og_card.png",
    "site/assets/exactkv_logo.png",
    "site/assets/exactkv_icon.png",
    "site/assets/public_exactkv_one_page_summary.png",
    "site/assets/exp035_first_divergence_histogram.png",
    "site/assets/exp035_category_heatmap.png",
    "reports/public_release/leaderboard_final.json",
    "reports/public_release/README_PUBLIC.md",
    "RELEASE.md",
    "README.md",
    "REPRODUCE.md",
    "environment.yml",
    "Dockerfile",
    "SHA256SUMS",
    "docs/CLAIM_BOUNDARIES.md",
    "docs/THREATS_TO_VALIDITY.md",
    "docs/ARTIFACT_AUDIT.md",
    "reports/public_release/confidence_intervals.json",
    "reports/systems/latency_microbench.json",
    "reports/systems/gpu_memory_trace.json",
    "reports/systems/verifier_overhead.json",
    "reports/systems/recompression_overhead.json",
]

PUBLIC_COPY = [
    "paper/ExactKV_Technical_Report.md",
    "site/index.html",
    "RELEASE.md",
    "README.md",
]

ALLOW = re.compile(
    r"\b(not|no|without|does not|don't|never|isn't|fallback|proxy|probe|"
    r"microbenchmark|stored tensor|not a|not real|inspired by|forbidden|do not)\b",
    re.I,
)

FORBIDDEN = [
    (re.compile(r"\bfirst ever\b", re.I), "first ever"),
    (re.compile(r"\bfirst and only\b", re.I), "first and only"),
    (re.compile(r"\bnothing like this exists\b", re.I), "nothing like this exists"),
    (re.compile(r"\bproduction[- ]ready\b", re.I), "production ready"),
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
    (re.compile(r"\b10x compression\b", re.I), "10x compression"),
    (re.compile(r"\b\d+(\.\d+)?x\s+speedup\b", re.I), "Nx speedup (unqualified)"),
]

CAVEAT_GROUPS = [
    (["kernel microbenchmark", "microbenchmark"], "Phase F microbenchmark caveat"),
    (["stored tensor byte ratio", "stored byte ratio", "stored tensor"], "compression ratio caveat"),
    (["fallback/proxy", "fallback", "proxy"], "SpectralQuant fallback caveat"),
    (["probe-first", "probe"], "Shard probe caveat"),
    (["does not reproduce vericache", "not reproduce", "not a vericache"], "VeriCache caveat"),
    (["not a production serving system", "not a production", "not production"], "production caveat"),
]

SECRETS = [
    (re.compile(r"\b(sk|pk)-[A-Za-z0-9]{20,}\b"), "API key"),
    (re.compile(r"\bgh[posu]_[A-Za-z0-9]{20,}\b"), "GitHub token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "HuggingFace token"),
]

SOURCE_OF_TRUTH = "reports/scale_7b/raw.json"


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required artifact: {rel}")

    for rel in PUBLIC_COPY:
        p = ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        lower = re.sub(r"[*_`]", "", text).lower()
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, label in SECRETS:
                if pat.search(line):
                    errors.append(f"{rel}:L{lineno} possible secret [{label}]")
            if ALLOW.search(line):
                continue
            for pat, label in FORBIDDEN:
                if pat.search(line):
                    errors.append(f"{rel}:L{lineno} forbidden claim [{label}]: {line.strip()[:80]}")
        for phrases, label in CAVEAT_GROUPS:
            if rel == "README.md" and "production caveat" in label:
                continue
            if not any(ph in lower for ph in phrases):
                errors.append(f"{rel} missing caveat: {label}")

    if not (ROOT / SOURCE_OF_TRUTH).is_file():
        errors.append(f"benchmark source of truth missing: {SOURCE_OF_TRUTH}")

    lb_path = ROOT / "reports/public_release/leaderboard_final.json"
    if lb_path.is_file():
        lb = json.loads(lb_path.read_text(encoding="utf-8"))
        if lb.get("validation_result", {}).get("valid") is False:
            errors.append("leaderboard_final.json validation_result.valid is false")

    es_path = ROOT / "paper/export_status.json"
    if es_path.is_file():
        es = json.loads(es_path.read_text(encoding="utf-8"))
        if not es.get("pdf_generated", False) and not es.get("pdf_reason"):
            errors.append("export_status.json: pdf not generated but no pdf_reason given")

    print("ExactKV public release validation")
    print(f"  checked {len(REQUIRED_FILES)} artifacts; {len(PUBLIC_COPY)} public-copy files")
    for w in warnings:
        print(f"  WARN: {w}")
    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPASSED: public package present, claim-safe, no secrets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
