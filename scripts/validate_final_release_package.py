#!/usr/bin/env python3
"""Validate the ExactKV final release synthesis package (Part 9).

Checks presence of all generated artifacts, claim safety of public copy
(negation-aware), required caveats, claim->evidence mapping, secret hygiene, and
the PDF export-status explanation.

Exit 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    # Paper
    "paper/ExactKV_Technical_Report.md",
    "paper/ExactKV_Technical_Report.tex",
    "paper/references.bib",
    "paper/export_status.json",
    # Website
    "site/index.html",
    "site/styles.css",
    "site/main.js",
    "site/README.md",
    "site/content_manifest.json",
    "site/claim_safe_copy.json",
    # Launch
    "launch/x_thread.md",
    "launch/linkedin_post.md",
    "launch/short_announcement.md",
    "launch/launch_manifest.json",
    # Release synthesis
    "release_synthesis/artifact_inventory.md",
    "release_synthesis/artifact_inventory.json",
    "release_synthesis/artifact_inventory.csv",
    "release_synthesis/project_lineage.md",
    "release_synthesis/version_lineage.md",
    "release_synthesis/phase_lineage.md",
    "release_synthesis/source_of_truth_map.md",
    "release_synthesis/project_lineage.json",
    "release_synthesis/evidence_ledger.md",
    "release_synthesis/evidence_ledger.json",
    "release_synthesis/claim_decision_table.md",
    "release_synthesis/claim_decision_table.json",
    "release_synthesis/related_work_audit.md",
    "release_synthesis/references.bib",
    "release_synthesis/final_release_checklist.md",
    # GitHub/release
    "RELEASE.md",
]

# Public-facing copy scanned for forbidden positive claims + required caveats.
PUBLIC_COPY = [
    "paper/ExactKV_Technical_Report.md",
    "site/index.html",
    "launch/x_thread.md",
    "launch/linkedin_post.md",
    "launch/short_announcement.md",
    "RELEASE.md",
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

# Each public-copy file must contain at least one phrase from each caveat group.
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

    # 1. Required files exist.
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required artifact: {rel}")

    # 2. Public copy: forbidden claims + caveats + secrets.
    for rel in PUBLIC_COPY:
        p = ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        # Strip markdown emphasis/code markers so caveat phrases match across
        # bold/italic formatting (e.g. "does **not** reproduce VeriCache").
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
            if not any(ph in lower for ph in phrases):
                errors.append(f"{rel} missing caveat: {label}")

    # 3. Claim->evidence mapping present and non-trivial.
    cdt_path = ROOT / "release_synthesis/claim_decision_table.json"
    if cdt_path.is_file():
        cdt = json.loads(cdt_path.read_text(encoding="utf-8"))
        claims = cdt.get("claims", [])
        if len(claims) < 15:
            errors.append(f"claim_decision_table.json has too few claims ({len(claims)})")
        for c in claims:
            if not c.get("evidence_artifact"):
                errors.append(f"claim missing evidence_artifact: {c.get('claim')}")
            if c.get("decision") not in {"allowed", "allowed_with_qualification", "forbidden"}:
                errors.append(f"claim has invalid decision: {c.get('claim')}")
        # Spot-check critical forbidden claims.
        forbidden_claims = {c["claim"].lower() for c in claims if c.get("decision") == "forbidden"}
        for must in ("end-to-end speedup", "active gpu memory", "reproduces vericache",
                     "production ready", "unique / first ever"):
            if not any(must in fc for fc in forbidden_claims):
                errors.append(f"expected a forbidden claim covering: {must}")

    led_path = ROOT / "release_synthesis/evidence_ledger.json"
    if led_path.is_file():
        led = json.loads(led_path.read_text(encoding="utf-8"))
        if led.get("benchmark_source_of_truth") != SOURCE_OF_TRUTH:
            errors.append("evidence_ledger source-of-truth is not reports/scale_7b/raw.json")
        if led.get("headline_facts", {}).get("exactkv_failures") != 0:
            errors.append("evidence_ledger headline exactkv_failures != 0")

    # 4. Source of truth exists.
    if not (ROOT / SOURCE_OF_TRUTH).is_file():
        errors.append(f"benchmark source of truth missing: {SOURCE_OF_TRUTH}")

    # 5. PDF export status explains absence if PDF not generated.
    es_path = ROOT / "paper/export_status.json"
    if es_path.is_file():
        es = json.loads(es_path.read_text(encoding="utf-8"))
        if not es.get("pdf_generated", False) and not es.get("pdf_reason"):
            errors.append("export_status.json: pdf not generated but no pdf_reason given")
        if not (ROOT / "paper/ExactKV_Technical_Report.pdf").is_file() and es.get("pdf_generated"):
            warnings.append("export_status claims pdf_generated but no PDF file found")

    # Report
    print("ExactKV final release package validation")
    print(f"  checked {len(REQUIRED_FILES)} artifacts; {len(PUBLIC_COPY)} public-copy files")
    for w in warnings:
        print(f"  WARN: {w}")
    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPASSED: all artifacts present, claim-safe, evidence-mapped, no secrets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
