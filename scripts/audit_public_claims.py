#!/usr/bin/env python3
"""Scan public-facing docs for forbidden positive claims (Phase 9B).

Conservative phrase matching with negation/caveat allowlists.
Exit 0 if clean, 1 if violations found.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Public-facing files for strict claims audit (internal research docs excluded).
PUBLIC_SCAN_REL = (
    "README.md",
    "docs/INSTALL.md",
    "docs/QUICKSTART.md",
    "docs/leaderboard.md",
    "docs/leaderboard.html",
    "docs/PUBLIC_VISUAL_PACKAGE.md",
    "docs/EXACTKV_TERMINAL_CRASH_TEST.md",
    "docs/EXACTKV_CRASH_TEST_VIDEO.md",
    "docs/DEMO_EXACTKV_LIVE_CORRECTION.md",
    "docs/PRELAUNCH_HARDENING_REPORT.md",
)

# Files that list forbidden terms — skip if discovered via rglob.
SKIP_FILES = frozenset({
    "CLAIMS_AUDIT.md",
    "LAUNCH_READINESS_GAP_AUDIT.md",
    "PRELAUNCH_HARDENING_PLAN.md",
    "REPRO_CHECKLIST.md",
    "METRICS.md",
})

SKIP_GLOBS = (
    "EXPERIMENT_*.md",
    "RELEASE_NOTES_*.md",
    "PRIVATE_*.md",
)

# Line matches any of these → not a positive forbidden claim.
ALLOWLIST_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bno\b.{0,40}\b(speedup|faster|throughput|latency|tokens/?sec|runtime|vram|serving)\b",
        r"\bnot\b.{0,50}\b(speedup|faster|throughput|latency|serving|ready|integrated|production)\b",
        r"\b(does|do)\s+not\b.{0,40}\b(claim|measure|report|produce)\b",
        r"\bwithout\b.{0,30}\b(speedup|exactness)\b",
        r"\bforbidden\b",
        r"\bdeferred\b",
        r"\bnot\s+integrated\b",
        r"\bnot\s+approved\b",
        r"\bnot\s+ready\b",
        r"\bno-go\b",
        r"\bsmoke[- ]only\b",
        r"\brestricted\b",
        r"\bfactory-only\b",
        r"\bdiagnostic\s+only\b",
        r"\bfuture\s+work\b",
        r"\bnot\s+universal\b",
        r"\bnot\s+a\s+.*benchmark\b",
        r"\bnot\s+production\b",
        r"\bnot\s+exactkv\s+results?\b",
        r"\bexternal\b.{0,30}\bnot\s+exactkv\b",
        r"\bnegat",
        r"\bavoid\b",
        r"\bdo\s+not\b",
        r"\bdon'?t\b",
        r"\bnever\b",
        r"\bwithout\s+exactness\b",
        r"\bif\s+this\s+is\s+nonzero\b",
        r"\bspeedup\s+number\s+without\b",
        r"\bnot\s+an\s+exactkv\s+result\b",
        r"\bnot\s+real\s+packed\b",
        r"\b_sim\b",
        r"\bsimulated\b",
        r"\bint8\s+containers?\b",
        r"\|\s*Forbidden\b",
        r"^#+\s",
        r"^\|.*forbidden",
        r"^\s*[-*]\s*❌",
        r"forbidden\s+claim",
        r"forbidden\s+positive",
        r"forbidden\s+metrics",
        r"forbidden\s+fields",
        r"not\s+public[- ]launch",
        r"public\s+launch\s+(is\s+)?not",
        r"v1\.0(\.0)?\s+(is\s+)?not",
        r"not\s+v1\.0",
        r"launch\s+not\s+approved",
        r"not\s+approved",
        r"prelaunch",
        r"audit",
        r"^\s*-\s+",
        r"^\s*>\s+",
        r"do\s+not\s+(interpret|cite|report|imply|claim)",
        r"does\s+not\s+report",
        r"they\s+do\s+not\s+report",
        r"\*\*not\*\*",
        r"forbidden\s+claims?",
        r"claims?\s+boundary",
        r"tokens/second,\s*latency,\s*throughput,\s*or\s+speedup",
        r"no\s+throughput,\s*latency",
        r"not\s+production[- ]ready",
        r"not\s+public[- ]launch",
    )
]

# (pattern, human label) — positive-claim detectors.
FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpublic[- ]launch[- ]ready\b", re.I), "public launch ready"),
    (re.compile(r"\bv1\.0[- ]ready\b", re.I), "v1.0 ready"),
    (re.compile(r"\bproduction[- ]ready\s+serving\b", re.I), "production-ready serving"),
    (re.compile(r"\bproduction\s+serving\b", re.I), "production serving (positive)"),
    (re.compile(r"\bthroughput\s+improvement\b", re.I), "throughput improvement"),
    (re.compile(r"\blatency\s+improvement\b", re.I), "latency improvement"),
    (re.compile(r"\btokens/?sec\s+improvement\b", re.I), "tokens/sec improvement"),
    (re.compile(r"\bruntime\s+improvement\b", re.I), "runtime improvement"),
    (re.compile(r"\bactive\s+gpu\s+memory\s+savings\b", re.I), "active GPU memory savings"),
    (re.compile(r"\bvram\s+savings\b", re.I), "VRAM savings"),
    (re.compile(r"\bvram\s+reduction\b", re.I), "VRAM reduction"),
    (re.compile(r"\bvllm\s+integration\b", re.I), "vLLM integration"),
    (re.compile(r"\blmcache\s+integration\b", re.I), "lmcache integration"),
    (re.compile(r"\bpagedattention\s+integration\b", re.I), "pagedattention integration"),
    (re.compile(r"\bmodel\s+accuracy\s+improvement\b", re.I), "model accuracy improvement"),
    (re.compile(r"\bshard\s+exactkv\s+results?\b", re.I), "Shard ExactKV results"),
    (re.compile(r"\bspectralquant\s+exactkv\s+results?\b", re.I), "SpectralQuant ExactKV results"),
    (re.compile(r"\bsnapkv\s+full[- ]suite\b", re.I), "SnapKV full-suite"),
    (re.compile(r"\bpacked\s+int4\b(?!\s+memory\s+savings)", re.I), "packed INT4"),
    (re.compile(r"\bpacked\s+int2\b", re.I), "packed INT2"),
    (re.compile(r"\buniversal\s+benchmark\s+coverage\b", re.I), "universal benchmark coverage"),
    (re.compile(r"\b\d+(\.\d+)?x\s+speedup\b", re.I), "Nx speedup"),
    (re.compile(r"\bspeedup\s+over\s+full\b", re.I), "speedup over full"),
    (re.compile(r"\blossy\s+speedup\b", re.I), "lossy speedup"),
    (re.compile(r"\bfaster\s+than\s+full\b", re.I), "faster than full"),
    (re.compile(r"\bimproves?\s+throughput\b", re.I), "improves throughput"),
    (re.compile(r"\bimproves?\s+latency\b", re.I), "improves latency"),
    (re.compile(r"\bclaims?\s+speedup\b", re.I), "claims speedup"),
]

# Standalone "speedup" / "faster" need extra care.
EXTRA_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<![\w-])\bspeedup\b(?![\s:]*\s*(claim|number|without))", re.I), "speedup (positive)"),
    (re.compile(r"\bis\s+faster\b", re.I), "is faster"),
    (re.compile(r"\bgets?\s+faster\b", re.I), "gets faster"),
    (re.compile(r"\bfaster\s+inference\b", re.I), "faster inference"),
]

# In disclaimer sections, allow production serving mentions on list lines.
_DISCLAIMER_SECTION = re.compile(
    r"(forbidden|claims boundary|does not claim|not claim|no speedup)",
    re.I,
)


def _should_skip_file(path: Path) -> bool:
    if path.name in SKIP_FILES:
        return True
    for glob in SKIP_GLOBS:
        if path.match(glob):
            return True
    return False


def _is_allowlisted(line: str) -> bool:
    return any(p.search(line) for p in ALLOWLIST_PATTERNS)


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    violations: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return violations
    lines = text.splitlines()
    in_disclaimer = False
    for lineno, line in enumerate(lines, start=1):
        if _DISCLAIMER_SECTION.search(line):
            in_disclaimer = True
        if line.startswith("## ") and "forbidden" not in line.lower():
            if "claims" not in line.lower() and "boundary" not in line.lower():
                in_disclaimer = False
        if _is_allowlisted(line):
            continue
        if in_disclaimer and line.strip().startswith("-"):
            continue
        for pattern, label in FORBIDDEN_PATTERNS + EXTRA_PATTERNS:
            if pattern.search(line):
                violations.append((lineno, label, line.strip()[:120]))
                break
    return violations


def collect_scan_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for rel in PUBLIC_SCAN_REL:
        p = root / rel
        if p.is_file():
            paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit public docs for forbidden claims")
    parser.add_argument("--root", type=Path, default=_ROOT)
    args = parser.parse_args()

    paths = collect_scan_paths(args.root)
    total = 0
    print("ExactKV public claims audit")
    print(f"Scanning {len(paths)} file(s)...")
    for path in paths:
        rel = path.relative_to(args.root)
        hits = scan_file(path)
        if hits:
            print(f"\n{rel}:")
            for lineno, label, excerpt in hits:
                print(f"  L{lineno} [{label}] {excerpt}")
                total += 1
    if total:
        print(f"\nFAILED: {total} potential forbidden positive claim(s)")
        return 1
    print("PASSED: no forbidden positive claims detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
