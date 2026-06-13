#!/usr/bin/env python3
"""Verify report git hygiene and allowed committed artifacts (Phase 9B)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# These must NOT appear in git ls-files
FORBIDDEN_TRACKED_GLOBS = (
    "reports/*.json",
    "reports/*.csv",
)

# Public artifacts that should exist when hygiene check runs in dev tree
EXPECTED_PUBLIC = (
    "docs/leaderboard.md",
    "docs/leaderboard.html",
)


def git_ls_files(root: Path, pattern: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", pattern],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]


def git_staged_forbidden(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    staged = result.stdout.splitlines()
    bad = []
    for path in staged:
        path = path.strip()
        if path.startswith("reports/") and (path.endswith(".json") or path.endswith(".csv")):
            bad.append(path)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Check report git hygiene")
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--require-public", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    failures = 0
    print("ExactKV report hygiene audit")

    for pattern in FORBIDDEN_TRACKED_GLOBS:
        tracked = git_ls_files(root, pattern)
        if tracked:
            print(f"\nFAIL: tracked forbidden files matching {pattern!r}:")
            for t in tracked:
                print(f"  {t}")
            failures += len(tracked)

    staged_bad = git_staged_forbidden(root)
    if staged_bad:
        print("\nFAIL: staged raw reports (unstage before commit):")
        for s in staged_bad:
            print(f"  {s}")
        failures += len(staged_bad)

    gitignore = root / ".gitignore"
    if gitignore.is_file():
        gi = gitignore.read_text(encoding="utf-8")
        for need in ("reports/*.json", "reports/*.csv"):
            if need not in gi:
                print(f"\nFAIL: .gitignore missing {need!r}")
                failures += 1
    else:
        print("\nFAIL: .gitignore not found")
        failures += 1

    if args.require_public:
        for rel in EXPECTED_PUBLIC:
            if not (root / rel).is_file():
                print(f"\nFAIL: expected public artifact missing: {rel}")
                failures += 1

    # reports/ dir should exist
    reports = root / "reports"
    if not reports.is_dir():
        print("\nWARN: reports/ directory missing (ok for minimal clone)")

    if failures:
        print(f"\nFAILED: {failures} hygiene issue(s)")
        return 1
    print("PASSED: report hygiene OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
