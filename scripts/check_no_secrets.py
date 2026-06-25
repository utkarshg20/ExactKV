#!/usr/bin/env python3
"""Scan repository paths for leaked secrets (Gate R0)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

SCAN_DIRS = ("docs", "reports", "scripts", "exactkv", "tests")

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv-runpod",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".cache",
    "hub",
}

# Test modules that embed intentional scanner probe strings.
SKIP_FILES = frozenset({
    "test_no_secret_leakage.py",
    "test_public_claim_safety.py",
})

PLACEHOLDER_PATTERNS = (
    re.compile(r"hf_xxx\b", re.I),
    re.compile(r"hf_\.\.\.", re.I),
    re.compile(r"PASTE_YOUR_TOKEN", re.I),
    re.compile(r"<\s*your[-_]?token\s*>", re.I),
    re.compile(r"\$\{?HF_TOKEN\}?", re.I),
    re.compile(r"HF_TOKEN=\.\.\.", re.I),
    re.compile(r"HF_TOKEN=['\"]?\.\.\.['\"]?", re.I),
)

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("huggingface_token", re.compile(r"\bhf_[a-zA-Z0-9]{20,}\b")),
    ("hf_env_assignment", re.compile(r"HUGGING_FACE_HUB_TOKEN\s*=\s*['\"]?hf_[a-zA-Z0-9]{10,}", re.I)),
    ("hf_token_assignment", re.compile(r"HF_TOKEN\s*=\s*['\"]?hf_[a-zA-Z0-9]{10,}", re.I)),
    ("github_token", re.compile(r"\bghp_[a-zA-Z0-9]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b")),
]


def _is_placeholder(line: str) -> bool:
    return any(p.search(line) for p in PLACEHOLDER_PATTERNS)


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _is_placeholder(line):
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, label, line.strip()[:100]))
                break
    return hits


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirname in SCAN_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.name in SKIP_FILES:
                continue
            if _should_skip(path):
                continue
            if path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pt", ".bin", ".safetensors"}:
                continue
            files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for leaked secrets")
    parser.add_argument("--root", type=Path, default=_ROOT)
    args = parser.parse_args()

    total = 0
    print("ExactKV secret scan")
    for path in collect_files(args.root):
        hits = scan_file(path)
        if hits:
            rel = path.relative_to(args.root)
            print(f"\n{rel}:")
            for lineno, label, excerpt in hits:
                print(f"  L{lineno} [{label}] {excerpt}")
                total += 1
    if total:
        print(f"\nFAILED: {total} potential secret(s)")
        return 1
    print("PASSED: no secrets detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
