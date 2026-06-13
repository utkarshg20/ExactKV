#!/usr/bin/env python3
"""Verify local Markdown/HTML links point to existing files (Phase 9B)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

_ROOT = Path(__file__).resolve().parents[1]

# [text](url) and ![alt](url)
LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
# href="..." in HTML
HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.I)

SKIP_URL_PREFIXES = ("http://", "https://", "mailto:", "#", "data:")


def _resolve_link(source: Path, target: str, root: Path) -> Path | None:
    target = unquote(target.strip())
    if not target or target.startswith(SKIP_URL_PREFIXES):
        return None
    # Strip anchors
    target = target.split("#", 1)[0]
    if not target:
        return None
    if target.startswith("/"):
        return root / target.lstrip("/")
    return (source.parent / target).resolve()


def scan_file(path: Path, root: Path) -> list[tuple[int, str, str]]:
    missing: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [(0, str(path), str(e))]

    patterns: list[tuple[int, str]] = []
    for m in LINK_RE.finditer(text):
        patterns.append((text[: m.start()].count("\n") + 1, m.group(2)))
    if path.suffix == ".html":
        for m in HREF_RE.finditer(text):
            patterns.append((text[: m.start()].count("\n") + 1, m.group(1)))

    for lineno, raw in patterns:
        resolved = _resolve_link(path, raw, root)
        if resolved is None:
            continue
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            # Outside repo — skip
            continue
        if not resolved.exists():
            missing.append((lineno, raw, str(resolved.relative_to(root))))
    return missing


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = [root / "README.md"]
    docs = root / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.rglob("*.md")))
        files.extend(sorted(docs.rglob("*.html")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local doc links")
    parser.add_argument("--root", type=Path, default=_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    total = 0
    print("ExactKV docs link audit")
    for path in collect_files(root):
        rel = path.relative_to(root)
        hits = scan_file(path, root)
        if hits:
            print(f"\n{rel}:")
            for lineno, raw, resolved in hits:
                print(f"  L{lineno} {raw!r} -> missing {resolved}")
                total += 1
    if total:
        print(f"\nFAILED: {total} broken local link(s)")
        return 1
    print("PASSED: all local links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
