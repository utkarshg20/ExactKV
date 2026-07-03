#!/usr/bin/env python3
"""Verify GitHub has only the public v-release tag (fail CI if stale tags reappear)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

REPO = "utkarshg20/ExactKV"
ALLOWED = {"v-release"}


def fetch_tags_api() -> list[str]:
    url = f"https://api.github.com/repos/{REPO}/tags?per_page=100"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "exactkv-tag-verify",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return [item["name"] for item in data]


def fetch_tags_gh() -> list[str]:
    out = subprocess.check_output(
        ["gh", "api", f"repos/{REPO}/tags", "--paginate", "--jq", ".[].name"],
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def fetch_tags() -> list[str]:
    try:
        return fetch_tags_api()
    except Exception:
        return fetch_tags_gh()


def local_tags() -> set[str]:
    out = subprocess.check_output(["git", "tag", "-l"], text=True)
    return {line.strip() for line in out.splitlines() if line.strip()}


def main() -> int:
    errors: list[str] = []
    try:
        remote = set(fetch_tags())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not fetch remote tags: {exc}")
        remote = set()

    local = local_tags()

    if remote:
        extra = remote - ALLOWED
        missing = ALLOWED - remote
        if extra:
            errors.append(f"unexpected remote tags: {sorted(extra)}")
        if missing:
            errors.append(f"missing required remote tag(s): {sorted(missing)}")

    extra_local = local - ALLOWED
    if extra_local:
        errors.append(f"unexpected local tags: {sorted(extra_local)}")

    if errors:
        print("Git tag verification: FAILED")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("Git tag verification: PASSED (only v-release)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
