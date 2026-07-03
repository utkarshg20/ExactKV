#!/usr/bin/env python3
"""Build frozen research-release artifact bundle + SHA256SUMS for GitHub Release."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUNDLE_NAME = "exactkv-research-release-artifact-bundle.tar.gz"

# Public headline artifacts included in the frozen bundle (paths relative to repo root).
BUNDLE_PATHS = [
    "REPRODUCE.md",
    "README.md",
    "RELEASE.md",
    "LICENSE",
    "SHA256SUMS",
    "environment.yml",
    "Dockerfile",
    "docs/THREATS_TO_VALIDITY.md",
    "docs/CLAIM_BOUNDARIES.md",
    "docs/EVALUATOR_GUIDE.md",
    "docs/VERSIONING.md",
    "docs/METRIC_DEFINITIONS.md",
    "paper/ExactKV_Technical_Report.md",
    "reports/public_release/leaderboard_final.json",
    "reports/public_release/release_manifest.json",
    "reports/public_release/repro_command.sh",
    "reports/scale_7b/raw.json",
    "reports/external_panels/summary_all.json",
    "reports/systems/latency_microbench.json",
    "reports/systems/gpu_memory_trace.json",
    "site/data/leaderboard.json",
    "site/data/case_studies.json",
]


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def git_tag() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "v-release"


def write_manifest(staging: Path) -> None:
    manifest = {
        "schema": "exactkv.research_release.bundle_manifest.v1",
        "git_tag": git_tag(),
        "commit": git_head(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": [p for p in BUNDLE_PATHS if p != "SHA256SUMS"],
        "repro_entrypoint": "REPRODUCE.md",
        "claim_boundary": "Drift diagnostics and harness safety gates — not production serving or official benchmark scores.",
    }
    (staging / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def build() -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    staging = DIST / "artifact-bundle-staging"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    missing: list[str] = []
    for rel in BUNDLE_PATHS:
        if rel == "SHA256SUMS":
            continue
        src = ROOT / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())

    if missing:
        raise SystemExit(f"missing bundle inputs: {missing}")

    write_manifest(staging)

    bundle_path = DIST / BUNDLE_NAME
    if bundle_path.exists():
        bundle_path.unlink()

    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(staging, arcname="exactkv-research-release")

    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    sums_path = ROOT / "SHA256SUMS"
    sums_path.write_text(f"{digest}  {BUNDLE_NAME}\n", encoding="utf-8")

    print(f"Wrote {bundle_path} ({bundle_path.stat().st_size} bytes)")
    print(f"SHA256 {digest}")
    print(f"Wrote {sums_path}")
    return bundle_path


if __name__ == "__main__":
    build()
