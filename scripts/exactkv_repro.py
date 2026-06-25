#!/usr/bin/env python3
"""ExactKV reproducibility wrapper (Phase J)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CLAIM_BOUNDARY_NOTE = (
    "ExactKV is a research-grade compressor-agnostic KV exactness benchmark. "
    "Not production serving. No end-to-end speedups. No active GPU memory savings claims."
)


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _torch_info() -> tuple[str, bool, bool]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        triton = False
        try:
            import triton  # noqa: F401

            triton = True
        except ImportError:
            pass
        return torch.__version__, cuda, triton
    except ImportError:
        return "not_installed", False, False


def _run(cmd: list[str], *, cwd: Path, manifest: dict[str, Any], skip_gpu: bool = False) -> bool:
    label = " ".join(cmd)
    if skip_gpu and any(x in label for x in ("phaseF", "scale_7b", "full-scale-7b", "cuda")):
        manifest["skipped_steps"].append(f"skipped_gpu: {label}")
        return True
    print(f">>> {label}")
    proc = subprocess.run(cmd, cwd=cwd)
    manifest["commands_run"].append(label)
    if proc.returncode != 0:
        manifest["failures"].append(f"{label} exit={proc.returncode}")
        return False
    return True


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _base_manifest(root: Path) -> dict[str, Any]:
    torch_v, cuda, triton = _torch_info()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "python_version": sys.version.split()[0],
        "torch_version": torch_v,
        "cuda_available": cuda,
        "triton_available": triton,
        "commands_run": [],
        "reports_generated": [],
        "skipped_steps": [],
        "failures": [],
        "release_artifacts_checked": [],
        "claim_boundary_note": CLAIM_BOUNDARY_NOTE,
    }


def cmd_reports_only(root: Path, manifest: dict[str, Any]) -> bool:
    ok = True
    ok &= _run([sys.executable, "scripts/build_project_lineage.py"], cwd=root, manifest=manifest)
    ok &= _run([sys.executable, "scripts/build_launch_pack.py"], cwd=root, manifest=manifest)
    ok &= _run([sys.executable, "scripts/exactkv.py", "run", "publish"], cwd=root, manifest=manifest)
    ok &= _run([sys.executable, "scripts/run_novelty_audit.py"], cwd=root, manifest=manifest)
    ok &= _run([sys.executable, "scripts/check_release_evidence.py"], cwd=root, manifest=manifest)
    manifest["reports_generated"].extend(
        [
            "reports/public_release/",
            "docs/NOVELTY_AUDIT.md",
            "reports/novelty_audit.json",
            "reports/release_evidence_status.json",
            "docs/PROJECT_LINEAGE.md",
            "docs/HISTORICAL_ARTIFACT_INVENTORY.md",
            "reports/historical_artifact_inventory.json",
            "reports/project_lineage_graph.json",
            "reports/public_release/demo_cards.json",
            "reports/public_release/launch_manifest.json",
        ]
    )
    return ok


def cmd_release_check(root: Path, manifest: dict[str, Any]) -> bool:
    ok = True
    checks = [
        [sys.executable, "scripts/check_release_evidence.py"],
        [sys.executable, "scripts/check_no_secrets.py"],
        [sys.executable, "scripts/audit_public_claims.py"],
        [sys.executable, "scripts/check_public_release.py"],
        [sys.executable, "scripts/check_project_lineage.py"],
        [sys.executable, "scripts/check_launch_pack.py"],
        [sys.executable, "-m", "pytest", "tests/test_public_release_artifacts.py", "-q"],
    ]
    for cmd in checks:
        ok &= _run(cmd, cwd=root, manifest=manifest)
    manifest["release_artifacts_checked"] = [
        "reports/public_release/README_PUBLIC.md",
        "reports/public_release/release_manifest.json",
        "reports/scale_7b/raw.json",
    ]
    return ok


def cmd_quick(root: Path, manifest: dict[str, Any]) -> bool:
    ok = True
    for cmd in (
        [sys.executable, "scripts/check_no_secrets.py"],
        [sys.executable, "scripts/audit_public_claims.py"],
        [sys.executable, "scripts/check_public_release.py"],
        [sys.executable, "-m", "pytest", "tests/test_public_claim_safety.py", "tests/test_novelty_audit.py", "-q"],
    ):
        ok &= _run(cmd, cwd=root, manifest=manifest)
    return ok


def cmd_full(root: Path, manifest: dict[str, Any], *, confirm_expensive: bool, skip_gpu: bool) -> bool:
    if not confirm_expensive:
        print(
            "ERROR: --full may run expensive GPU benchmarks. "
            "Pass --confirm-expensive to proceed, or use --release-check / --reports-only.",
            file=sys.stderr,
        )
        manifest["failures"].append("refused --full without --confirm-expensive")
        return False
    ok = cmd_reports_only(root, manifest)
    if not skip_gpu:
        ok &= _run(
            [sys.executable, "scripts/exactkv.py", "run", "full-scale-7b"],
            cwd=root,
            manifest=manifest,
        )
    else:
        manifest["skipped_steps"].append("full-scale-7b inference (skip_gpu)")
    ok &= cmd_release_check(root, manifest)
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ExactKV reproducibility wrapper")
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--manifest-out", type=Path, default=_ROOT / "reports" / "repro_manifest.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--reports-only", action="store_true")
    mode.add_argument("--release-check", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--confirm-expensive", action="store_true")
    args = parser.parse_args(argv)

    manifest = _base_manifest(args.root)
    if not any((args.quick, args.reports_only, args.release_check, args.full)):
        args.release_check = True

    ok = True
    if args.quick:
        ok = cmd_quick(args.root, manifest)
    elif args.reports_only:
        ok = cmd_reports_only(args.root, manifest)
    elif args.release_check:
        ok = cmd_release_check(args.root, manifest)
    elif args.full:
        ok = cmd_full(args.root, manifest, confirm_expensive=args.confirm_expensive, skip_gpu=args.skip_gpu)

    _write_manifest(args.manifest_out, manifest)
    print(f"wrote {args.manifest_out}")
    return 0 if ok and not manifest["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
