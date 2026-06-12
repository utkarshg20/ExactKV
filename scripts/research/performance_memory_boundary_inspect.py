#!/usr/bin/env python3
"""Diagnostic inspect for Experiment 027 — claim boundary only.

This script does **not** run timed benchmarks or report speed/memory savings.
It audits codebase and optional pilot artifacts for forbidden claim fields and
summarizes what ExactKV measures vs what remains forbidden.

Usage:
    python3 scripts/research/performance_memory_boundary_inspect.py
    python3 scripts/research/performance_memory_boundary_inspect.py --pilot-json reports/experiment_018_gpu_memory_pilot.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_CLAIM_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

ALLOWED_MEMORY_FIELDS = frozenset({
    "total_kv_footprint_bytes",
    "stored_kv_bytes",
    "materialized_kv_bytes",
    "metadata_bytes",
    "temporary_workspace_bytes",
})

PILOT_GPU_FIELDS = frozenset({
    "gpu_baseline_model_loaded_bytes",
    "gpu_allocated_after_prefill_bytes",
    "gpu_peak_allocated_during_run_bytes",
    "gpu_allocated_after_run_bytes",
    "gpu_allocated_after_cleanup_bytes",
})

SCAN_PATHS = [
    ROOT / "exactkv",
    ROOT / "scripts",
]

SKIP_DIRS = {"__pycache__", ".venv", ".git", "vendor"}


def _scan_python_for_forbidden(root: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for field in FORBIDDEN_CLAIM_FIELDS:
                if field in line and "FORBIDDEN" not in line.upper():
                    # Allow explicit negation / guard lists in metrics modules
                    if "frozenset" in line or "_FORBIDDEN" in line:
                        continue
                    if field in line and ("not " in line.lower() or "does not" in line.lower()):
                        continue
                    hits.append((str(path.relative_to(ROOT)), i, line.strip()[:120]))
    return hits


def _summarize_pilot_json(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(data)
    forbidden_in_blob = [f for f in FORBIDDEN_CLAIM_FIELDS if f'"{f}"' in blob]
    cells = data.get("cells") or data.get("results") or []
    n = len(cells) if isinstance(cells, list) else 0
    failures = sum(1 for c in cells if c.get("exactkv_failure")) if isinstance(cells, list) else None
    return {
        "status": "loaded",
        "path": str(path),
        "cell_count": n,
        "exactkv_failures": failures,
        "forbidden_fields_in_artifact": forbidden_in_blob,
        "note": "Pilot observations are diagnostic only — not VRAM savings proof.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 027 claim-boundary inspect (diagnostic only)")
    parser.add_argument(
        "--pilot-json",
        type=Path,
        default=ROOT / "reports" / "experiment_018_gpu_memory_pilot.json",
        help="Optional Exp 018 pilot artifact (gitignored)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 027 — performance/memory truth boundary inspect")
    print("DIAGNOSTIC ONLY — not a benchmark; no speed or savings claims")
    print("=" * 60)

    print("\n## Claim boundary summary\n")
    print("| Question | Answer |")
    print("|---|---|")
    print("| Measures tokens/sec? | **No** — forbidden in standard schema |")
    print("| Proves lower runtime? | **No** — no controlled timing study |")
    print("| Proves active GPU memory savings? | **No** — Exp 018 pilot; weights dominate peak |")
    print("| Stable memory metric? | V5 `total_kv_footprint_bytes` (accounting sum) |")
    print("| Verification model | Sequential one-token (`verify_sequential`) |")
    print("| Parallel span verify | **Not implemented** (D21 deferred) |")

    print("\n## Allowed vs forbidden fields\n")
    print("Allowed (V5 accounting):", ", ".join(sorted(ALLOWED_MEMORY_FIELDS)))
    print("Pilot-only GPU:", ", ".join(sorted(PILOT_GPU_FIELDS)))
    print("Forbidden launch claims:", ", ".join(sorted(FORBIDDEN_CLAIM_FIELDS)))

    print("\n## Code scan (suspicious forbidden-field references)\n")
    all_hits: list[tuple[str, int, str]] = []
    for scan_root in SCAN_PATHS:
        if scan_root.is_dir():
            all_hits.extend(_scan_python_for_forbidden(scan_root))
    if not all_hits:
        print("No suspicious forbidden-field usages in scanned Python (guard lists excluded).")
    else:
        for rel, line_no, snippet in all_hits[:30]:
            print(f"  {rel}:{line_no}: {snippet}")
        if len(all_hits) > 30:
            print(f"  ... and {len(all_hits) - 30} more")

    print("\n## Optional Exp 018 pilot artifact\n")
    pilot = _summarize_pilot_json(args.pilot_json)
    for k, v in pilot.items():
        print(f"  {k}: {v}")

    print("\n## Recommendation\n")
    print("  Speed and active GPU memory savings claims: **FORBIDDEN**")
    print("  Next phase: **V13 Practicality Proof** (parallel verify + isolated memory methodology)")
    print("  See: docs/EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md")
    print("       docs/PRACTICALITY_GAP_ANALYSIS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
