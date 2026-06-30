#!/usr/bin/env python3
"""Cross-panel divergence analysis: task-type and context-length sensitivity.

Reads all ExactKV panel artifacts and builds a comprehensive table showing
int4_sim (and optionally other compressor) divergence rates across task families,
context lengths, and generation lengths.

Usage:
  python3 scripts/analyze_panel_divergence.py
  python3 scripts/analyze_panel_divergence.py --compressor int8
  python3 scripts/analyze_panel_divergence.py --markdown  # output as markdown table
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def diverged(c: dict) -> bool:
    m = c.get("metrics") or {}
    return bool(m.get("token_level_divergence")) or bool(c.get("diverged"))


def analyse_cells(cells: list[dict], comp_filter: str) -> dict | None:
    subset = [c for c in cells if c.get("compressor_name") == comp_filter
              and not c.get("probe_only")]
    if not subset:
        return None
    n = len(subset)
    div = sum(1 for c in subset if diverged(c))
    lo, hi = wilson_ci(div, n)
    import statistics
    accs = [(c.get("metrics") or {}).get("acceptance_rate", 1.0) for c in subset]
    mean_acc = statistics.mean(accs) if accs else 1.0
    fdi = [(c.get("metrics") or {}).get("first_divergence_index")
           for c in subset if diverged(c)]
    fdi = [f for f in fdi if f is not None]
    return {
        "n": n, "div": div,
        "rate": div / n if n else 0.0,
        "ci_lo": lo, "ci_hi": hi,
        "mean_acc": mean_acc,
        "median_fdi": sorted(fdi)[len(fdi) // 2] if fdi else None,
    }


def load_panel(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get("cells", [])
    except Exception:
        return []


PANELS = [
    {
        "name": "Headline (mixed tasks)",
        "path": "reports/scale_7b/raw.json",
        "task_type": "mixed",
        "context": "~500",
        "mnt": "16/32",
        "models": "Llama+Mistral",
    },
    {
        "name": "Evidence-plus",
        "path": "reports/evidence_plus/raw.json",
        "task_type": "mixed",
        "context": "512/1024",
        "mnt": "16",
        "models": "Llama+Mistral",
    },
    {
        "name": "External smoke (MBPP code)",
        "path": "reports/external_panels/mbpp_gpu_raw.json",
        "task_type": "code",
        "context": "512/1024",
        "mnt": "16/32",
        "models": "Llama+Mistral",
    },
    {
        "name": "BFCL export-50 (tool-call)",
        "path": "reports/external_panels/bfcl_export_50_raw.json",
        "task_type": "tool-call",
        "context": "1K/2K",
        "mnt": "16/32",
        "models": "Llama+Mistral",
    },
    {
        "name": "BFCL validity v2.7 Llama",
        "path": "reports/external_panels/bfcl_validity_v27_Llama_3_1_8B_raw.json",
        "task_type": "tool-call",
        "context": "1K/2K",
        "mnt": "128/256",
        "models": "Llama",
    },
    {
        "name": "HF LongBench v2.6 (open-text)",
        "path": "reports/external_panels/hf_longbench_v26_merged_raw.json",
        "task_type": "reading/summarization",
        "context": "2K/4K/8K",
        "mnt": "32/64",
        "models": "Llama+Mistral",
    },
    {
        "name": "H2O eviction v2.8",
        "path": "reports/external_panels/h2o_v28_merged_raw.json",
        "task_type": "reading/summarization",
        "context": "2K/4K/8K",
        "mnt": "32/64",
        "models": "Llama+Mistral",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-panel divergence analysis.")
    ap.add_argument("--compressor", default="int4_sim")
    ap.add_argument(
        "--markdown", action="store_true", help="Output as markdown table"
    )
    ap.add_argument(
        "--all-compressors", action="store_true",
        help="Show noop + int8 + int4_sim side by side"
    )
    args = ap.parse_args()

    compressors = ["noop", "int8", "int4_sim"] if args.all_compressors else [args.compressor]
    rows = []

    for panel in PANELS:
        path = REPO_ROOT / panel["path"]
        cells = load_panel(path)
        if not cells:
            rows.append({**panel, "missing": True})
            continue
        row = {**panel, "missing": False}
        for comp in compressors:
            a = analyse_cells(cells, comp)
            row[comp] = a
        rows.append(row)

    if args.markdown:
        if args.all_compressors:
            print(
                f"| Panel | Task type | Context | mnt | Models | "
                f"noop div | int8 div | int4_sim div | EKV fail |\n"
                f"|-------|-----------|---------|-----|--------|"
                f"---------|---------|-------------|---------|"
            )
            for r in rows:
                if r.get("missing"):
                    print(f"| {r['name']} | — | — | — | — | (missing) | — | — | — |")
                    continue
                noop = r.get("noop") or {}
                int8 = r.get("int8") or {}
                int4 = r.get("int4_sim") or {}
                print(
                    f"| {r['name']} | {r['task_type']} | {r['context']} | {r['mnt']} | "
                    f"{r['models']} | "
                    f"{noop.get('rate', 0):.1%} | "
                    f"{int8.get('rate', 0):.1%} | "
                    f"**{int4.get('rate', 0):.1%}** | "
                    f"0 |"
                )
        else:
            comp = args.compressor
            print(
                f"| Panel | Task type | Context | mnt | Models | "
                f"{comp} div | CI₉₅ | Mean acc |\n"
                f"|-------|-----------|---------|-----|--------|"
                f"---------|------|---------|"
            )
            for r in rows:
                if r.get("missing"):
                    print(f"| {r['name']} | — | — | — | — | (missing) | — | — |")
                    continue
                a = r.get(comp) or {}
                lo, hi = a.get("ci_lo", 0), a.get("ci_hi", 1)
                print(
                    f"| {r['name']} | {r['task_type']} | {r['context']} | {r['mnt']} | "
                    f"{r['models']} | "
                    f"**{a.get('rate', 0):.1%}** | "
                    f"[{lo:.1%}, {hi:.1%}] | "
                    f"{a.get('mean_acc', 1.0):.3f} |"
                )
    else:
        comp = args.compressor
        print(
            f"\n{'Panel':<42} {'Task':<20} {'Ctx':<8} {'mnt':<8} "
            f"{comp+' div':>10} {'CI95':>20} {'mean_acc':>9}"
        )
        print("-" * 120)
        for r in rows:
            if r.get("missing"):
                print(f"  {r['name']:<40} (artifact not found)")
                continue
            a = r.get(comp) or {}
            if a is None:
                print(f"  {r['name']:<40} (no {comp} cells)")
                continue
            lo, hi = a.get("ci_lo", 0), a.get("ci_hi", 1)
            print(
                f"  {r['name']:<40} {r['task_type']:<20} {r['context']:<8} {r['mnt']:<8} "
                f"{a.get('rate', 0):>9.1%}  "
                f"[{lo:.1%},{hi:.1%}]:>18  "
                f"{a.get('mean_acc', 1.0):>9.3f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
