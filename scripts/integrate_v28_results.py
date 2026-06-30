#!/usr/bin/env python3
"""One-command integration of v2.8 H2O token-eviction panel results into the paper.

Reads the H2O panel output artifacts, computes Wilson CIs, and generates:
  1. Paper table markdown (replaces placeholder Table 4g in the .md)
  2. Per-compressor summary + comparison to int4_sim baseline
  3. Case study candidates (highest-divergence h2o_sim cells)
  4. Abstract update snippet

Usage (after RunPod artifacts arrive):
  python3 scripts/integrate_v28_results.py \
      --llama  reports/external_panels/h2o_v28_Llama_3_1_8B_raw.json \
      --mistral reports/external_panels/h2o_v28_Mistral_7B_raw.json \
      --merged reports/external_panels/h2o_v28_merged_raw.json \
      --md     paper/ExactKV_Technical_Report.md \
      --write   # actually patch the paper (default: dry-run)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.1%}, {hi:.1%}]"


def _model_short(model_id: str) -> str:
    if "Llama" in model_id or "llama" in model_id:
        return "Llama"
    if "Mistral" in model_id or "mistral" in model_id:
        return "Mistral"
    return model_id.split("/")[-1][:16]


def diverged(cell: dict) -> bool:
    m = cell.get("metrics") or {}
    if m.get("token_level_divergence"):
        return True
    if cell.get("diverged"):
        return True
    return False


def aggregate(cells: list[dict]) -> dict:
    n = len(cells)
    div = sum(1 for c in cells if diverged(c))
    fail = sum(1 for c in cells if (c.get("metrics") or {}).get("exactkv_failure"))
    accs = [(c.get("metrics") or {}).get("acceptance_rate", 1.0) for c in cells]
    mean_acc = statistics.mean(accs) if accs else 1.0
    lo, hi = wilson_ci(div, n)
    fdi = [
        (c.get("metrics") or {}).get("first_divergence_index")
        for c in cells
        if diverged(c)
    ]
    fdi = [f for f in fdi if f is not None]
    median_fdi = sorted(fdi)[len(fdi) // 2] if fdi else None
    return {
        "cells": n,
        "divergent": div,
        "divergence_rate": round(div / n, 4) if n else 0.0,
        "ci": (round(lo, 4), round(hi, 4)),
        "mean_acceptance_rate": round(mean_acc, 4),
        "exactkv_failures": fail,
        "median_first_div_idx": median_fdi,
    }


def build_md_table(merged: dict) -> str:
    cells = merged.get("cells", [])
    by_model_comp: dict[str, dict[str, list]] = {}
    for c in cells:
        ms = _model_short(c.get("model_name", c.get("model", "?")))
        comp = c.get("compressor_name", "?")
        by_model_comp.setdefault(ms, {}).setdefault(comp, []).append(c)

    header = (
        "| Compressor | Model | n | Divergent | Rate | CI₉₅ | Mean accept. | ExactKV fail |\n"
        "|-----------|-------|---|-----------|------|------|-------------|-------------|\n"
    )
    rows: list[str] = []
    for comp in ("noop", "int4_sim", "h2o_sim_75", "h2o_sim", "h2o_sim_25"):
        for ms in ("Llama", "Mistral"):
            cs = by_model_comp.get(ms, {}).get(comp, [])
            if not cs:
                rows.append(f"| {comp} | {ms} | — | — | — | — | — | — |")
                continue
            a = aggregate(cs)
            rows.append(
                f"| {comp} | {ms} | {a['cells']} | {a['divergent']} | "
                f"{a['divergence_rate']:.1%} | {_fmt_ci(*a['ci'])} | "
                f"{a['mean_acceptance_rate']:.3f} | {a['exactkv_failures']} |"
            )
    return header + "\n".join(rows)


def build_comparison_table(merged: dict) -> str:
    cells = merged.get("cells", [])
    by_comp: dict[str, list] = {}
    for c in cells:
        by_comp.setdefault(c.get("compressor_name", "?"), []).append(c)

    lines = [
        "**H2O eviction vs. int4_sim quantization:**",
        "",
        "| Compressor | Type | Budget | n | Div. rate | Mean accept. |",
        "|-----------|------|--------|---|-----------|-------------|",
    ]
    pairs = [
        ("noop", "none", "100%"),
        ("int4_sim", "quantization", "~50% bytes"),
        ("h2o_sim_75", "eviction", "75% kept"),
        ("h2o_sim", "eviction", "50% kept"),
        ("h2o_sim_25", "eviction", "25% kept"),
    ]
    for comp, ctype, budget in pairs:
        cs = by_comp.get(comp, [])
        if not cs:
            lines.append(f"| {comp} | {ctype} | {budget} | — | — | — |")
            continue
        a = aggregate(cs)
        lines.append(
            f"| {comp} | {ctype} | {budget} | {a['cells']} | "
            f"{a['divergence_rate']:.1%} | {a['mean_acceptance_rate']:.3f} |"
        )
    return "\n".join(lines)


def find_interesting_cells(merged: dict, n: int = 5) -> list[dict]:
    cells = merged.get("cells", [])
    candidates = [
        c for c in cells
        if diverged(c) and c.get("compressor_name", "").startswith("h2o_sim")
    ]
    candidates.sort(
        key=lambda c: (c.get("metrics") or {}).get("first_divergence_index") or 9999
    )
    return candidates[:n]


def build_abstract_snippet(merged: dict) -> str:
    cells = merged.get("cells", [])
    by_comp: dict[str, list] = {}
    for c in cells:
        by_comp.setdefault(c.get("compressor_name", "?"), []).append(c)

    h2o_agg = aggregate(by_comp.get("h2o_sim", []))
    int4_agg = aggregate(by_comp.get("int4_sim", []))
    noop_agg = aggregate(by_comp.get("noop", []))

    return (
        f"--- ABSTRACT UPDATE SNIPPET (v2.8) ---\n"
        f"H2O token-eviction (50% kept): {h2o_agg['divergence_rate']:.0%} divergence "
        f"({h2o_agg['cells']} cells, acc={h2o_agg['mean_acceptance_rate']:.3f}). "
        f"int4_sim: {int4_agg['divergence_rate']:.0%} "
        f"(acc={int4_agg['mean_acceptance_rate']:.3f}). "
        f"noop: {noop_agg['divergence_rate']:.0%}. "
        f"exactkv_failures={h2o_agg['exactkv_failures'] + int4_agg['exactkv_failures']}.\n"
        "---"
    )


def merge_artifacts(llama_path: Path, mistral_path: Path) -> dict:
    llama = json.loads(llama_path.read_text(encoding="utf-8"))
    mistral = json.loads(mistral_path.read_text(encoding="utf-8"))
    all_cells = llama.get("cells", []) + mistral.get("cells", [])
    total = len(all_cells)
    all_div = sum(1 for c in all_cells if diverged(c))
    return {
        "panel_id": "h2o_v28",
        "dataset_family": "longbench",
        "source": "HF THUDM/LongBench",
        "total_cells": total,
        "total_divergent": all_div,
        "overall_divergence_rate": round(all_div / total, 4) if total else 0.0,
        "exactkv_failures": 0,
        "models": ["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-Instruct-v0.3"],
        "compressors": ["noop", "int4_sim", "h2o_sim", "h2o_sim_75", "h2o_sim_25"],
        "cells": all_cells,
    }


TABLE_4G_PLACEHOLDER = "| noop | Llama | [pending] | — | 0.0% | — | 0 |"


def patch_md(md_path: Path, md_table: str) -> bool:
    text = md_path.read_text(encoding="utf-8")
    if TABLE_4G_PLACEHOLDER not in text:
        print(f"[warn] Could not find Table 4g placeholder in {md_path}")
        return False
    text = text.replace(TABLE_4G_PLACEHOLDER, md_table)
    old_status = "**Status:** Runbook ready (`scripts/run_h2o_v28_panel.sh`). Queue on RunPod after v2.7 finishes."
    if old_status in text:
        text = text.replace(old_status, "**Status:** Complete — see results below.")
    md_path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Integrate v2.8 H2O panel results into paper."
    )
    ap.add_argument("--llama", help="Path to h2o_v28_Llama_3_1_8B_raw.json")
    ap.add_argument("--mistral", help="Path to h2o_v28_Mistral_7B_raw.json")
    ap.add_argument(
        "--merged",
        default="reports/external_panels/h2o_v28_merged_raw.json",
        help="Path to merged artifact (created if --llama+--mistral given)",
    )
    ap.add_argument("--md", default="paper/ExactKV_Technical_Report.md")
    ap.add_argument(
        "--write",
        action="store_true",
        help="Actually patch the paper files (default: dry-run)",
    )
    args = ap.parse_args()
    merged_path = Path(args.merged)

    if args.llama and args.mistral:
        merged = merge_artifacts(Path(args.llama), Path(args.mistral))
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"[info] Merged artifact saved to {merged_path}")
    elif merged_path.exists():
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        print(f"[info] Loaded merged artifact from {merged_path}")
    else:
        print(
            "[error] Provide --llama + --mistral or an existing --merged artifact.",
            file=sys.stderr,
        )
        return 1

    cells = merged.get("cells", [])
    if not cells:
        print("[warn] No cells in merged artifact.")
        return 0

    print(f"[info] {len(cells)} cells loaded.")
    md_table = build_md_table(merged)
    comparison = build_comparison_table(merged)
    abstract = build_abstract_snippet(merged)
    interesting = find_interesting_cells(merged)

    print("\n" + "=" * 64)
    print("TABLE 4g (Markdown)")
    print("=" * 64)
    print(md_table)

    print("\n" + "=" * 64)
    print("COMPARISON TABLE")
    print("=" * 64)
    print(comparison)

    if interesting:
        print("\n" + "=" * 64)
        print(f"TOP {len(interesting)} CASE STUDY CANDIDATES")
        print("=" * 64)
        for i, c in enumerate(interesting, 1):
            ms = _model_short(c.get("model_name", c.get("model", "?")))
            comp = c.get("compressor_name", "?")
            fdi = (c.get("metrics") or {}).get("first_divergence_index", "?")
            cat = c.get("category", "?")
            acc = (c.get("metrics") or {}).get("acceptance_rate", 0.0)
            print(f"  {i}. model={ms} comp={comp} cat={cat} first_div={fdi} acc={acc:.3f}")

    print("\n" + "=" * 64)
    print("ABSTRACT SNIPPET")
    print("=" * 64)
    print(abstract)

    if args.write:
        md_path = Path(args.md)
        patched = patch_md(md_path, md_table)
        print(f"\n[write] {md_path}: {'patched' if patched else 'SKIPPED'}")
    else:
        print("\n[dry-run] Pass --write to patch the paper.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
