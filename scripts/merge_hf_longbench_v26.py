#!/usr/bin/env python3
"""Merge Llama + Mistral HF LongBench v2.6 panel artifacts.

Combines two per-model raw.json files into a single merged JSON and
prints a per-subset, per-compressor breakdown with Wilson CIs.

Usage:
  python3 scripts/merge_hf_longbench_v26.py \
      --llama  reports/external_panels/hf_longbench_v26_Llama_3_1_8B_raw.json \
      --mistral reports/external_panels/hf_longbench_v26_Mistral_7B_raw.json \
      --output  reports/external_panels/hf_longbench_v26_merged_raw.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_json(path: Path) -> dict:
    if not path.is_file():
        print(f"[WARN] File not found: {path}", file=sys.stderr)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarise_cells(cells: list[dict]) -> dict:
    """Compute aggregate stats over a list of cells."""
    n = len(cells)
    div   = sum(1 for c in cells if c.get("diverged", False))
    fail  = sum(1 for c in cells if c.get("exactkv_failure", False))
    acc   = sum(c.get("acceptance_rate", 1.0) for c in cells) / n if n else 0.0
    ci_lo, ci_hi = wilson_ci(div, n)
    return {
        "cells": n,
        "divergent": div,
        "divergence_rate": round(div / n, 4) if n else 0.0,
        "divergence_ci95_low": round(ci_lo, 4),
        "divergence_ci95_high": round(ci_hi, 4),
        "exactkv_failures": fail,
        "mean_acceptance_rate": round(acc, 4),
    }


def breakdown(cells: list[dict]) -> dict:
    """Return per-model × per-compressor × per-subset breakdown."""
    by_model: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for c in cells:
        model = c.get("model_id", "unknown").split("/")[-1]
        comp  = c.get("compressor_name", "unknown")
        subset = c.get("category", "unknown")
        by_model.setdefault(model, {}).setdefault(comp, {}).setdefault(subset, []).append(c)

    result: dict = {}
    for model, by_comp in by_model.items():
        result[model] = {}
        for comp, by_sub in by_comp.items():
            all_comp_cells = [c for sub_cells in by_sub.values() for c in sub_cells]
            result[model][comp] = {
                "_aggregate": summarise_cells(all_comp_cells),
            }
            for subset, sub_cells in sorted(by_sub.items()):
                result[model][comp][subset] = summarise_cells(sub_cells)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge HF LongBench v2.6 Llama+Mistral panels")
    parser.add_argument("--llama",   required=True, type=Path)
    parser.add_argument("--mistral", required=True, type=Path)
    parser.add_argument("--output",  required=True, type=Path)
    args = parser.parse_args()

    llama_data   = load_json(args.llama)
    mistral_data = load_json(args.mistral)

    llama_cells   = llama_data.get("cells",   [])
    mistral_cells = mistral_data.get("cells", [])
    all_cells = llama_cells + mistral_cells

    merged = {
        "panel_id":    "hf_longbench_v26",
        "description": "ExactKV v2.6 — Real HF LongBench drift panel (Llama-3.1-8B + Mistral-7B, noop/int8/int4_sim, 2K/4K/8K, mnt=32/64)",
        "source_llama":   str(args.llama),
        "source_mistral": str(args.mistral),
        "total_cells":        len(all_cells),
        "llama_cells":        len(llama_cells),
        "mistral_cells":      len(mistral_cells),
        **summarise_cells(all_cells),
        "breakdown": breakdown(all_cells),
        "cells": all_cells,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary table
    print(f"\nMerged {len(all_cells)} cells into {args.output}")
    print(f"  exactkv_failures = {merged['exactkv_failures']}")
    print(f"  divergent        = {merged['divergent']} / {merged['total_cells']}")
    print(f"  divergence_rate  = {merged['divergence_rate']:.1%}  "
          f"CI95=[{merged['divergence_ci95_low']:.1%}, {merged['divergence_ci95_high']:.1%}]")
    print()

    print(f"{'Model':<28} {'Compressor':<14} {'Cells':>6} {'Div':>5} {'Rate':>8}  CI95")
    print("-" * 75)
    for model, by_comp in sorted(merged["breakdown"].items()):
        for comp, data in sorted(by_comp.items()):
            agg = data["_aggregate"]
            lo  = agg["divergence_ci95_low"]
            hi  = agg["divergence_ci95_high"]
            print(
                f"{model:<28} {comp:<14} {agg['cells']:>6}  "
                f"{agg['divergent']:>4}  {agg['divergence_rate']:>6.1%}  "
                f"[{lo:.1%}, {hi:.1%}]"
            )

    print()
    print("Subset breakdown (int4_sim only):")
    print(f"{'Subset':<30} {'Llama div/n':>14} {'Mistral div/n':>14}")
    print("-" * 62)
    # Gather all subsets
    all_subsets: set[str] = set()
    for model_data in merged["breakdown"].values():
        for comp_data in model_data.values():
            all_subsets.update(k for k in comp_data if k != "_aggregate")

    llama_key   = next((k for k in merged["breakdown"] if "llama" in k.lower() or "Llama" in k), None)
    mistral_key = next((k for k in merged["breakdown"] if "Mistral" in k or "mistral" in k), None)

    for subset in sorted(all_subsets):
        def _get(mkey: str | None, comp: str, sub: str) -> str:
            if not mkey:
                return "—"
            d = merged["breakdown"].get(mkey, {}).get(comp, {}).get(sub)
            if not d:
                return "—"
            return f"{d['divergent']}/{d['cells']}"

        l_val = _get(llama_key,   "int4_sim", subset)
        m_val = _get(mistral_key, "int4_sim", subset)
        print(f"  {subset:<28}  {l_val:>13}  {m_val:>13}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
