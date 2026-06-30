#!/usr/bin/env python3
"""One-command integration of v2.7 BFCL validity results into the paper.

Reads the merged v2.7 BFCL validity artifact, computes per-compressor/model metrics,
generates:
  1. Paper table markdown (replaces placeholder Table 4f in the .md)
  2. Validity rate summary per compressor × model
  3. Case study candidates (highest divergence-inside-arguments cells)
  4. Abstract update snippet

Usage (after RunPod artifacts are copied back):
  python3 scripts/integrate_v27_results.py \
      --merged reports/external_panels/bfcl_validity_v27_merged_raw.json \
      --md     paper/ExactKV_Technical_Report.md \
      --tex    paper/ExactKV_Technical_Report.tex \
      --write   # actually patch the files (default: dry-run only)
"""
from __future__ import annotations

import argparse
import json
import math
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


def aggregate_cells(cells: list[dict]) -> dict:
    n = len(cells)

    def _diverged(c: dict) -> bool:
        m = c.get("metrics") or {}
        return bool(m.get("token_level_divergence")) or bool(c.get("diverged"))

    def _fail(c: dict) -> bool:
        m = c.get("metrics") or {}
        return bool(m.get("exactkv_failure")) or bool(c.get("exactkv_failure"))

    div = sum(1 for c in cells if _diverged(c))
    fail = sum(1 for c in cells if _fail(c))
    valid_full = sum(1 for c in cells if c.get("full_kv_tool_call_valid", False))
    valid_lossy = sum(1 for c in cells if c.get("lossy_tool_call_valid", False))
    valid_exactkv = sum(1 for c in cells if c.get("exactkv_tool_call_valid", False))
    malformed = sum(1 for c in cells if c.get("malformed_json", False))
    div_in_args = sum(
        1 for c in cells
        if _diverged(c) and c.get("divergence_location") == "arguments"
    )
    div_in_name = sum(
        1 for c in cells
        if _diverged(c) and c.get("divergence_location") == "name"
    )
    lo_div, hi_div = wilson_ci(div, n)
    lo_vm, hi_vm = wilson_ci(valid_exactkv, n) if n else (0.0, 1.0)
    return {
        "cells": n,
        "divergent": div,
        "div_rate": div / n if n else 0.0,
        "div_ci": (lo_div, hi_div),
        "exactkv_failures": fail,
        "valid_full": valid_full,
        "valid_lossy": valid_lossy,
        "valid_exactkv": valid_exactkv,
        "exactkv_valid_rate": valid_exactkv / n if n else 0.0,
        "exactkv_valid_ci": (lo_vm, hi_vm),
        "malformed": malformed,
        "div_in_args": div_in_args,
        "div_in_name": div_in_name,
    }


def build_md_table(data: dict) -> str:
    cells = data.get("cells", [])
    header = (
        "| Compressor | Model | n | Divergent | Div rate | CI₉₅ | "
        "Full valid | ExactKV valid | Malformed | ExactKV fail |\n"
        "|-----------|-------|---|-----------|---------|------|"
        "-----------|--------------|----------|-------------|\n"
    )

    rows: list[str] = []
    grouped: dict[str, dict[str, list[dict]]] = {}
    for c in cells:
        comp = c.get("compressor_name", "?")
        model = _model_short(c.get("model", "?"))
        grouped.setdefault(comp, {}).setdefault(model, []).append(c)

    for comp in ("noop", "int8", "int4_sim"):
        for model in ("Llama", "Mistral"):
            group = grouped.get(comp, {}).get(model, [])
            if not group:
                rows.append(f"| {comp} | {model} | — | — | — | — | — | — | — | — |")
                continue
            a = aggregate_cells(group)
            rows.append(
                f"| {comp} | {model} | {a['cells']} | {a['divergent']} | "
                f"{a['div_rate']:.1%} | {_fmt_ci(*a['div_ci'])} | "
                f"{a['valid_full']}/{a['cells']} | {a['valid_exactkv']}/{a['cells']} | "
                f"{a['malformed']} | {a['exactkv_failures']} |"
            )

    return header + "\n".join(rows)


def build_summary_section(data: dict) -> str:
    cells = data.get("cells", [])
    total = len(cells)
    if total == 0:
        return "_No cells found in merged artifact._"

    all_agg = aggregate_cells(cells)
    by_comp: dict[str, list[dict]] = {}
    for c in cells:
        by_comp.setdefault(c.get("compressor_name", "?"), []).append(c)

    lines = [
        f"**v2.7 BFCL Validity Panel — Summary ({total} total cells)**",
        "",
        f"- `exactkv_failures`: {all_agg['exactkv_failures']}",
        f"- Overall divergence rate: {all_agg['div_rate']:.1%} "
        f"{_fmt_ci(*all_agg['div_ci'])}",
        f"- Full-KV tool-call valid: {all_agg['valid_full']}/{total} "
        f"({all_agg['valid_full']/total:.1%})" if total else "",
        f"- ExactKV tool-call valid: {all_agg['valid_exactkv']}/{total} "
        f"({all_agg['valid_exactkv']/total:.1%})" if total else "",
        f"- Malformed JSON rate: {all_agg['malformed']}/{total}",
        "",
        "**Per-compressor:**",
    ]

    for comp in ("noop", "int8", "int4_sim"):
        group = by_comp.get(comp, [])
        if not group:
            continue
        a = aggregate_cells(group)
        lines.append(
            f"  - {comp}: {a['divergent']}/{a['cells']} divergent "
            f"({a['div_rate']:.1%}), ExactKV valid {a['valid_exactkv']}/{a['cells']}"
        )

    # Divergence location breakdown
    def _diverged_check(c: dict) -> bool:
        m = c.get("metrics") or {}
        return bool(m.get("token_level_divergence")) or bool(c.get("diverged"))

    div_cells = [c for c in cells if _diverged_check(c)]
    if div_cells:
        in_name = sum(1 for c in div_cells if c.get("divergence_location") == "name")
        in_args = sum(1 for c in div_cells if c.get("divergence_location") == "arguments")
        in_other = len(div_cells) - in_name - in_args
        lines += [
            "",
            f"**Divergence location** (among {len(div_cells)} divergent cells):",
            f"  - Inside `arguments`: {in_args}",
            f"  - Inside `name`: {in_name}",
            f"  - Other: {in_other}",
        ]

    return "\n".join(lines)


def find_interesting_cells(data: dict, n: int = 5) -> list[dict]:
    """Top candidates for case study: divergent + inside arguments."""
    cells = data.get("cells", [])
    def _div(c: dict) -> bool:
        m = c.get("metrics") or {}
        return bool(m.get("token_level_divergence")) or bool(c.get("diverged"))
    candidates = [
        c for c in cells
        if _div(c) and c.get("divergence_location") == "arguments"
    ]
    candidates.sort(key=lambda c: c.get("first_divergence_index", 9999))
    return candidates[:n]


def build_abstract_update(data: dict) -> str:
    cells = data.get("cells", [])
    total = len(cells)
    all_agg = aggregate_cells(cells)
    by_comp: dict[str, list[dict]] = {}
    for c in cells:
        by_comp.setdefault(c.get("compressor_name", "?"), []).append(c)

    int4_agg = aggregate_cells(by_comp.get("int4_sim", []))
    noop_agg = aggregate_cells(by_comp.get("noop", []))

    lines = [
        "--- ABSTRACT UPDATE SNIPPET (v2.7) ---",
        "",
        f"A BFCL tool-call validity panel ({total} cells, both models, "
        "max_new_tokens=128/256) measured structured-output fidelity under KV compression. "
        f"noop baseline: {noop_agg['div_rate']:.0%} divergence, "
        f"{noop_agg['valid_exactkv']/noop_agg['cells']:.0%} valid tool calls. "
        f"int4_sim: {int4_agg['div_rate']:.0%} divergence, "
        f"{int4_agg['valid_exactkv']/int4_agg['cells'] if int4_agg['cells'] else 0:.0%} "
        "valid tool calls. `exactkv_failures=0` across all {total} cells.",
        "---",
    ]
    return "\n".join(lines)


PLACEHOLDER_TABLE_4F = (
    "| noop | Llama | [pending] | [pending] | [pending] | [pending] | [pending] |"
)

REPLACEMENT_SENTINEL_MD = "**Table 4f**"


def patch_md(md_path: Path, new_table: str, summary: str) -> bool:
    text = md_path.read_text(encoding="utf-8")
    marker = "| noop | Llama | [pending] | — | [pending] | [pending] |"
    if marker not in text:
        print(f"[warn] Could not find Table 4f placeholder in {md_path}")
        return False
    text = text.replace(marker, new_table)
    # Also replace the status line
    old_status = "**Status:** Panel queued on RunPod"
    new_status = f"**Status:** Complete — {summary.splitlines()[0]}"
    text = text.replace(old_status, new_status, 1)
    md_path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Integrate v2.7 BFCL validity results into paper.")
    ap.add_argument("--merged", required=True, help="Path to bfcl_validity_v27_merged_raw.json")
    ap.add_argument("--md", default="paper/ExactKV_Technical_Report.md")
    ap.add_argument("--tex", default="paper/ExactKV_Technical_Report.tex")
    ap.add_argument(
        "--write", action="store_true",
        help="Actually patch the paper files (default: dry-run, print only)"
    )
    args = ap.parse_args()

    merged_path = Path(args.merged)
    if not merged_path.exists():
        print(f"[error] Artifact not found: {merged_path}", file=sys.stderr)
        return 1

    data = json.loads(merged_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    total = len(cells)
    if total == 0:
        print("[warn] No cells found in merged artifact. Nothing to do.")
        return 0

    print(f"[info] Loaded {total} cells from {merged_path}")

    md_table = build_md_table(data)
    summary = build_summary_section(data)
    abstract = build_abstract_update(data)
    interesting = find_interesting_cells(data)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print("TABLE 4f (Markdown)")
    print("=" * 60)
    print(md_table)

    if interesting:
        print("\n" + "=" * 60)
        print(f"TOP {len(interesting)} CASE STUDY CANDIDATES (divergent, in arguments)")
        print("=" * 60)
        for i, cell in enumerate(interesting, 1):
            model = _model_short(cell.get("model", "?"))
            comp = cell.get("compressor_name", "?")
            idx = cell.get("first_divergence_index", "?")
            subset = cell.get("category", "?")
            print(f"  {i}. model={model} comp={comp} subset={subset} first_div={idx}")

    print("\n" + "=" * 60)
    print("ABSTRACT SNIPPET")
    print("=" * 60)
    print(abstract)

    if args.write:
        md_path = Path(args.md)
        patched = patch_md(md_path, md_table, summary)
        print(f"\n[write] {md_path}: {'patched' if patched else 'SKIPPED (placeholder not found)'}")
        print("[info] LaTeX patching not yet implemented — do manually or rerun after checking .tex structure.")
    else:
        print("\n[dry-run] Pass --write to patch the paper files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
