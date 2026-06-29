#!/usr/bin/env python3
"""Summarize v3.0 panel results (int6_sim + int4_per_vec_sim validation).

Reads reports/external_panels/v30/*_raw.json and prints:
  - per-compressor per-family divergence rate table (for paper)
  - exactkv_failure check
  - comparison vs int8/int4_sim baselines

Usage:
    python3 scripts/summarize_v30_panel.py
    python3 scripts/summarize_v30_panel.py --output reports/external_panels/v30/summary.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
V30_DIR = _ROOT / "reports" / "external_panels" / "v30"

COMPRESSOR_ORDER = ["int8", "int6_sim", "int4_per_vec_sim", "int4_sim"]
FAMILY_ORDER = ["mbpp", "bfcl", "longbench"]

# Expected baseline ranges from v2.9 panel (for sanity check)
EXPECTED_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("int8", "mbpp"): (0.0, 0.10),
    ("int8", "bfcl"): (0.0, 0.10),
    ("int8", "longbench"): (0.0, 0.30),
    ("int4_sim", "mbpp"): (0.03, 0.15),
    ("int4_sim", "bfcl"): (0.10, 0.70),
    ("int4_sim", "longbench"): (0.70, 1.00),
}


def _extract_cells(raw: dict) -> list[dict]:
    """Support both flat and nested cell formats."""
    cells = raw.get("cells") or raw.get("results") or []
    if not cells and "panels" in raw:
        for panel in raw["panels"]:
            cells.extend(panel.get("cells", []))
    return cells


def _divergence_rate(cells: list[dict]) -> float | None:
    completed = [c for c in cells if c.get("status") in ("ok", "diverged", "accepted")]
    if not completed:
        return None
    diverged = [c for c in completed if c.get("diverged", False) or c.get("status") == "diverged"]
    return len(diverged) / len(completed)


def _exactkv_failures(cells: list[dict]) -> int:
    return sum(1 for c in cells if c.get("exactkv_failure", False))


def load_v30_results() -> dict[tuple[str, str, str], list[dict]]:
    """Returns dict[(family, model, compressor)] -> cells."""
    results: dict[tuple[str, str, str], list[dict]] = {}
    for f in sorted(V30_DIR.glob("*_raw.json")):
        try:
            raw = json.loads(f.read_text())
        except Exception as e:
            print(f"  [WARN] Could not parse {f.name}: {e}")
            continue
        cells = _extract_cells(raw)
        family = raw.get("family", "unknown")
        for cell in cells:
            model = cell.get("model", "unknown").split("/")[-1]
            comp = cell.get("compressor", "unknown")
            key = (family, model, comp)
            results.setdefault(key, []).append(cell)
    return results


def build_summary_table(
    results: dict[tuple[str, str, str], list[dict]],
) -> tuple[list[dict], int]:
    """Returns (rows, total_exactkv_failures)."""
    rows = []
    total_failures = 0

    # Aggregate across models
    agg: dict[tuple[str, str], list[dict]] = {}
    for (family, model, comp), cells in results.items():
        agg.setdefault((family, comp), []).extend(cells)

    for family in FAMILY_ORDER:
        for comp in COMPRESSOR_ORDER:
            cells = agg.get((family, comp), [])
            dr = _divergence_rate(cells)
            failures = _exactkv_failures(cells)
            total_failures += failures
            rows.append({
                "family": family,
                "compressor": comp,
                "cells": len(cells),
                "divergence_rate": dr,
                "exactkv_failures": failures,
            })

    return rows, total_failures


def format_table(rows: list[dict]) -> str:
    lines = []
    lines.append(
        "| Family | Compressor | Cells | Divergence Rate | exactkv_failures |"
    )
    lines.append("|--------|------------|------:|----------------:|-----------------:|")
    prev_family = None
    for row in rows:
        if row["family"] != prev_family and prev_family is not None:
            lines.append("|        |            |       |                 |                  |")
        prev_family = row["family"]
        dr = row["divergence_rate"]
        dr_str = f"{dr:.1%}" if dr is not None else "—"
        lines.append(
            f"| {row['family']} | {row['compressor']} | {row['cells']} "
            f"| {dr_str} | {row['exactkv_failures']} |"
        )
    return "\n".join(lines)


def sanity_check(rows: list[dict]) -> list[str]:
    warnings = []
    for row in rows:
        key = (row["compressor"], row["family"])
        dr = row["divergence_rate"]
        if dr is None or row["cells"] == 0:
            warnings.append(f"  [MISSING] {row['family']} / {row['compressor']} — no cells found")
            continue
        lo, hi = EXPECTED_RANGES.get(key, (None, None))
        if lo is not None and not (lo <= dr <= hi):
            warnings.append(
                f"  [RANGE] {row['family']}/{row['compressor']}: "
                f"{dr:.1%} outside expected [{lo:.0%}–{hi:.0%}]"
            )
    return warnings


def build_paper_claims(rows: list[dict]) -> str:
    """Build a short narrative for the paper about int6_sim and int4_per_vec_sim."""
    dr_map = {(r["family"], r["compressor"]): r["divergence_rate"] for r in rows}

    lines = ["### v3.0 int6_sim and int4_per_vec_sim GPU Panel Results\n"]
    lines.append(
        "The v3.0 panel runs `int6_sim` and `int4_per_vec_sim` alongside "
        "`int8` and `int4_sim` controls on three benchmark families:\n"
    )

    for comp in ["int6_sim", "int4_per_vec_sim"]:
        dr_mbpp = dr_map.get(("mbpp", comp))
        dr_bfcl = dr_map.get(("bfcl", comp))
        dr_lb = dr_map.get(("longbench", comp))
        int4_lb = dr_map.get(("longbench", "int4_sim"))
        int8_lb = dr_map.get(("longbench", "int8"))
        if dr_mbpp is None:
            continue
        lines.append(f"**{comp}**: MBPP {dr_mbpp:.0%} → BFCL {dr_bfcl:.0%} → LongBench {dr_lb:.0%}.")
        if int4_lb is not None and int8_lb is not None and dr_lb is not None:
            if int8_lb < dr_lb < int4_lb:
                lines.append(
                    f"  Lands between `int8` ({int8_lb:.0%}) and `int4_sim` ({int4_lb:.0%}) "
                    "on LongBench — consistent with the predicted non-catastrophic profile."
                )
            elif dr_lb <= int8_lb:
                lines.append(
                    f"  Matches `int8` ({int8_lb:.0%}) on LongBench — better than predicted."
                )
            else:
                lines.append(
                    f"  Exceeds `int4_sim` ({int4_lb:.0%}) on LongBench — "
                    "requires further investigation."
                )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v3.0 panel results")
    parser.add_argument("--output", default="", help="Write summary markdown to this path")
    args = parser.parse_args()

    if not V30_DIR.exists():
        print(f"ERROR: {V30_DIR} not found. Run scripts/run_v30_new_compressors.sh first.")
        return

    results = load_v30_results()
    if not results:
        print(f"No results found in {V30_DIR}/. Ensure the panel has been run.")
        return

    rows, total_failures = build_summary_table(results)
    table = format_table(rows)
    claims = build_paper_claims(rows)
    warnings = sanity_check(rows)

    total_cells = sum(r["cells"] for r in rows)
    print(f"\n=== v3.0 Panel Summary ===")
    print(f"Total cells: {total_cells}")
    print(f"Total exactkv_failures: {total_failures} (must be 0)")
    print()
    print(table)
    print()
    if warnings:
        print("=== Sanity Check Warnings ===")
        for w in warnings:
            print(w)
        print()
    print("=== Paper Narrative ===")
    print(claims)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        content = f"# v3.0 Panel Summary\n\nTotal cells: {total_cells}  "
        content += f"exactkv_failures: {total_failures}\n\n"
        content += table + "\n\n"
        if warnings:
            content += "## Sanity Check Warnings\n\n"
            content += "\n".join(warnings) + "\n\n"
        content += claims
        out.write_text(content)
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
