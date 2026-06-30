#!/usr/bin/env python3
"""Acceptance vs drift analysis from panel JSON artifacts (reviewer systems ask)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_GLOBS = [
    "reports/scale_7b/raw.json",
    "reports/external_panels/v30/*_raw.json",
    "reports/external_panels/bfcl_validity_v27_merged_raw.json",
    "reports/external_panels/faithful/*_raw.json",
]


def _load_cells(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in data.get("cells") or [] if c.get("status") == "ok"]


def _summarise_file(path: Path) -> list[dict]:
    rows: list[dict] = []
    for c in _load_cells(path):
        m = c.get("metrics") or {}
        t = c.get("timing_ms") or {}
        rows.append({
            "file": path.name,
            "family": c.get("dataset_family") or path.stem,
            "compressor": c.get("compressor_name"),
            "model": c.get("model_name"),
            "context_bucket": c.get("context_bucket"),
            "max_new_tokens": c.get("max_new_tokens"),
            "diverged": bool(m.get("token_level_divergence")),
            "acceptance_rate": float(m.get("acceptance_rate", 1.0)),
            "first_divergence_index": m.get("first_divergence_index"),
            "total_cell_ms": t.get("total_cell"),
        })
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["family"], r["compressor"], r["model"])
        groups[key].append(r)

    out: list[dict] = []
    for (family, comp, model), items in sorted(groups.items()):
        n = len(items)
        div = sum(1 for x in items if x["diverged"])
        accs = [x["acceptance_rate"] for x in items]
        times = [x["total_cell_ms"] for x in items if x["total_cell_ms"] is not None]
        fdis = [x["first_divergence_index"] for x in items if x["first_divergence_index"] is not None]
        out.append({
            "dataset_family": family,
            "compressor": comp,
            "model": model,
            "cells": n,
            "divergence_rate": div / n if n else 0.0,
            "mean_acceptance": statistics.mean(accs) if accs else 1.0,
            "median_acceptance": statistics.median(accs) if accs else 1.0,
            "mean_first_divergence_index": statistics.mean(fdis) if fdis else None,
            "mean_cell_ms": statistics.mean(times) if times else None,
            "verify_overhead_note": (
                "Higher acceptance + lower divergence => verifier does less correction work. "
                "total_cell_ms is diagnostic cell time, not serving throughput."
            ),
        })
    return out


def _md(summary: list[dict], note: str) -> str:
    lines = [
        "# Acceptance vs Drift Analysis",
        "",
        note,
        "",
        "| Family | Compressor | Model | Cells | Div% | Mean accept | Mean FDI | Mean ms |",
        "|--------|------------|-------|------:|-----:|------------:|---------:|--------:|",
    ]
    for s in summary:
        fdi = s["mean_first_divergence_index"]
        ms = s["mean_cell_ms"]
        lines.append(
            f"| {s['dataset_family']} | `{s['compressor']}` | {s['model'].split('/')[-1]} | "
            f"{s['cells']} | {100 * s['divergence_rate']:.1f} | {s['mean_acceptance']:.3f} | "
            f"{fdi if fdi is not None else '—'} | "
            f"{ms if ms is not None else '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", nargs="*", default=DEFAULT_GLOBS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out-json", default="reports/external_panels/acceptance_drift_analysis.json")
    parser.add_argument("--out-md", default="reports/external_panels/acceptance_drift_analysis.md")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.glob:
        paths.extend(sorted(Path(".").glob(pattern)))
    paths = [p for p in paths if p.is_file()]

    if not paths:
        print("No panel JSON files found.")
        return 1

    rows: list[dict] = []
    for p in paths:
        rows.extend(_summarise_file(p))

    summary = _aggregate(rows)
    note = (
        "Diagnostic only — not end-to-end serving cost. "
        "Useful for: when does verify/commit pay off (high acceptance, low drift)?"
    )
    pack = {"sources": [str(p) for p in paths], "summary": summary, "note": note}
    md = _md(summary, note)
    print(md)

    if args.write:
        out_json = Path(args.out_json)
        out_md = Path(args.out_md)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
        out_md.write_text(md + "\n", encoding="utf-8")
        print(f"Wrote {out_json}")
        print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
