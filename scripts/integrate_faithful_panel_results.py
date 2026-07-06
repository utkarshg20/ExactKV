#!/usr/bin/env python3
"""Merge faithful external compressor panel JSONs and write summary markdown."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.phase_a_scale_benchmark import _aggregate_compressor_metrics


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _summarize(report: dict, *, label: str) -> str:
    cells = [c for c in report.get("cells", []) if c.get("status") == "ok"]
    summary = _aggregate_compressor_metrics(cells)
    lines = [
        f"## {label}",
        "",
        f"- **Cells:** {len(cells)} ok / {len(report.get('cells', []))} total",
        f"- **ExactKV failures:** {sum(1 for c in cells if (c.get('metrics') or {}).get('exactkv_failure'))}",
        "",
        "| Compressor | n | Div. rate | Mean accept. |",
        "|------------|--:|----------:|-------------:|",
    ]
    for name, stats in sorted(summary.items()):
        lines.append(
            f"| `{name}` | {stats['num_cells']} | "
            f"{100 * stats['divergence_rate']:.1f}% | {stats['mean_acceptance_rate']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("reports/external_panels/faithful"),
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    raw_files = sorted(
        p for p in args.dir.glob("*_raw.json") if p.name != "merged_raw.json"
    )
    if not raw_files:
        print(f"No raw JSON in {args.dir}")
        return 1

    merged_cells: list[dict] = []
    sections: list[str] = ["# Faithful External Compressor Panel Summary", ""]
    for path in raw_files:
        report = _load(path)
        merged_cells.extend(report.get("cells", []))
        sections.append(_summarize(report, label=path.stem))

    ok = [c for c in merged_cells if c.get("status") == "ok"]
    overall = _aggregate_compressor_metrics(ok)
    sections.extend(["## Overall (all families)", ""])
    sections.append("| Compressor | n | Div. rate | Mean accept. |")
    sections.append("|------------|--:|----------:|-------------:|")
    for name, stats in sorted(overall.items()):
        sections.append(
            f"| `{name}` | {stats['num_cells']} | "
            f"{100 * stats['divergence_rate']:.1f}% | {stats['mean_acceptance_rate']:.3f} |"
        )
    sections.append("")

    md = "\n".join(sections)
    print(md)
    if args.write:
        out = args.dir / "summary.md"
        out.write_text(md, encoding="utf-8")
        merged = {
            "panel_id": "faithful_external_compressors",
            "cells": merged_cells,
            "compressor_summary": overall,
        }
        (args.dir / "merged_raw.json").write_text(
            json.dumps(merged, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {out} and merged_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
