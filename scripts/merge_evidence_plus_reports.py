#!/usr/bin/env python3
"""Merge per-model evidence-plus JSON reports into combined raw.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.phase_a_scale_benchmark import _aggregate_compressor_metrics  # noqa: E402


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("no reports to merge")
    base = dict(reports[0])
    cells: list[dict[str, Any]] = []
    models_evaluated: list[str] = []
    models_blocked: dict[str, str] = {}
    for rep in reports:
        cells.extend(rep.get("cells") or [])
        for m in rep.get("models_evaluated") or []:
            if m not in models_evaluated:
                models_evaluated.append(m)
        models_blocked.update(rep.get("models_blocked") or {})

    ok_cells = [c for c in cells if c.get("status") == "ok"]
    by_bucket: dict[int, list[dict[str, Any]]] = {}
    for c in ok_cells:
        b = c.get("context_bucket")
        if b is not None:
            by_bucket.setdefault(int(b), []).append(c)

    bucket_summary: dict[str, Any] = {}
    for b, bucket_cells in sorted(by_bucket.items()):
        metrics = [c.get("metrics") or {} for c in bucket_cells]
        n = max(len(metrics), 1)
        div = sum(1 for m in metrics if m.get("token_level_divergence"))
        bucket_summary[str(b)] = {
            "num_cells": len(bucket_cells),
            "divergence_rate": div / n,
            "mean_acceptance_rate": sum(m.get("acceptance_rate", 0.0) for m in metrics) / n,
        }

    timings = [
        float((c.get("timing_ms") or {}).get("total_cell") or 0.0)
        for c in ok_cells
        if (c.get("timing_ms") or {}).get("total_cell")
    ]
    timing_summary = {}
    if timings:
        timings_sorted = sorted(timings)
        timing_summary = {
            "num_timed_cells": len(timings),
            "mean_ms": sum(timings) / len(timings),
            "p50_ms": timings_sorted[len(timings_sorted) // 2],
            "p90_ms": timings_sorted[int(len(timings_sorted) * 0.9)],
        }

    base.update(
        {
            "status": "benchmark_complete",
            "models_evaluated": models_evaluated,
            "models_blocked": models_blocked,
            "total_cells": len(cells),
            "cells_run": len(ok_cells),
            "cells_skipped": len(cells) - len(ok_cells),
            "exactkv_failures": sum(1 for c in ok_cells if c.get("exactkv_failure")),
            "compressor_summary": _aggregate_compressor_metrics(ok_cells),
            "bucket_summary": bucket_summary,
            "timing_summary": timing_summary,
            "cells": cells,
            "merged_from": [r.get("phase_id") for r in reports],
        },
    )
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("reports/evidence_plus/raw.json"))
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.inputs]
    merged = merge_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"wrote {args.output} cells={merged['total_cells']} ok={merged['cells_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
