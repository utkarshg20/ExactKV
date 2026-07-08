#!/usr/bin/env python3
"""Build public review artifacts: Wilson CIs + systems overhead summaries."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> dict[str, float | None]:
    if n <= 0:
        return {"lower": None, "upper": None, "mid": None}
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return {"lower": round(lo, 4), "upper": round(hi, 4), "mid": round(p, 4)}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _headline_cis(scale_path: Path) -> dict:
    data = _load_json(scale_path)
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "div": 0})
    for cell in data.get("cells", []):
        if cell.get("status") != "ok":
            continue
        comp = cell.get("compressor_name") or cell.get("compressor")
        if not comp:
            continue
        buckets[comp]["n"] += 1
        if cell.get("token_level_divergence"):
            buckets[comp]["div"] += 1
    out = {}
    for comp, stats in sorted(buckets.items()):
        n, div = stats["n"], stats["div"]
        out[comp] = {
            "cells": n,
            "divergent_cells": div,
            "divergence_rate": round(div / n, 4) if n else None,
            "divergence_rate_ci95": wilson_ci(div, n),
        }
    return out


def _evidence_plus_timing(evidence_path: Path) -> dict:
    data = _load_json(evidence_path)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ms": [], "accept": []}
    )
    for cell in data.get("cells", []):
        if cell.get("status") != "ok":
            continue
        comp = cell.get("compressor_name")
        if not comp:
            continue
        timing = cell.get("timing_ms") or {}
        total_ms = timing.get("total_cell")
        if total_ms is not None:
            buckets[comp]["ms"].append(float(total_ms))
        acc = cell.get("acceptance_rate")
        if acc is not None:
            buckets[comp]["accept"].append(float(acc))
    by_comp = {}
    for comp, vals in sorted(buckets.items()):
        ms = vals["ms"]
        acc = vals["accept"]
        by_comp[comp] = {
            "cells": len(ms),
            "mean_total_cell_ms": round(sum(ms) / len(ms), 3) if ms else None,
            "median_total_cell_ms": round(sorted(ms)[len(ms) // 2], 3) if ms else None,
            "mean_acceptance_rate": round(sum(acc) / len(acc), 4) if acc else None,
        }
    return by_comp


def build(
    *,
    scale_path: Path,
    analysis_pack_path: Path,
    evidence_path: Path,
    out_ci: Path,
    out_verifier: Path,
    out_recompress: Path,
) -> None:
    headline = _headline_cis(scale_path)
    analysis = _load_json(analysis_pack_path) if analysis_pack_path.is_file() else {}
    totals = analysis.get("totals", {})

    ci_doc = {
        "schema": "exactkv.public_release.confidence_intervals.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Wilson score interval, 95% two-sided (z=1.96)",
        "claim_boundary": "Intervals describe observed drift/acceptance on cited panels only; not official benchmark scores.",
        "headline_panel": {
            "source": str(scale_path.relative_to(ROOT)),
            "cells_total": sum(v["cells"] for v in headline.values()),
            "by_compressor": headline,
        },
        "external_smoke_summary": {
            "source": str(analysis_pack_path.relative_to(ROOT)),
            "cells_ok": totals.get("external_gpu_cells_ok"),
            "divergence_rate_overall_ci95": totals.get("divergence_rate_overall_ci95"),
            "acceptance_full_rate_ci95": totals.get("acceptance_full_rate_ci95"),
        },
    }
    out_ci.parent.mkdir(parents=True, exist_ok=True)
    out_ci.write_text(json.dumps(ci_doc, indent=2) + "\n", encoding="utf-8")

    verifier_doc = {
        "schema": "exactkv.systems.verifier_overhead.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Cell-level wall-clock proxy from the evidence-plus panel (144 cells, 7B/8B). "
            "NOT a per-token verifier cost model on the 1,500-cell headline panel and NOT end-to-end serving latency."
        ),
        "source_artifact": str(evidence_path.relative_to(ROOT)),
        "panel": "evidence_plus",
        "by_compressor": _evidence_plus_timing(evidence_path),
        "not_measured": [
            "Per-token verifier latency breakdown on headline scale panel",
            "Verifier cost vs draft acceptance curve on external LongBench/BFCL panels",
            "End-to-end speedup vs full-KV-only greedy decode",
        ],
    }
    out_verifier.parent.mkdir(parents=True, exist_ok=True)
    out_verifier.write_text(json.dumps(verifier_doc, indent=2) + "\n", encoding="utf-8")

    recompress_doc = {
        "schema": "exactkv.systems.recompression_overhead.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_measured",
        "claim_boundary": (
            "Recompression cost during draft/verify/commit is not isolated in headline GPU panels. "
            "Phase F reports kernel compress/decompress microbenchmarks only."
        ),
        "related_artifacts": [
            "reports/systems/latency_microbench.json",
            "reports/phaseF_kernel_benchmark.json",
        ],
    }
    out_recompress.write_text(json.dumps(recompress_doc, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", type=Path, default=ROOT / "reports/scale_7b/raw.json")
    p.add_argument(
        "--analysis-pack",
        type=Path,
        default=ROOT / "reports/external_panels/analysis_pack.json",
    )
    p.add_argument("--evidence", type=Path, default=ROOT / "reports/evidence_plus/raw.json")
    p.add_argument(
        "--out-ci",
        type=Path,
        default=ROOT / "reports/public_release/confidence_intervals.json",
    )
    p.add_argument(
        "--out-verifier",
        type=Path,
        default=ROOT / "reports/systems/verifier_overhead.json",
    )
    p.add_argument(
        "--out-recompress",
        type=Path,
        default=ROOT / "reports/systems/recompression_overhead.json",
    )
    args = p.parse_args()
    build(
        scale_path=args.scale,
        analysis_pack_path=args.analysis_pack,
        evidence_path=args.evidence,
        out_ci=args.out_ci,
        out_verifier=args.out_verifier,
        out_recompress=args.out_recompress,
    )
    print(f"Wrote {args.out_ci}")
    print(f"Wrote {args.out_verifier}")
    print(f"Wrote {args.out_recompress}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
