#!/usr/bin/env python3
"""Aggregate serving_microbench raw JSONs into the public systems pack."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.serving_microbench_panel import (  # noqa: E402
    CLAIM_BOUNDARY,
    FORBIDDEN_FIELDS,
    assert_no_forbidden_fields,
)


def _giB(n: int | None) -> float | None:
    if n is None:
        return None
    return round(n / (1024**3), 3)


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 3) if xs else None


def _short_model(name: str) -> str:
    lower = name.lower()
    if "llama" in lower:
        return "llama"
    if "mistral" in lower:
        return "mistral"
    return name.split("/")[-1]


def build_pack(cells: list[dict[str, Any]], *, sources: list[str]) -> dict[str, Any]:
    by: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: {
            "peak_gib": [],
            "ttft_ms": [],
            "e2e_ms": [],
            "rps": [],
            "delta_gib": [],
        }
    )
    for cell in cells:
        model = _short_model(str(cell.get("model_name") or ""))
        comp = str(cell.get("compressor_name") or "")
        nreq = int(cell.get("n_requests") or 0)
        for arm in ("full", "lossy", "exactkv"):
            arm_obj = cell.get(arm) or {}
            key = (model, comp, arm, nreq)
            peak = arm_obj.get("gpu_peak_allocated_bytes")
            if peak is not None:
                by[key]["peak_gib"].append(float(_giB(int(peak)) or 0.0))
            if arm_obj.get("mean_ttft_like_ms") is not None:
                by[key]["ttft_ms"].append(float(arm_obj["mean_ttft_like_ms"]))
            if arm_obj.get("mean_e2e_ms") is not None:
                by[key]["e2e_ms"].append(float(arm_obj["mean_e2e_ms"]))
            if arm_obj.get("completed_requests_per_sec") is not None:
                by[key]["rps"].append(float(arm_obj["completed_requests_per_sec"]))
            delta = arm_obj.get("peak_delta_vs_full_bytes")
            if delta is not None:
                by[key]["delta_gib"].append(float(_giB(int(delta)) or 0.0))

    tables: dict[str, Any] = {
        "peak_cuda_allocation_gib": {},
        "ttft_like_ms": {},
        "completed_requests_per_sec": {},
        "peak_delta_vs_full_gib": {},
    }
    for (model, comp, arm, nreq), vals in sorted(by.items()):
        slot = f"serial_{nreq}"
        for table_key, metric in (
            ("peak_cuda_allocation_gib", "peak_gib"),
            ("ttft_like_ms", "ttft_ms"),
            ("completed_requests_per_sec", "rps"),
            ("peak_delta_vs_full_gib", "delta_gib"),
        ):
            tables[table_key].setdefault(model, {}).setdefault(comp, {}).setdefault(
                slot, {}
            )[arm] = {
                "mean": _mean(vals[metric]),
                "n": len(vals[metric]),
            }

    pack = {
        "schema": "exactkv.systems.serving_microbench.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
        "source_artifacts": sources,
        "n_cells": len(cells),
        "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
        **tables,
        "notes": [
            "Serial request load on the ExactKV HF harness (one generate after another).",
            "TTFT-like: prefill→first-token for full/lossy; first verify-commit round for ExactKV.",
            "peak_delta_vs_full_gib > 0 means the arm peaked higher than full (common for ExactKV).",
            "NOT vLLM integration / continuous batching / production serving.",
        ],
        "not_measured": [
            "vLLM continuous batching RPS",
            "OpenAI-compatible HTTP server load",
            "Isolated KV-only VRAM without weights",
            "Packed INT4 production memory savings",
        ],
    }
    bad = FORBIDDEN_FIELDS & pack.keys()
    if bad:
        raise ValueError(f"Forbidden top-level fields: {bad}")
    assert_no_forbidden_fields(pack)
    return pack


def render_md(pack: dict[str, Any]) -> str:
    lines = [
        "# Serving microbench pack",
        "",
        f"**Cells:** {pack.get('n_cells')} · **exactkv_failures:** {pack.get('exactkv_failures')}",
        "",
        f"**Claim boundary:** {pack.get('claim_boundary')}",
        "",
        "## Completed requests/sec (mean, serial load)",
        "",
        "| Model | Comp | Load | full | lossy | ExactKV |",
        "|-------|------|------|-----:|------:|--------:|",
    ]
    rps = pack.get("completed_requests_per_sec") or {}
    for model, comps in sorted(rps.items()):
        for comp, slots in sorted(comps.items()):
            for slot, arms in sorted(slots.items()):
                lines.append(
                    f"| {model} | `{comp}` | {slot} | "
                    f"{(arms.get('full') or {}).get('mean')} | "
                    f"{(arms.get('lossy') or {}).get('mean')} | "
                    f"{(arms.get('exactkv') or {}).get('mean')} |"
                )
    lines += [
        "",
        "## Peak CUDA (GiB, mean) and Δ vs full",
        "",
        "| Model | Comp | Load | full | lossy | ExactKV | Δlossy | ΔExactKV |",
        "|-------|------|------|-----:|------:|--------:|-------:|---------:|",
    ]
    peak = pack.get("peak_cuda_allocation_gib") or {}
    delta = pack.get("peak_delta_vs_full_gib") or {}
    for model, comps in sorted(peak.items()):
        for comp, slots in sorted(comps.items()):
            for slot, arms in sorted(slots.items()):
                d = (delta.get(model) or {}).get(comp) or {}
                dslot = d.get(slot) or {}
                lines.append(
                    f"| {model} | `{comp}` | {slot} | "
                    f"{(arms.get('full') or {}).get('mean')} | "
                    f"{(arms.get('lossy') or {}).get('mean')} | "
                    f"{(arms.get('exactkv') or {}).get('mean')} | "
                    f"{(dslot.get('lossy') or {}).get('mean')} | "
                    f"{(dslot.get('exactkv') or {}).get('mean')} |"
                )
    lines += [
        "",
        "## TTFT-like (ms, mean)",
        "",
        "| Model | Comp | Load | full | lossy | ExactKV |",
        "|-------|------|------|-----:|------:|--------:|",
    ]
    ttft = pack.get("ttft_like_ms") or {}
    for model, comps in sorted(ttft.items()):
        for comp, slots in sorted(comps.items()):
            for slot, arms in sorted(slots.items()):
                lines.append(
                    f"| {model} | `{comp}` | {slot} | "
                    f"{(arms.get('full') or {}).get('mean')} | "
                    f"{(arms.get('lossy') or {}).get('mean')} | "
                    f"{(arms.get('exactkv') or {}).get('mean')} |"
                )
    lines += ["", "## Notes", ""]
    for note in pack.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/external_panels/serving_microbench"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/systems/serving_microbench.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/systems/serving_microbench.md"),
    )
    args = parser.parse_args()

    cells: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in sorted(args.input_dir.glob("*_raw.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cells.extend(data.get("cells") or [])
        sources.append(str(path))
    if not cells:
        raise SystemExit(f"No *_raw.json cells found under {args.input_dir}")

    pack = build_pack(cells, sources=sources)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_md(pack), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md} from {len(cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
