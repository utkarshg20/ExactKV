#!/usr/bin/env python3
"""Aggregate systems_diagnostic raw JSONs into the public systems pack."""
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

from exactkv.benchmarks.systems_diagnostic_panel import (  # noqa: E402
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


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 3) if xs else None


def _short_model(name: str) -> str:
    lower = name.lower()
    if "llama" in lower:
        return "llama"
    if "mistral" in lower:
        return "mistral"
    return name.split("/")[-1]


def build_pack(cells: list[dict[str, Any]], *, sources: list[str]) -> dict[str, Any]:
    by: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"peak_gib": [], "wall_ms": []}
    )
    for cell in cells:
        model = _short_model(str(cell.get("model_name") or ""))
        comp = str(cell.get("compressor_name") or "")
        for arm in ("full", "lossy", "exactkv"):
            arm_obj = cell.get(arm) or {}
            peak = arm_obj.get("gpu_peak_allocated_bytes")
            wall = arm_obj.get("wall_clock_ms")
            key = (model, comp, arm)
            if peak is not None:
                by[key]["peak_gib"].append(float(_giB(int(peak)) or 0.0))
            if wall is not None:
                by[key]["wall_ms"].append(float(wall))

    peak_table: dict[str, Any] = {}
    wall_table: dict[str, Any] = {}
    for (model, comp, arm), vals in sorted(by.items()):
        peak_table.setdefault(model, {}).setdefault(comp, {})[arm] = {
            "mean_gib": _mean(vals["peak_gib"]),
            "median_gib": _median(vals["peak_gib"]),
            "n": len(vals["peak_gib"]),
        }
        wall_table.setdefault(model, {}).setdefault(comp, {})[arm] = {
            "mean_ms": _mean(vals["wall_ms"]),
            "median_ms": _median(vals["wall_ms"]),
            "n": len(vals["wall_ms"]),
        }

    pack = {
        "schema": "exactkv.systems.systems_diagnostic.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
        "source_artifacts": sources,
        "n_cells": len(cells),
        "exactkv_failures": sum(1 for c in cells if c.get("exactkv_failure")),
        "peak_cuda_allocation_gib": peak_table,
        "path_wall_clock_ms": wall_table,
        "notes": [
            "Peak is process-level torch.cuda.max_memory_allocated (weights+KV+temps).",
            "ExactKV arm often peaks higher than lossy-only because full+compressed state coexist.",
            "Wall-clock is harness path timing, not serving TTFT/RPS.",
        ],
        "not_measured": [
            "Serving requests-per-second",
            "TTFT / continuous batching",
            "Isolated KV-only VRAM (without model weights)",
            "Packed INT4 production VRAM savings",
        ],
    }
    bad = FORBIDDEN_FIELDS & pack.keys()
    if bad:
        raise ValueError(f"Forbidden top-level fields: {bad}")
    assert_no_forbidden_fields(pack)
    return pack


def render_md(pack: dict[str, Any]) -> str:
    lines = [
        "# Systems diagnostic pack",
        "",
        f"**Cells:** {pack.get('n_cells')} · **exactkv_failures:** {pack.get('exactkv_failures')}",
        "",
        f"**Claim boundary:** {pack.get('claim_boundary')}",
        "",
        "## Peak CUDA allocation (GiB, mean)",
        "",
        "| Model | Compressor | full | lossy | ExactKV |",
        "|-------|------------|-----:|------:|--------:|",
    ]
    peaks = pack.get("peak_cuda_allocation_gib") or {}
    for model, comps in peaks.items():
        for comp, arms in comps.items():
            lines.append(
                f"| {model} | `{comp}` | "
                f"{(arms.get('full') or {}).get('mean_gib')} | "
                f"{(arms.get('lossy') or {}).get('mean_gib')} | "
                f"{(arms.get('exactkv') or {}).get('mean_gib')} |"
            )
    lines += [
        "",
        "## Path wall-clock (ms, mean)",
        "",
        "| Model | Compressor | full | lossy | ExactKV |",
        "|-------|------------|-----:|------:|--------:|",
    ]
    walls = pack.get("path_wall_clock_ms") or {}
    for model, comps in walls.items():
        for comp, arms in comps.items():
            lines.append(
                f"| {model} | `{comp}` | "
                f"{(arms.get('full') or {}).get('mean_ms')} | "
                f"{(arms.get('lossy') or {}).get('mean_ms')} | "
                f"{(arms.get('exactkv') or {}).get('mean_ms')} |"
            )
    lines += ["", "## Notes", ""]
    for n in pack.get("notes") or []:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/external_panels/systems_diagnostic"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/systems/systems_diagnostic.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/systems/systems_diagnostic.md"),
    )
    args = parser.parse_args()

    cells: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in sorted(args.input_dir.glob("*_raw.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        chunk = list(data.get("cells") or [])
        cells.extend(chunk)
        sources.append(str(path))

    if not cells:
        # Allow building from a single combined report if present.
        for path in sorted(args.input_dir.glob("*.json")):
            if path.name in {"summary.json"}:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("cells"), list):
                cells.extend(data["cells"])
                sources.append(str(path))

    pack = build_pack(cells, sources=sources)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_md(pack), encoding="utf-8")
    # Also write a directory summary.json for the panel folder.
    summary_path = args.input_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md} from {len(cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
