#!/usr/bin/env python3
"""Build length-opportunity pack: FDI / max_new and per-token hazard from existing panels.

Cell-level divergence rises with generation budget partly because each extra token
is another argmax trial. This pack reports both cell divergence and early-vs-late FDI.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PANELS = [
    ("mbpp_smoke", "reports/external_panels/mbpp_gpu_raw.json", "mbpp"),
    ("bfcl_export_50", "reports/external_panels/bfcl_export_50_raw.json", "bfcl_short"),
    ("bfcl_validity_v27", "reports/external_panels/bfcl_validity_v27_merged_raw.json", "bfcl_long"),
    ("hf_longbench_v26", "reports/external_panels/hf_longbench_v26_merged_raw.json", "longbench"),
]


def _fdi(cell: dict[str, Any]) -> int | None:
    metrics = cell.get("metrics") or {}
    for key in ("first_divergence_index",):
        val = metrics.get(key)
        if val is not None:
            return int(val)
    lossy = cell.get("lossy") or {}
    val = lossy.get("first_divergence_idx")
    if val is not None:
        return int(val)
    return None


def _divergent(cell: dict[str, Any]) -> bool:
    metrics = cell.get("metrics") or {}
    if metrics.get("token_level_divergence") is not None:
        return bool(metrics.get("token_level_divergence"))
    lossy = cell.get("lossy") or {}
    return lossy.get("token_exact_match") is False


def _exposure_tokens(cell: dict[str, Any], *, divergent: bool, fdi: int | None) -> int:
    mnt = int(cell.get("max_new_tokens") or 0)
    if mnt <= 0:
        return 0
    if divergent and fdi is not None:
        # Tokens until first divergence (inclusive opportunity count ≈ fdi, 1-indexed).
        return max(1, min(mnt, int(fdi)))
    return mnt


def summarise_panel(path: Path, *, family: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in (data.get("cells") or []) if c.get("status") == "ok"]
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        comp = cell.get("compressor_name") or cell.get("compressor")
        if not comp:
            continue
        by_comp[str(comp)].append(cell)

    out_by_comp: dict[str, Any] = {}
    for comp, group in sorted(by_comp.items()):
        by_mnt: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cell in group:
            mnt = int(cell.get("max_new_tokens") or 0)
            by_mnt[mnt].append(cell)

        mnt_rows = []
        exposure_div = 0
        exposure_tokens = 0
        fdi_ratios: list[float] = []
        fdi_abs: list[float] = []
        n_div = 0
        for mnt, bucket in sorted(by_mnt.items()):
            local_fdi: list[float] = []
            n_local_div = 0
            for cell in bucket:
                divergent = _divergent(cell)
                fdi = _fdi(cell) if divergent else None
                exp = _exposure_tokens(cell, divergent=divergent, fdi=fdi)
                exposure_tokens += exp
                if divergent:
                    n_div += 1
                    exposure_div += 1
                    n_local_div += 1
                    if fdi is not None and mnt > 0:
                        fdi_abs.append(float(fdi))
                        fdi_ratios.append(float(fdi) / float(mnt))
                        local_fdi.append(float(fdi))
            mnt_rows.append(
                {
                    "max_new_tokens": mnt,
                    "cells": len(bucket),
                    "divergent_cells": n_local_div,
                    "divergence_rate": round(n_local_div / len(bucket), 4) if bucket else None,
                    "mean_fdi_divergent": round(mean(local_fdi), 3) if local_fdi else None,
                    "mean_fdi_over_mnt": round(mean([f / mnt for f in local_fdi]), 4)
                    if local_fdi and mnt > 0
                    else None,
                }
            )

        hazard = round(exposure_div / exposure_tokens, 6) if exposure_tokens else None
        out_by_comp[comp] = {
            "cells": len(group),
            "divergent_cells": n_div,
            "divergence_rate": round(n_div / len(group), 4) if group else None,
            "mean_fdi_divergent": round(mean(fdi_abs), 3) if fdi_abs else None,
            "mean_fdi_over_mnt": round(mean(fdi_ratios), 4) if fdi_ratios else None,
            "per_token_hazard_proxy": hazard,
            "hazard_definition": (
                "divergent_cells / sum(tokens_until_first_divergence_or_max_new); "
                "not a formal survival model"
            ),
            "by_max_new_tokens": mnt_rows,
        }

    return {
        "family": family,
        "source": str(path.relative_to(ROOT)),
        "cells_ok": len(cells),
        "by_compressor": out_by_comp,
    }


def render_markdown(doc: dict[str, Any]) -> str:
    lines = [
        "# Length-opportunity pack",
        "",
        "Cell-level divergence rises with `max_new_tokens` partly because each extra "
        "token is another greedy argmax trial. Report **both** cell divergence rate and "
        "FDI / `max_new` (early vs late).",
        "",
        f"Generated: `{doc['generated_at']}`",
        "",
    ]
    for panel in doc["panels"]:
        lines.append(f"## {panel['family']} (`{panel['source']}`)")
        lines.append("")
        lines.append("| Compressor | n | Div. rate | Mean FDI (div) | Mean FDI/`mnt` | Hazard proxy |")
        lines.append("|------------|--:|----------:|---------------:|---------------:|-------------:|")
        for comp, stats in panel["by_compressor"].items():
            lines.append(
                "| `{comp}` | {n} | {div} | {fdi} | {ratio} | {haz} |".format(
                    comp=comp,
                    n=stats["cells"],
                    div="—" if stats["divergence_rate"] is None else f"{100 * stats['divergence_rate']:.1f}%",
                    fdi="—" if stats["mean_fdi_divergent"] is None else stats["mean_fdi_divergent"],
                    ratio="—" if stats["mean_fdi_over_mnt"] is None else stats["mean_fdi_over_mnt"],
                    haz="—" if stats["per_token_hazard_proxy"] is None else f"{stats['per_token_hazard_proxy']:.4f}",
                )
            )
        lines.append("")
    lines.append(
        "Hazard proxy is diagnostic only — not a Kaplan–Meier / formal survival estimate."
    )
    lines.append("")
    return "\n".join(lines)


def build(*, panels: list[tuple[str, str, str]], out_json: Path, out_md: Path) -> dict[str, Any]:
    panel_docs = []
    for _name, rel, family in panels:
        path = ROOT / rel
        if not path.is_file():
            continue
        panel_docs.append(summarise_panel(path, family=family))

    doc = {
        "schema": "exactkv.public_release.length_opportunity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Opportunity framing from committed panels only. Cross-family divergence "
            "spans remain observational (task, context, and max_new often change together). "
            "Within-task BFCL mnt slices are the controlled generation-length evidence."
        ),
        "panels": panel_docs,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(doc), encoding="utf-8")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "reports/public_release/length_opportunity.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "reports/public_release/length_opportunity.md",
    )
    args = ap.parse_args()
    doc = build(panels=DEFAULT_PANELS, out_json=args.out_json, out_md=args.out_md)
    print(f"Wrote {args.out_json} ({len(doc['panels'])} panels)")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
