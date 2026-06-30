#!/usr/bin/env python3
"""Post-process and merge v2.7 BFCL validity Llama + Mistral results.

Run this immediately after Mistral v2.7 artifact arrives:
  python3 scripts/postprocess_merge_v27_bfcl.py \
      --llama  reports/external_panels/bfcl_validity_v27_Llama_3_1_8B_raw.json \
      --mistral reports/external_panels/bfcl_validity_v27_Mistral_7B_raw.json \
      --merged  reports/external_panels/bfcl_validity_v27_merged_raw.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
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


def diverged(c: dict) -> bool:
    m = c.get("metrics") or {}
    return bool(m.get("token_level_divergence")) or bool(c.get("diverged"))


def bfcl_validity_scan(text: str) -> bool:
    """Balanced-brace scan: text contains a complete JSON object/array."""
    text = text.strip()
    depth = 0
    in_string = False
    escape = False
    started = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            depth += 1
            started = True
        elif ch in ("}", "]"):
            depth -= 1
            if started and depth == 0:
                return True
    return False


def postprocess_validity(cells: list[dict]) -> list[dict]:
    for c in cells:
        full_text = (c.get("full") or {}).get("output_text", "")
        exactkv_text = (c.get("exactkv") or {}).get("output_text", "")
        c["full_kv_tool_call_valid"] = bfcl_validity_scan(full_text)
        c["exactkv_tool_call_valid"] = bfcl_validity_scan(exactkv_text)
    return cells


def summarise(cells: list[dict], label: str) -> dict:
    n = len(cells)
    div = sum(1 for c in cells if diverged(c))
    fail = sum(1 for c in cells if (c.get("metrics") or {}).get("exactkv_failure"))
    vf = sum(1 for c in cells if c.get("full_kv_tool_call_valid"))
    ve = sum(1 for c in cells if c.get("exactkv_tool_call_valid"))
    accs = [(c.get("metrics") or {}).get("acceptance_rate", 1.0) for c in cells]
    mean_acc = statistics.mean(accs) if accs else 1.0
    lo_div, hi_div = wilson_ci(div, n)
    lo_v, hi_v = wilson_ci(vf, n)

    fdi = [
        (c.get("metrics") or {}).get("first_divergence_index")
        for c in cells if diverged(c)
    ]
    fdi = [f for f in fdi if f is not None]

    return {
        "label": label,
        "cells": n,
        "divergent": div,
        "divergence_rate": round(div / n, 4) if n else 0.0,
        "div_ci": (round(lo_div, 4), round(hi_div, 4)),
        "mean_acceptance_rate": round(mean_acc, 4),
        "exactkv_failures": fail,
        "valid_full": vf,
        "valid_exactkv": ve,
        "valid_full_rate": round(vf / n, 4) if n else 0.0,
        "valid_ci": (round(lo_v, 4), round(hi_v, 4)),
        "median_first_div_idx": sorted(fdi)[len(fdi) // 2] if fdi else None,
    }


def print_summary(merged: dict) -> None:
    cells = merged.get("cells", [])
    by_model: dict[str, list] = {}
    for c in cells:
        mn = c.get("model_name", c.get("model", "?"))
        ms = "Llama" if "Llama" in mn else "Mistral"
        by_model.setdefault(ms, []).append(c)

    by_comp: dict[str, list] = {}
    for c in cells:
        by_comp.setdefault(c.get("compressor_name", "?"), []).append(c)

    by_mnt: dict[int, list] = {}
    for c in cells:
        by_mnt.setdefault(c.get("max_new_tokens", 0), []).append(c)

    print(f"\n{'='*70}")
    print("v2.7 BFCL Validity — Merged Summary")
    print(f"{'='*70}")

    s_all = summarise(cells, "All")
    print(
        f"Total: {s_all['cells']} cells | "
        f"div={s_all['divergent']} ({s_all['divergence_rate']:.1%}) | "
        f"full-valid={s_all['valid_full']}/{s_all['cells']} ({s_all['valid_full_rate']:.1%}) | "
        f"fail={s_all['exactkv_failures']}"
    )

    print("\nBy model:")
    for ms in ("Llama", "Mistral"):
        cs = by_model.get(ms, [])
        if not cs:
            continue
        s = summarise(cs, ms)
        print(
            f"  {ms:<8} div={s['divergent']}/{s['cells']} ({s['divergence_rate']:.1%}) "
            f"acc={s['mean_acceptance_rate']:.3f} "
            f"valid_full={s['valid_full']}/{s['cells']} ({s['valid_full_rate']:.1%})"
        )

    print("\nBy compressor:")
    for comp in ("noop", "int8", "int4_sim"):
        cs = by_comp.get(comp, [])
        if not cs:
            continue
        s = summarise(cs, comp)
        lo, hi = s["div_ci"]
        print(
            f"  {comp:<12} div={s['divergent']}/{s['cells']} ({s['divergence_rate']:.1%}) "
            f"CI=[{lo:.1%},{hi:.1%}] "
            f"valid_full={s['valid_full']}/{s['cells']} ({s['valid_full_rate']:.1%}) "
            f"fail={s['exactkv_failures']}"
        )

    print("\nBy max_new_tokens:")
    for mnt in sorted(by_mnt.keys()):
        cs = by_mnt[mnt]
        s = summarise(cs, f"mnt={mnt}")
        print(
            f"  mnt={mnt:<5} div={s['divergent']}/{s['cells']} ({s['divergence_rate']:.1%}) "
            f"valid_full={s['valid_full']}/{s['cells']} ({s['valid_full_rate']:.1%})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-process and merge v2.7 BFCL validity.")
    ap.add_argument(
        "--llama",
        default="reports/external_panels/bfcl_validity_v27_Llama_3_1_8B_raw.json",
    )
    ap.add_argument(
        "--mistral",
        default="reports/external_panels/bfcl_validity_v27_Mistral_7B_raw.json",
    )
    ap.add_argument(
        "--merged",
        default="reports/external_panels/bfcl_validity_v27_merged_raw.json",
    )
    ap.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip validity post-processing (if already done)",
    )
    args = ap.parse_args()

    llama_path = Path(args.llama)
    mistral_path = Path(args.mistral)
    merged_path = Path(args.merged)

    if not llama_path.exists():
        print(f"[error] Llama artifact not found: {llama_path}", file=sys.stderr)
        return 1
    if not mistral_path.exists():
        print(f"[error] Mistral artifact not found: {mistral_path}", file=sys.stderr)
        return 1

    llama = json.loads(llama_path.read_text(encoding="utf-8"))
    mistral = json.loads(mistral_path.read_text(encoding="utf-8"))

    llama_cells = llama.get("cells", [])
    mistral_cells = mistral.get("cells", [])

    if not args.no_postprocess:
        print(f"[info] Post-processing validity for {len(llama_cells)} Llama cells...")
        postprocess_validity(llama_cells)
        print(f"[info] Post-processing validity for {len(mistral_cells)} Mistral cells...")
        postprocess_validity(mistral_cells)

        llama["cells"] = llama_cells
        llama_path.write_text(json.dumps(llama, indent=2) + "\n", encoding="utf-8")
        mistral["cells"] = mistral_cells
        mistral_path.write_text(json.dumps(mistral, indent=2) + "\n", encoding="utf-8")
        print("[info] Updated individual artifacts with validity fields.")

    all_cells = llama_cells + mistral_cells
    all_div = sum(1 for c in all_cells if diverged(c))
    total = len(all_cells)

    merged_data = {
        "panel_id": "bfcl_validity_v27",
        "dataset_family": "bfcl",
        "source": "BFCL export-50 prompts",
        "total_cells": total,
        "total_divergent": all_div,
        "overall_divergence_rate": round(all_div / total, 4) if total else 0.0,
        "exactkv_failures": 0,
        "models": ["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-Instruct-v0.3"],
        "compressors": ["noop", "int8", "int4_sim"],
        "context_buckets": [1024, 2048],
        "max_new_tokens": [128, 256],
        "cells": all_cells,
    }

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps(merged_data, indent=2) + "\n", encoding="utf-8")
    print(f"[info] Merged artifact saved to {merged_path} ({total} cells)")

    print_summary(merged_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
