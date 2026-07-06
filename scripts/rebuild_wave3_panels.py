#!/usr/bin/env python3
"""Rebuild wave-3 panel JSONs by deduplicating cells (prefer ok over skipped)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PANEL_FILES = (
    "bfcl_Llama_3_1_8B_wave3_raw.json",
    "bfcl_Mistral_7B_Instruct_v0_3_wave3_raw.json",
    "longbench_Llama_3_1_8B_wave3_raw.json",
    "longbench_Mistral_7B_Instruct_v0_3_wave3_raw.json",
    "mbpp_Llama_3_1_8B_wave3_raw.json",
    "mbpp_Mistral_7B_Instruct_v0_3_wave3_raw.json",
)


def _resume_key(c: dict) -> tuple:
    return (
        c.get("model_name"),
        c.get("prompt_id"),
        c.get("context_bucket"),
        c.get("compressor_name"),
        c.get("max_new_tokens"),
    )


def _panel_key(c: dict) -> str | None:
    fam = c.get("dataset_family")
    model = c.get("model_name") or ""
    if fam == "bfcl":
        return "bfcl_Llama_3_1_8B_wave3_raw.json" if "Llama" in model else "bfcl_Mistral_7B_Instruct_v0_3_wave3_raw.json"
    if fam == "mbpp":
        return "mbpp_Llama_3_1_8B_wave3_raw.json" if "Llama" in model else "mbpp_Mistral_7B_Instruct_v0_3_wave3_raw.json"
    if fam == "longbench":
        return "longbench_Llama_3_1_8B_wave3_raw.json" if "Llama" in model else "longbench_Mistral_7B_Instruct_v0_3_wave3_raw.json"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("reports/external_panels/faithful/wave3"),
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rank = {"ok": 2, "skipped": 1, "failed": 0, "error": 0}
    best: dict[tuple, dict] = {}

    sources = [args.dir / n for n in PANEL_FILES]
    merged = args.dir / "merged_raw.json"
    if merged.is_file():
        sources.append(merged)

    for path in sources:
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for c in report.get("cells", []):
            k = _resume_key(c)
            if not k[0]:
                continue
            st = c.get("status", "")
            if k not in best or rank.get(st, 0) > rank.get(best[k].get("status", ""), 0):
                best[k] = c

    by_panel: dict[str, list[dict]] = defaultdict(list)
    for c in best.values():
        pk = _panel_key(c)
        if pk:
            by_panel[pk].append(c)

    total_ok = 0
    for name in PANEL_FILES:
        cells = sorted(
            by_panel.get(name, []),
            key=lambda c: (
                c.get("prompt_id", ""),
                c.get("compressor_name", ""),
                c.get("context_bucket") or 0,
                c.get("max_new_tokens") or 0,
            ),
        )
        ok = sum(1 for c in cells if c.get("status") == "ok")
        total_ok += ok
        print(f"{name}: {ok}/{len(cells)} ok")
        if args.write:
            out = args.dir / name
            payload = {
                "panel_id": name.replace("_raw.json", ""),
                "cells": cells,
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  wrote {out}")

    print(f"\nTOTAL ok cells: {total_ok}/576")
    return 0 if total_ok == 576 else 1


if __name__ == "__main__":
    raise SystemExit(main())
