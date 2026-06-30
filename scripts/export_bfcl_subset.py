#!/usr/bin/env python3
"""Export a reproducible BFCL v3 subset JSONL for ExactKV external panels.

Uses huggingface_hub (not datasets.load_dataset) per BFCL upstream guidance.

Example:
  python3 scripts/export_bfcl_subset.py --max-per-category 13 --max-total 50 \\
    --output benchmarks/prompts/bfcl_export.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.external_dataset_loaders import (  # noqa: E402
    BFCL_HF_CATEGORIES,
    export_prompts_jsonl,
    load_bfcl_hf,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BFCL v3 subset for ExactKV panels")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/prompts/bfcl_export.jsonl"),
    )
    parser.add_argument("--max-per-category", type=int, default=13)
    parser.add_argument("--max-total", type=int, default=50)
    args = parser.parse_args()

    rows = load_bfcl_hf(
        categories=BFCL_HF_CATEGORIES,
        max_per_category=args.max_per_category,
        max_total=args.max_total,
    )
    out = export_prompts_jsonl(rows, args.output)
    cats = sorted({r["category"] for r in rows})
    print(f"Exported {len(rows)} prompts to {out}")
    print(f"Categories: {', '.join(cats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
