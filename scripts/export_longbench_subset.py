#!/usr/bin/env python3
"""Export a reproducible LongBench subset JSONL for ExactKV external panels.

Requires: pip install "datasets>=3.2,<4" (LongBench uses HF dataset scripts),
network access, optional HF token. Set HF_DATASETS_TRUST_REMOTE_CODE=1 or rely
on trust_remote_code=True in the loader.

Example:
  python3 scripts/export_longbench_subset.py --max-per-subset 2 --output benchmarks/prompts/longbench_export.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.external_dataset_loaders import (  # noqa: E402
    LONGBENCH_HF_SUBSETS,
    export_prompts_jsonl,
    load_longbench_hf,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LongBench subset for ExactKV panels")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/prompts/longbench_export.jsonl"))
    parser.add_argument("--max-per-subset", type=int, default=2)
    parser.add_argument("--max-total", type=int, default=24)
    parser.add_argument(
        "--subsets",
        default="",
        help=f"Comma-separated subsets (default: {','.join(LONGBENCH_HF_SUBSETS[:6])}...)",
    )
    args = parser.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()] or None
    rows = load_longbench_hf(
        subsets=subsets,
        max_per_subset=args.max_per_subset,
        max_total=args.max_total,
    )
    out = export_prompts_jsonl(rows, args.output)
    print(f"Exported {len(rows)} prompts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
