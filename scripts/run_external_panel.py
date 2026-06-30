#!/usr/bin/env python3
"""Run ExactKV drift panels on established external benchmark families.

Families (priority): longbench, ruler, bfcl, humaneval, mbpp.

Writes ``reports/external_panels/<family>_raw.json`` and summary markdown.

Examples:
  # Offline CI / no GPU
  python3 scripts/run_external_panel.py --family longbench --deterministic-mode --smoke

  # GPU pilot (bundled prompts)
  python3 scripts/run_external_panel.py --family longbench --device cuda --dtype float16 --max-prompts 6

  # LongBench from Hugging Face (requires datasets + network)
  python3 scripts/run_external_panel.py --family longbench --prompt-source hf --max-prompts 10

  # RULER-style controlled buckets
  python3 scripts/run_external_panel.py --family ruler --context-buckets 4096,8192 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.external_panel import (  # noqa: E402
    run_external_panel,
    write_external_panel_outputs,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="External benchmark drift panel runner")
    parser.add_argument(
        "--family",
        required=True,
        choices=("longbench", "ruler", "bfcl", "humaneval", "mbpp"),
        help="Dataset family to evaluate",
    )
    parser.add_argument("--deterministic-mode", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument(
        "--prompt-source",
        default="pilot",
        choices=("pilot", "hf", "export"),
        help="pilot = bundled JSONL, hf/export = BFCL v3 via HF hub (longbench/humaneval/mbpp/bfcl)",
    )
    parser.add_argument(
        "--longbench-subsets",
        default="",
        help="Comma-separated LongBench HF subsets (default: built-in list)",
    )
    parser.add_argument("--context-buckets", default="", help="Override default family buckets")
    parser.add_argument("--max-new-tokens", default="", help="Comma-separated generation lengths")
    parser.add_argument("--models", default="", help="Comma-separated HF model ids")
    parser.add_argument("--compressors", default="", help="Comma-separated compressor names")
    parser.add_argument(
        "--store-top-k-logits",
        action="store_true",
        help=(
            "For each divergent cell, run an extra forward pass to capture top-5 logit "
            "distributions at the divergence point. Adds ~2 GPU passes per divergent cell."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Override output JSON path (default reports/external_panels/<family>_raw.json)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    compressors = [c.strip() for c in args.compressors.split(",") if c.strip()] or None
    buckets = _parse_int_list(args.context_buckets) if args.context_buckets else None
    mnt = _parse_int_list(args.max_new_tokens) if args.max_new_tokens else None
    subsets = [s.strip() for s in args.longbench_subsets.split(",") if s.strip()] or None

    report = run_external_panel(
        args.family,
        models=models,
        compressors=compressors,
        context_buckets=buckets,
        max_new_tokens_values=mnt,
        max_prompts=args.max_prompts,
        prompt_source=args.prompt_source,
        longbench_subsets=subsets,
        device=args.device,
        dtype=args.dtype,
        deterministic_mode=args.deterministic_mode,
        local_files_only=args.local_files_only,
        smoke=args.smoke,
        store_top_k_logits=args.store_top_k_logits,
    )

    json_path = Path(args.output_json) if args.output_json else None
    jp, mp = write_external_panel_outputs(report, json_path=json_path)
    print(f"Wrote {jp}")
    print(f"Wrote {mp}")
    print(
        f"family={args.family} cells={report['total_cells']} ok={report['cells_run']} "
        f"exactkv_failures={report['exactkv_failures']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
