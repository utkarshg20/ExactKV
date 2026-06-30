#!/usr/bin/env python3
"""Evidence-plus panel runner — long-context + external compressors + timing.

Writes ``reports/evidence_plus/raw.json`` and ``reports/evidence_plus/summary.md``.

Example (RunPod A5000):
  python3 scripts/run_evidence_plus_panel.py --device cuda --dtype float16

Smoke (no GPU):
  python3 scripts/run_evidence_plus_panel.py --deterministic-mode --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.evidence_plus_panel import (  # noqa: E402
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    EVIDENCE_PLUS_ID,
    run_evidence_plus_panel,
    write_evidence_plus_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidence-plus benchmark panel")
    parser.add_argument(
        "--deterministic-mode",
        action="store_true",
        help="Hash-seeded synthetic cells (no GPU / no HF download)",
    )
    parser.add_argument("--smoke", action="store_true", help="Minimal grid for quick validation")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda, cpu)")
    parser.add_argument("--dtype", default="float16", help="Torch dtype (float16, bfloat16)")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached HF weights",
    )
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Skip KIVI / KVQuant / SnapKV adapters",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=8,
        help="Number of base prompts (long_context + stress panel)",
    )
    parser.add_argument(
        "--context-buckets",
        type=str,
        default="512,1024,2048",
        help="Comma-separated prefill token targets",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=str,
        default="16,32,64",
        help="Comma-separated generation lengths",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated HF model ids (default: Llama-8B + Mistral-7B)",
    )
    parser.add_argument(
        "--compressors",
        type=str,
        default="",
        help="Comma-separated compressor names (default: builtins + external try)",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    compressors = [c.strip() for c in args.compressors.split(",") if c.strip()] or None
    buckets = [int(x.strip()) for x in args.context_buckets.split(",") if x.strip()]
    mnt = [int(x.strip()) for x in args.max_new_tokens.split(",") if x.strip()]

    report = run_evidence_plus_panel(
        models=models,
        compressors=compressors,
        context_buckets=buckets,
        max_new_tokens_values=mnt,
        max_prompts=args.max_prompts,
        device=args.device,
        dtype=args.dtype,
        deterministic_mode=args.deterministic_mode,
        local_files_only=args.local_files_only,
        try_external=not args.no_external,
        smoke=args.smoke,
    )

    json_path, md_path = write_evidence_plus_outputs(
        report,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )

    print(f"phase_id={EVIDENCE_PLUS_ID}")
    print(f"status={report['status']}")
    print(f"models_evaluated={report.get('models_evaluated')}")
    print(f"total_cells={report.get('total_cells')} ok={report.get('cells_run')} skipped={report.get('cells_skipped')}")
    print(f"exactkv_failures={report.get('exactkv_failures')}")
    print(f"bucket_summary={report.get('bucket_summary')}")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
