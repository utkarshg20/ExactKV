#!/usr/bin/env python3
"""Experiment 084: expanded guarded decode-time shadow panel (Phase 16S).

Runs baseline vs guarded-shadow generation across prompts, compressors, and
max_new_tokens values; compares decode-time vs post-hoc shadow diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.decode_time_shadow_observer import (  # noqa: E402
    DEFAULT_EXP084_COMPRESSORS,
    DEFAULT_EXP084_MAX_NEW_TOKENS_VALUES,
    DEFAULT_EXP084_REPORT,
    EXPERIMENT_084_ID,
    PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
    default_exp084_prompts,
    run_exp084_guarded_decode_time_shadow_panel,
    validate_exp084_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 084: expanded guarded decode-time shadow panel (Phase 16S)",
    )
    parser.add_argument(
        PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
        action="store_true",
        dest="guarded_decode_time_shadow",
        help="Required: enable guarded decode-time shadow panel.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_EXP084_MAX_NEW_TOKENS_VALUES),
    )
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_EXP084_COMPRESSORS))
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP084_REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    args = parser.parse_args()

    if not args.guarded_decode_time_shadow:
        print(
            f"Skipped: pass {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG} to run panel.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_084_ID,
            "status": "skipped",
            "blockers": [f"requires {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    prompts = default_exp084_prompts()[: args.max_prompts]
    report = run_exp084_guarded_decode_time_shadow_panel(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        prompts=prompts,
        max_new_tokens_values=args.max_new_tokens_values,
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        allow_shadow_fail=args.allow_shadow_fail,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["guarded_decode_time_shadow_enabled"] = True

    errors = validate_exp084_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"parity={report.get('baseline_vs_guarded_token_match_cells')}/"
        f"{report.get('total_cells')} "
        f"callbacks={report.get('decode_time_shadow_callback_count')} "
        f"dt_vs_posthoc={report.get('decode_time_vs_posthoc_shadow_match_cells')}/"
        f"{report.get('posthoc_shadow_comparison_cells')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
