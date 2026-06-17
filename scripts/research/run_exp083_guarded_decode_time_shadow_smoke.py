#!/usr/bin/env python3
"""Experiment 083: guarded decode-time shadow observer smoke (Phase 16R).

Runs baseline vs guarded decode-time shadow observer generation parity, then
compares decode-time shadow callbacks with post-hoc shadow on the same snapshots.
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
    DEFAULT_EXP083_REPORT,
    EXPERIMENT_083_ID,
    PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
    default_exp083_prompts,
    run_exp083_guarded_decode_time_shadow_smoke,
    validate_exp083_report,
)

DEFAULT_COMPRESSORS = ("noop", "int8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 083: guarded decode-time shadow observer smoke (Phase 16R)",
    )
    parser.add_argument(
        PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
        action="store_true",
        dest="guarded_decode_time_shadow",
        help="Required: enable guarded decode-time shadow observer dry-run.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS))
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP083_REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    args = parser.parse_args()

    if not args.guarded_decode_time_shadow:
        print(
            f"Skipped: pass {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG} to run smoke.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_083_ID,
            "status": "skipped",
            "blockers": [f"requires {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    prompts = default_exp083_prompts()[: args.max_prompts]
    report = run_exp083_guarded_decode_time_shadow_smoke(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        allow_shadow_fail=args.allow_shadow_fail,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["guarded_decode_time_shadow_enabled"] = True

    errors = validate_exp083_report(report)
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
        f"decode_callbacks={report.get('decode_time_shadow_callback_count')} "
        f"dt_vs_posthoc={report.get('decode_time_vs_posthoc_shadow_match_cells')}/"
        f"{report.get('posthoc_shadow_comparison_cells')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
