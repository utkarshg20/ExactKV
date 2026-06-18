#!/usr/bin/env python3
"""Experiment 088: longer-context guarded-shadow validation panel (Phase 17C).

Validates guarded decode-time shadow on deterministic long prompts.
Not a benchmark suite or performance test.
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
    PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
)
from exactkv.demo.long_context_validation import (  # noqa: E402
    DEFAULT_COMPRESSORS,
    DEFAULT_EXP088_REPORT,
    DEFAULT_PROMPT_FAMILIES,
    DEFAULT_TARGET_CONTEXT_TOKENS,
    EXPERIMENT_088_ID,
    run_exp088_long_context_validation_panel,
    validate_exp088_report,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 088: longer-context validation panel (Phase 17C)",
    )
    parser.add_argument(
        PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
        action="store_true",
        dest="guarded_decode_time_shadow",
        help="Required: run guarded decode-time shadow validation path.",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--include-instruct", action="store_true")
    parser.add_argument(
        "--target-context-tokens",
        default=",".join(str(x) for x in DEFAULT_TARGET_CONTEXT_TOKENS),
        help="Comma-separated target prompt token lengths (approximate).",
    )
    parser.add_argument(
        "--prompt-families",
        nargs="+",
        default=list(DEFAULT_PROMPT_FAMILIES),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-model-blocked", action="store_true", default=True)
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP088_REPORT)
    args = parser.parse_args()

    if not args.guarded_decode_time_shadow:
        print(
            f"Skipped: pass {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG} to run panel.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_088_ID,
            "status": "skipped",
            "blockers": [f"requires {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp088_long_context_validation_panel(
        model_id=args.model_id,
        include_instruct=args.include_instruct,
        device=args.device,
        dtype=args.dtype,
        target_context_tokens=_parse_int_list(args.target_context_tokens),
        prompt_families=args.prompt_families,
        max_new_tokens=args.max_new_tokens,
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        allow_model_blocked=args.allow_model_blocked,
        allow_shadow_fail=args.allow_shadow_fail,
        max_cells=args.max_cells,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["guarded_decode_time_shadow_enabled"] = True

    errors = validate_exp088_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"models={len(report.get('models_loaded', []))} loaded "
        f"{len(report.get('models_blocked', []))} blocked "
        f"parity={report.get('baseline_vs_guarded_token_match_cells')}/"
        f"{report.get('total_cells')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
