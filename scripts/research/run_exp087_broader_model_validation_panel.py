#!/usr/bin/env python3
"""Experiment 087: broader model validation panel (Phase 17B).

Runs guarded decode-time shadow diagnostics across a small Qwen model panel.
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
from exactkv.demo.broader_model_validation import (  # noqa: E402
    DEFAULT_COMPRESSORS,
    DEFAULT_EXP087_REPORT,
    DEFAULT_MODEL_IDS,
    EXPERIMENT_087_ID,
    OPTIONAL_MODEL_IDS,
    run_exp087_broader_model_validation_panel,
    validate_exp087_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 087: broader model validation panel (Phase 17B)",
    )
    parser.add_argument(
        PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG,
        action="store_true",
        dest="guarded_decode_time_shadow",
        help="Required: run guarded decode-time shadow validation path.",
    )
    parser.add_argument("--include-optional-models", action="store_true")
    parser.add_argument("--model-ids", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-model-blocked", action="store_true", default=True)
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP087_REPORT)
    args = parser.parse_args()

    if not args.guarded_decode_time_shadow:
        print(
            f"Skipped: pass {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG} to run panel.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_087_ID,
            "status": "skipped",
            "blockers": [f"requires {PROPOSED_GUARDED_DECODE_TIME_SHADOW_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp087_broader_model_validation_panel(
        model_ids=args.model_ids,
        include_optional_models=args.include_optional_models,
        device=args.device,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        allow_model_blocked=args.allow_model_blocked,
        allow_shadow_fail=args.allow_shadow_fail,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["guarded_decode_time_shadow_enabled"] = True

    errors = validate_exp087_report(report)
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
