#!/usr/bin/env python3
"""Experiment 102: L4 no-op opt-in scaffold (Phase 21A).

Stage 1 no-op scaffolding only — not L4 runtime commit integration.
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

from exactkv.safety.l4_noop_opt_in_scaffold import (  # noqa: E402
    DEFAULT_COMPRESSORS,
    DEFAULT_EXP102_REPORT,
    DEFAULT_MODEL_ID,
    EXPERIMENT_102_ID,
    L4_OPT_IN_FLAG,
    run_exp102_l4_noop_opt_in_scaffold,
    validate_exp102_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 102: L4 no-op opt-in scaffold",
    )
    parser.add_argument(
        L4_OPT_IN_FLAG,
        action="store_true",
        dest="l4_noop_enabled",
        help="Required: enable L4 no-op opt-in scaffold (research script only).",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-model-blocked", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP102_REPORT)
    args = parser.parse_args()

    if not args.l4_noop_enabled:
        print(
            f"Skipped: pass {L4_OPT_IN_FLAG} to run L4 no-op scaffold.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_102_ID,
            "status": "skipped",
            "blockers": [f"requires {L4_OPT_IN_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp102_l4_noop_opt_in_scaffold(
        l4_noop_enabled=True,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        allow_model_blocked=args.allow_model_blocked,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp102_report(report)
    if errors and report.get("status") == "scaffold_complete":
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"parity={report.get('token_match_cells')}/{report.get('total_cells')} "
        f"baseline={report.get('successful_baseline_cells')} "
        f"noop={report.get('successful_noop_scaffold_cells')}",
    )
    validation = report.get("validation_result") or {}
    print(f"validation_valid={validation.get('valid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
