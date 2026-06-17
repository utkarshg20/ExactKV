#!/usr/bin/env python3
"""Experiment 081: live round observer smoke (Phase 16P).

Runs baseline ExactKV generation vs opt-in live round observer instrumentation
and compares token/text parity plus snapshot-vs-round-log agreement.
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

from exactkv.attention.generation_shadow_review import (  # noqa: E402
    PROPOSED_SHADOW_CLI_FLAG,
)
from exactkv.attention.generation_shadow_observer import default_exp080_prompts  # noqa: E402
from exactkv.attention.live_round_observer import (  # noqa: E402
    DEFAULT_EXP081_REPORT,
    EXPERIMENT_081_ID,
    PROPOSED_LIVE_OBSERVER_CLI_FLAG,
    run_exp081_live_round_observer_panel,
    validate_exp081_report,
)

DEFAULT_COMPRESSORS = ("noop", "int8", "int4_sim", "k8_v4_sim")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 081: opt-in live round observer smoke (Phase 16P)",
    )
    parser.add_argument(
        PROPOSED_LIVE_OBSERVER_CLI_FLAG,
        action="store_true",
        dest="live_round_observer",
        help="Required opt-in flag to enable live round observer instrumentation.",
    )
    parser.add_argument(
        PROPOSED_SHADOW_CLI_FLAG,
        action="store_true",
        dest="generation_shadow_observer",
        help="Optional post-hoc shadow analysis after generation.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS))
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP081_REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    args = parser.parse_args()

    if not args.live_round_observer:
        print(
            f"Skipped: pass {PROPOSED_LIVE_OBSERVER_CLI_FLAG} to run live observer smoke.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_081_ID,
            "status": "skipped",
            "live_round_observer_enabled": False,
            "cli_flag": PROPOSED_LIVE_OBSERVER_CLI_FLAG,
            "blockers": [f"requires {PROPOSED_LIVE_OBSERVER_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    prompts = default_exp080_prompts()[: args.max_prompts]
    report = run_exp081_live_round_observer_panel(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        run_posthoc_shadow=args.generation_shadow_observer,
        allow_shadow_fail=args.allow_shadow_fail,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["live_round_observer_enabled"] = True
    report["generation_shadow_observer_enabled"] = bool(args.generation_shadow_observer)

    errors = validate_exp081_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"baseline={report.get('baseline_generation_successful_cells')}/"
        f"{report.get('total_cells')} "
        f"observer={report.get('observer_generation_successful_cells')}/"
        f"{report.get('total_cells')} "
        f"token_match={report.get('baseline_vs_observer_token_match_cells')}/"
        f"{report.get('total_cells')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
