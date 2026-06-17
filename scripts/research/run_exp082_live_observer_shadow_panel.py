#!/usr/bin/env python3
"""Experiment 082: live observer + post-hoc shadow panel (Phase 16Q).

Runs baseline vs live-observer generation parity, then post-hoc round-boundary
shadow diagnostics from live snapshots after generation completes.
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

from exactkv.attention.generation_shadow_review import PROPOSED_SHADOW_CLI_FLAG  # noqa: E402
from exactkv.attention.generation_shadow_observer import default_exp080_prompts  # noqa: E402
from exactkv.attention.live_round_observer import (  # noqa: E402
    DEFAULT_EXP082_REPORT,
    EXPERIMENT_082_ID,
    PROPOSED_LIVE_OBSERVER_CLI_FLAG,
    run_exp082_live_observer_shadow_panel,
    validate_exp082_report,
)

DEFAULT_COMPRESSORS = ("noop", "int8", "int4_sim", "k8_v4_sim")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 082: live observer + post-hoc shadow panel (Phase 16Q)",
    )
    parser.add_argument(
        PROPOSED_LIVE_OBSERVER_CLI_FLAG,
        action="store_true",
        dest="live_round_observer",
        help="Required: enable live round observer instrumentation.",
    )
    parser.add_argument(
        PROPOSED_SHADOW_CLI_FLAG,
        action="store_true",
        dest="generation_shadow_observer",
        help="Required: run post-hoc shadow after generation.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS))
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP082_REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-shadow-fail", action="store_true", default=True)
    args = parser.parse_args()

    if not args.live_round_observer or not args.generation_shadow_observer:
        print(
            f"Skipped: pass {PROPOSED_LIVE_OBSERVER_CLI_FLAG} and "
            f"{PROPOSED_SHADOW_CLI_FLAG} to run panel.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_082_ID,
            "status": "skipped",
            "blockers": [
                f"requires {PROPOSED_LIVE_OBSERVER_CLI_FLAG}",
                f"requires {PROPOSED_SHADOW_CLI_FLAG}",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    prompts = default_exp080_prompts()[: args.max_prompts]
    report = run_exp082_live_observer_shadow_panel(
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
    report["live_round_observer_enabled"] = True
    report["generation_shadow_observer_enabled"] = True

    errors = validate_exp082_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"parity={report.get('baseline_vs_observer_token_match_cells')}/"
        f"{report.get('total_cells')} "
        f"shadow={report.get('posthoc_shadow_successful_cells')}/"
        f"{report.get('posthoc_shadow_successful_cells', 0) + report.get('posthoc_shadow_blocked_cells', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
