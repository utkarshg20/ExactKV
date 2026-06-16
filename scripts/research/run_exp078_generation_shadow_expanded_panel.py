#!/usr/bin/env python3
"""Experiment 078: expanded generation-shadow panel (Phase 16M).

Runs ExactKV generation unchanged across prompts, max_new_tokens, and compressors
when cleanly exposed via get_compressor, then post-hoc shadow diagnostics on
prompt_prefix_only and prompt_plus_generated_tokens fixed sequences.

External observer only — not generation integration.
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

from exactkv.attention.generation_shadow_observer import (  # noqa: E402
    DEFAULT_EXP078_COMPRESSORS,
    DEFAULT_EXP078_MAX_NEW_TOKENS,
    DEFAULT_EXP078_REPORT,
    DEFAULT_EXP078_SHADOW_MODES,
    DEFAULT_MODEL_ID,
    EXPERIMENT_078_ID,
    default_exp078_prompts,
    run_exp078_expanded_panel,
    validate_exp078_report,
)
from exactkv.attention.generation_shadow_review import PROPOSED_SHADOW_CLI_FLAG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 078: expanded external generation-shadow panel (Phase 16M)",
    )
    parser.add_argument(
        PROPOSED_SHADOW_CLI_FLAG,
        action="store_true",
        dest="shadow_observer",
        help="Required opt-in flag; default ExactKV generation remains unchanged without it.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXP078_MAX_NEW_TOKENS),
    )
    parser.add_argument(
        "--compressors",
        nargs="+",
        default=list(DEFAULT_EXP078_COMPRESSORS),
    )
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--accumulator-mode", default="float32")
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP078_REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.shadow_observer:
        print(
            f"Skipped: pass {PROPOSED_SHADOW_CLI_FLAG} to run external shadow panel.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_078_ID,
            "status": "skipped",
            "generation_shadow_observer_enabled": False,
            "cli_flag": PROPOSED_SHADOW_CLI_FLAG,
            "blockers": [f"requires {PROPOSED_SHADOW_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    prompts = default_exp078_prompts()[: args.max_prompts]
    report = run_exp078_expanded_panel(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        prompts=prompts,
        max_new_tokens_values=args.max_new_tokens,
        shadow_modes=DEFAULT_EXP078_SHADOW_MODES,
        compressors_requested=args.compressors,
        draft_len=args.draft_len,
        chunk_size=args.chunk_size,
        accumulator_mode=args.accumulator_mode,
        local_files_only=args.local_files_only,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["generation_shadow_observer_enabled"] = True

    errors = validate_exp078_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"generation={report.get('generation_successful_cells')}/"
        f"{report.get('total_generation_cells')} "
        f"shadow={report.get('shadow_successful_cells')}/"
        f"{report.get('total_shadow_cells')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
