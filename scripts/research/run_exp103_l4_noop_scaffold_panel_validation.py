#!/usr/bin/env python3
"""Experiment 103: L4 no-op scaffold panel validation (Phase 21B).

Real-model panel validation for Stage 1 no-op scaffold only.
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
    DEFAULT_EXP103_REPORT,
    DEFAULT_MAX_NEW_TOKENS_VALUES,
    DEFAULT_PANEL_COMPRESSORS,
    DEFAULT_PANEL_MODEL_IDS,
    EXPERIMENT_103_ID,
    L4_OPT_IN_FLAG,
    run_exp103_l4_noop_scaffold_panel_validation,
    validate_exp103_panel_report,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 103: L4 no-op scaffold panel validation",
    )
    parser.add_argument(
        L4_OPT_IN_FLAG,
        action="store_true",
        dest="l4_noop_enabled",
        help="Required: enable L4 no-op opt-in scaffold (research script only).",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=list(DEFAULT_PANEL_MODEL_IDS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens-values",
        default=",".join(str(x) for x in DEFAULT_MAX_NEW_TOKENS_VALUES),
    )
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_PANEL_COMPRESSORS))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-model-blocked", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP103_REPORT)
    args = parser.parse_args()

    if not args.l4_noop_enabled:
        print(
            f"Skipped: pass {L4_OPT_IN_FLAG} to run panel validation.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_103_ID,
            "status": "skipped",
            "blockers": [f"requires {L4_OPT_IN_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp103_l4_noop_scaffold_panel_validation(
        l4_noop_enabled=True,
        model_ids=args.model_ids,
        device=args.device,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
        max_new_tokens_values=_parse_int_list(args.max_new_tokens_values),
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        allow_model_blocked=args.allow_model_blocked,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    validation = validate_exp103_panel_report(report)
    if not validation.valid and report.get("status") == "panel_complete":
        print("Report validation errors:", list(validation.errors), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"parity={report.get('token_match_cells')}/{report.get('total_cells')} "
        f"models_loaded={len(report.get('models_loaded') or [])} "
        f"models_blocked={len(report.get('models_blocked') or [])}",
    )
    validation_dict = report.get("validation_result") or {}
    print(f"validation_valid={validation_dict.get('valid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
