#!/usr/bin/env python3
"""Experiment 106: L4 trace-only dry-run panel validation (Phase 21E).

Real-model panel validation for Stage 2 trace-only dry-run scaffold only.
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

from exactkv.safety.l4_trace_only_dry_run_scaffold import (  # noqa: E402
    DEFAULT_EXP106_REPORT,
    DEFAULT_MAX_NEW_TOKENS_VALUES,
    DEFAULT_PANEL_COMPRESSORS,
    DEFAULT_PANEL_MODEL_ID,
    EXPERIMENT_106_ID,
    run_exp106_l4_trace_only_dry_run_panel_validation,
    validate_exp106_panel_report,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 106: L4 trace-only dry-run panel validation",
    )
    parser.add_argument("--model-id", default=DEFAULT_PANEL_MODEL_ID)
    parser.add_argument(
        "--include-instruct",
        action="store_true",
        help="Include Qwen/Qwen2.5-0.5B-Instruct in panel.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens-values",
        default=",".join(str(x) for x in DEFAULT_MAX_NEW_TOKENS_VALUES),
    )
    parser.add_argument(
        "--compressors",
        nargs="+",
        default=list(DEFAULT_PANEL_COMPRESSORS),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-model-blocked", action="store_true", default=True)
    parser.add_argument(
        "--allow-missing-verifier-evidence",
        action="store_true",
        default=True,
        help="Panel may complete with blocked dry-run decisions when verifier evidence missing.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP106_REPORT)
    args = parser.parse_args()

    report = run_exp106_l4_trace_only_dry_run_panel_validation(
        model_id=args.model_id,
        include_instruct=args.include_instruct,
        device=args.device,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
        max_new_tokens_values=_parse_int_list(args.max_new_tokens_values),
        compressors_requested=args.compressors,
        local_files_only=args.local_files_only,
        allow_model_blocked=args.allow_model_blocked,
        allow_missing_verifier_evidence=args.allow_missing_verifier_evidence,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    validation = validate_exp106_panel_report(report)
    report["validation_result"] = validation.to_dict()
    if not validation.valid and report.get("status") == "panel_complete":
        print("Report validation errors:", list(validation.errors), file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"cells={report.get('successful_generation_cells')}/"
        f"{report.get('total_generation_cells')} "
        f"verifier_coverage={report.get('verifier_evidence_coverage_rate'):.3f} "
        f"decision_recommendation={report.get('decision_recommendation')}",
    )
    print(f"validation_valid={validation.valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
