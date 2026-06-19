#!/usr/bin/env python3
"""Experiment 096: L3 round-log vs shadow top-1 proposal source comparison (Phase 19B).

Side-by-side comparison panel for L3 proposal sources. Not L4 verifier-mediated acceptance.
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

from exactkv.safety.guarded_draft_shadow import (  # noqa: E402
    DEFAULT_COMPARISON_MODEL_IDS,
    DEFAULT_COMPARISON_PROPOSAL_SOURCES,
    DEFAULT_EXP096_REPORT,
    DEFAULT_MAX_NEW_TOKENS_VALUES,
    DEFAULT_PANEL_COMPRESSORS,
    EXPERIMENT_096_ID,
    PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG,
    PROPOSAL_SOURCES,
    run_exp096_round_log_proposal_source_comparison_panel,
    validate_exp096_report,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 096: L3 proposal source comparison panel",
    )
    parser.add_argument(
        PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG,
        action="store_true",
        dest="guarded_draft_shadow_no_commit",
        help="Required: run L3 proposal source comparison panel.",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=list(DEFAULT_COMPARISON_MODEL_IDS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens-values",
        default=",".join(str(x) for x in DEFAULT_MAX_NEW_TOKENS_VALUES),
    )
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_PANEL_COMPRESSORS))
    parser.add_argument(
        "--proposal-sources",
        nargs="+",
        default=list(DEFAULT_COMPARISON_PROPOSAL_SOURCES),
        choices=list(PROPOSAL_SOURCES),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-model-blocked", action="store_true", default=True)
    parser.add_argument("--allow-provider-blocked", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP096_REPORT)
    args = parser.parse_args()

    if not args.guarded_draft_shadow_no_commit:
        print(
            f"Skipped: pass {PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG} to run panel.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_096_ID,
            "status": "skipped",
            "blockers": [f"requires {PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp096_round_log_proposal_source_comparison_panel(
        model_ids=args.model_ids,
        device=args.device,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
        max_new_tokens_values=_parse_int_list(args.max_new_tokens_values),
        compressors_requested=args.compressors,
        proposal_sources=args.proposal_sources,
        local_files_only=args.local_files_only,
        allow_model_blocked=args.allow_model_blocked,
        allow_provider_blocked=args.allow_provider_blocked,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["guarded_draft_shadow_no_commit_enabled"] = True

    errors = validate_exp096_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"decision={report.get('decision_recommendation')}")
    sbs = report.get("side_by_side_summary") or {}
    print(
        f"cells={report.get('successful_generation_cells')}/"
        f"{report.get('total_generation_cells')} "
        f"both_available={sbs.get('rounds_where_both_sources_available')} "
        f"agree={sbs.get('rounds_where_sources_agree')}"
    )
    for summ in report.get("source_summaries") or []:
        print(
            f"  {summ.get('proposal_source')}: "
            f"coverage={summ.get('proposal_coverage_rate'):.3f} "
            f"prefix_match={summ.get('prefix_match_rate'):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
