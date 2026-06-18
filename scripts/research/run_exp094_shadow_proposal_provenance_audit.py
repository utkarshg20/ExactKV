#!/usr/bin/env python3
"""Experiment 094: L3 shadow proposal provenance audit (Phase 18E).

Audits proposal provenance and recommends whether decode_time_shadow_top1 remains viable.
Not L4 verifier-mediated acceptance.
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
    DEFAULT_EXP094_REPORT,
    DEFAULT_MAX_NEW_TOKENS_VALUES,
    DEFAULT_MODEL_ID,
    DEFAULT_PANEL_COMPRESSORS,
    EXPERIMENT_094_ID,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCES,
    PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG,
    run_exp094_shadow_proposal_provenance_audit,
    validate_exp094_report,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 094: L3 shadow proposal provenance audit",
    )
    parser.add_argument(
        PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG,
        action="store_true",
        dest="guarded_draft_shadow_no_commit",
        help="Required: run L3 shadow proposal provenance audit panel.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens-values",
        default=",".join(str(x) for x in DEFAULT_MAX_NEW_TOKENS_VALUES),
    )
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_PANEL_COMPRESSORS))
    parser.add_argument(
        "--proposal-source",
        choices=list(PROPOSAL_SOURCES),
        default=PROPOSAL_SOURCE_DECODE_TOP1,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-provider-blocked", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP094_REPORT)
    args = parser.parse_args()

    if not args.guarded_draft_shadow_no_commit:
        print(
            f"Skipped: pass {PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG} to run audit.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_094_ID,
            "status": "skipped",
            "blockers": [f"requires {PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp094_shadow_proposal_provenance_audit(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
        max_new_tokens_values=_parse_int_list(args.max_new_tokens_values),
        compressors_requested=args.compressors,
        proposal_source=args.proposal_source,
        local_files_only=args.local_files_only,
        allow_provider_blocked=args.allow_provider_blocked,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["guarded_draft_shadow_no_commit_enabled"] = True

    errors = validate_exp094_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"audited={report.get('total_audited_rounds')} "
        f"safe={report.get('safe_extraction_count')} "
        f"match_rate_successful={report.get('match_rate_successful_extractions'):.3f} "
        f"decision={report.get('decision_recommendation')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
