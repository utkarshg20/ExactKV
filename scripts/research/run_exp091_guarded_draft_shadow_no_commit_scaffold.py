#!/usr/bin/env python3
"""Experiment 091: L3 guarded draft-shadow no-commit scaffold (Phase 18B).

Draft-shadow proposals are diagnostic only and cannot affect token commits.
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
    DEFAULT_COMPRESSORS,
    DEFAULT_EXP091_REPORT,
    DEFAULT_MODEL_ID,
    EXPERIMENT_091_ID,
    PROPOSAL_SOURCE_DECODE_TOP1,
    PROPOSAL_SOURCE_SYNTHETIC,
    PROPOSAL_SOURCES,
    PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG,
    run_exp091_guarded_draft_shadow_no_commit_scaffold,
    validate_exp091_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 091: L3 guarded draft-shadow no-commit scaffold",
    )
    parser.add_argument(
        PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG,
        action="store_true",
        dest="guarded_draft_shadow_no_commit",
        help="Required: run L3 guarded draft-shadow no-commit scaffold.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--compressors", nargs="+", default=list(DEFAULT_COMPRESSORS))
    parser.add_argument(
        "--proposal-source",
        choices=list(PROPOSAL_SOURCES),
        default=PROPOSAL_SOURCE_DECODE_TOP1,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-provider-blocked", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP091_REPORT)
    args = parser.parse_args()

    if not args.guarded_draft_shadow_no_commit:
        print(
            f"Skipped: pass {PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG} to run scaffold.",
            file=sys.stderr,
        )
        report = {
            "experiment_id": EXPERIMENT_091_ID,
            "status": "skipped",
            "blockers": [f"requires {PROPOSED_GUARDED_DRAFT_SHADOW_NO_COMMIT_CLI_FLAG}"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp091_guarded_draft_shadow_no_commit_scaffold(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        compressors_requested=args.compressors,
        proposal_source=args.proposal_source,
        local_files_only=args.local_files_only,
        allow_provider_blocked=args.allow_provider_blocked,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["guarded_draft_shadow_no_commit_enabled"] = True

    errors = validate_exp091_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(
        f"parity={report.get('baseline_vs_draft_shadow_token_match_cells')}/"
        f"{report.get('total_cells')} "
        f"proposals={report.get('total_proposals')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
