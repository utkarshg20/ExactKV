#!/usr/bin/env python3
"""Experiment 086: claim-safe demo packaging (Phase 17A).

Packages Phase 16 evidence into a claim-safe demo narrative and demo cards.
No new runtime functionality or experiments.
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

from exactkv.demo.phase17_claim_safe_demo import (  # noqa: E402
    DEFAULT_EXP086_REPORT,
    EXPERIMENT_086_ID,
    run_exp086_claim_safe_demo_packaging,
    validate_exp086_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 086: claim-safe demo packaging (Phase 17A)",
    )
    parser.add_argument(
        "--package-demo",
        action="store_true",
        help="Build claim-safe demo narrative and cards from Phase 16 evidence.",
    )
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP086_REPORT)
    args = parser.parse_args()

    if not args.package_demo:
        print("Skipped: pass --package-demo to build claim-safe demo package.", file=sys.stderr)
        report = {
            "experiment_id": EXPERIMENT_086_ID,
            "status": "skipped",
            "blockers": ["requires --package-demo"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp086_claim_safe_demo_packaging(root=args.root)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp086_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"docs={len(report.get('source_docs_found', []))} found, "
          f"{len(report.get('source_docs_missing', []))} missing")
    print(f"cards={len(report.get('demo_cards', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
