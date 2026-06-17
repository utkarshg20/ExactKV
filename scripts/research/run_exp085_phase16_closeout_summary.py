#!/usr/bin/env python3
"""Experiment 085: Phase 16 closeout summary and claim freeze (Phase 16T).

Inspects local Phase 16 evidence (Exp 066–084) and produces a machine-readable
closeout summary without adding shadow or integration functionality.
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

from exactkv.attention.phase16_closeout import (  # noqa: E402
    DEFAULT_EXP085_REPORT,
    EXPERIMENT_085_ID,
    run_exp085_phase16_closeout_summary,
    validate_exp085_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 085: Phase 16 closeout summary (Phase 16T)",
    )
    parser.add_argument(
        "--run-closeout",
        action="store_true",
        help="Run Phase 16 evidence inventory and claim freeze summary.",
    )
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP085_REPORT)
    args = parser.parse_args()

    if not args.run_closeout:
        print("Skipped: pass --run-closeout to generate Phase 16 closeout summary.", file=sys.stderr)
        report = {
            "experiment_id": EXPERIMENT_085_ID,
            "status": "skipped",
            "blockers": ["requires --run-closeout"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = run_exp085_phase16_closeout_summary(
        root=args.root,
        reports_root=args.reports_root,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp085_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"phase16_status={report.get('phase16_status')}")
    print(
        f"reports={len(report.get('reports_found', []))} found, "
        f"{len(report.get('reports_missing', []))} missing"
    )
    print(f"recommended_next_phase={report.get('recommended_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
