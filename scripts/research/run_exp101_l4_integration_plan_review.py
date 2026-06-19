#!/usr/bin/env python3
"""Experiment 101: L4 integration plan review (Phase 20D).

Integration plan review only — no L4 runtime implementation.
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

from exactkv.safety.l4_integration_plan_review import (  # noqa: E402
    DEFAULT_EXP101_REPORT,
    EXPERIMENT_101_ID,
    run_exp101_l4_integration_plan_review,
    validate_exp101_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 101: L4 integration plan review",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP101_REPORT)
    args = parser.parse_args()

    report = run_exp101_l4_integration_plan_review()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp101_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    decision = report.get("integration_plan_decision") or {}
    print(f"decision={decision.get('decision')}")
    print(f"l4_runtime_commit_authorized={report.get('l4_runtime_commit_authorized')}")
    print(f"allowed_next_phase={report.get('allowed_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
