#!/usr/bin/env python3
"""Experiment 090: integration safety spec (Phase 18A).

Defines machine-readable safety contract for future L3/L4 integration work.
Specification only — no runtime changes.
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

from exactkv.safety.integration_safety_spec import (  # noqa: E402
    DEFAULT_EXP090_REPORT,
    EXPERIMENT_090_ID,
    run_exp090_integration_safety_spec,
    validate_exp090_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 090: integration safety spec (Phase 18A)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP090_REPORT)
    args = parser.parse_args()

    report = run_exp090_integration_safety_spec()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp090_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"recommended_next={report.get('recommended_next_phase')}")
    summary = report.get("proposal_validator_summary", {})
    print(
        f"synthetic_pass={summary.get('all_passing_accepted')} "
        f"synthetic_fail_rejected={summary.get('all_failing_rejected')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
