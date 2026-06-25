#!/usr/bin/env python3
"""Experiment 110: L4 trace schema adversarial injection panel (Phase 21I)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.safety.l4_trace_schema_adversarial_injection_panel import (  # noqa: E402
    DEFAULT_EXP110_REPORT,
    EXPERIMENT_110_ID,
    run_exp110_l4_trace_schema_adversarial_injection_panel,
    validate_exp110_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 110: L4 trace schema adversarial injection panel",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP110_REPORT)
    args = parser.parse_args()

    report = run_exp110_l4_trace_schema_adversarial_injection_panel()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp110_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"panel_outcome={report.get('panel_outcome')}")
    print(f"cases_passed={report.get('cases_passed')}/{report.get('total_cases')}")
    print(f"false_acceptance_rate={report.get('false_acceptance_rate')}")
    print(f"schema_robustness_score={report.get('schema_robustness_score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
