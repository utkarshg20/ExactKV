#!/usr/bin/env python3
"""Experiment 100: L4 contract tests with no runtime integration (Phase 20C).

Pure synthetic contract evaluation only — no L4 runtime implementation.
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

from exactkv.safety.l4_contract_tests_no_runtime import (  # noqa: E402
    DEFAULT_EXP100_REPORT,
    EXPERIMENT_100_ID,
    run_exp100_l4_contract_tests_no_runtime,
    validate_exp100_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 100: L4 contract tests (no runtime)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP100_REPORT)
    args = parser.parse_args()

    report = run_exp100_l4_contract_tests_no_runtime()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp100_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    suite = report.get("suite_summary") or {}
    print(f"suite_status={suite.get('suite_status')}")
    print(f"passing_cases={suite.get('passing_cases')}/{suite.get('total_cases')}")
    print(f"allowed_next_phase={report.get('allowed_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
