#!/usr/bin/env python3
"""Experiment 105: L4 trace-only dry-run scaffold (Phase 21D).

Diagnostic-only trace evaluator — not L4 runtime implementation.
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
    DEFAULT_EXP105_REPORT,
    EXPERIMENT_105_ID,
    run_exp105_l4_trace_only_dry_run_scaffold,
    validate_exp105_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 105: L4 trace-only dry-run scaffold",
    )
    parser.add_argument(
        "--try-real-traces",
        action="store_true",
        help="Optionally attempt real trace extraction from local reports.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP105_REPORT)
    args = parser.parse_args()

    report = run_exp105_l4_trace_only_dry_run_scaffold(
        try_real_traces=args.try_real_traces,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp105_report(report)
    if errors and report.get("status") == "scaffold_complete":
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"total_decisions={report.get('total_decisions')}")
    synth = report.get("synthetic_suite_summary") or {}
    print(f"synthetic_decisions={synth.get('total_decisions')}")
    real = report.get("real_trace_mode_summary") or {}
    print(f"real_trace_mode={real.get('status')}")
    validation = report.get("validation_result") or {}
    print(f"validation_valid={validation.get('valid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
