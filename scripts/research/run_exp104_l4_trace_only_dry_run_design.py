#!/usr/bin/env python3
"""Experiment 104: L4 trace-only dry-run design (Phase 21C).

Stage 2 design specification only — no runtime implementation.
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

from exactkv.safety.l4_trace_only_dry_run_design import (  # noqa: E402
    DEFAULT_EXP104_REPORT,
    EXPERIMENT_104_ID,
    run_exp104_l4_trace_only_dry_run_design,
    validate_exp104_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 104: L4 trace-only dry-run design",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP104_REPORT)
    args = parser.parse_args()

    report = run_exp104_l4_trace_only_dry_run_design()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp104_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    decision = report.get("design_decision") or {}
    print(f"outcome={decision.get('outcome')}")
    print(f"runtime_commit_authorized={report.get('runtime_commit_authorized')}")
    print(f"allowed_next_phase={report.get('allowed_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
