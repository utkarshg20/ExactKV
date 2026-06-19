#!/usr/bin/env python3
"""Experiment 098: Pre-L4 safety gate review (Phase 20A).

Reviews local L3 evidence to determine whether L4 design specification may begin.
Does not authorize L4 implementation or run new model experiments.
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

from exactkv.safety.pre_l4_gate_review import (  # noqa: E402
    DEFAULT_EXP098_REPORT,
    EXPERIMENT_098_ID,
    run_exp098_pre_l4_safety_gate_review,
    validate_exp098_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 098: Pre-L4 safety gate review",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root for evidence inventory",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP098_REPORT)
    args = parser.parse_args()

    report = run_exp098_pre_l4_safety_gate_review(root=args.root)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp098_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"review_outcome={report.get('review_outcome')}")
    print(f"l4_design_spec_authorized={report.get('l4_design_spec_authorized')}")
    print(f"l4_implementation_authorized={report.get('l4_implementation_authorized')}")
    gs = report.get("gate_summary") or {}
    print(f"gates_passing={gs.get('gates_passing')}/{gs.get('gates_total')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
