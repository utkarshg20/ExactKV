#!/usr/bin/env python3
"""Experiment 099: L4 verifier-mediated compressed draft design spec (Phase 20B).

Design specification only — no L4 runtime implementation.
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

from exactkv.safety.l4_verifier_mediated_design_spec import (  # noqa: E402
    DEFAULT_EXP099_REPORT,
    EXPERIMENT_099_ID,
    run_exp099_l4_verifier_mediated_design_spec,
    validate_exp099_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 099: L4 verifier-mediated design specification",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP099_REPORT)
    args = parser.parse_args()

    report = run_exp099_l4_verifier_mediated_design_spec()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp099_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    review = report.get("design_review_result") or {}
    print(f"outcome={review.get('outcome')}")
    print(f"l4_implementation_authorized={report.get('l4_implementation_authorized')}")
    print(f"allowed_next_phase={report.get('allowed_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
