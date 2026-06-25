#!/usr/bin/env python3
"""Experiment 112: L4 Stage 3 verifier-mediated dry-run design (Phase 21K)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.safety.l4_stage3_verifier_mediated_dry_run_design import (  # noqa: E402
    DEFAULT_EXP112_REPORT,
    run_exp112_l4_stage3_verifier_mediated_dry_run_design,
    validate_exp112_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 112: L4 Stage 3 verifier-mediated dry-run design",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP112_REPORT)
    args = parser.parse_args()

    report = run_exp112_l4_stage3_verifier_mediated_dry_run_design()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp112_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"design_outcome={report.get('design_outcome')}")
    print(f"failure_modes={len(report.get('failure_modes') or [])}")
    print(f"synthetic_tests={len(report.get('synthetic_test_matrix') or [])}")
    print(
        "stage_3_scaffold_authorized="
        f"{report.get('design_decision', {}).get('stage_3_scaffold_authorized')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
