#!/usr/bin/env python3
"""Experiment 113: L4 Stage 3 verifier-mediated dry-run scaffold (Phase 21L)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.safety.l4_stage3_verifier_mediated_dry_run_scaffold import (  # noqa: E402
    DEFAULT_EXP113_REPORT,
    run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold,
    validate_exp113_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 113: L4 Stage 3 verifier-mediated dry-run scaffold",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP113_REPORT)
    args = parser.parse_args()

    report = run_exp113_l4_stage3_verifier_mediated_dry_run_scaffold()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp113_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"panel_outcome={report.get('panel_outcome')}")
    print(f"cases_passed={report.get('cases_passed')}/{report.get('total_cases')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
