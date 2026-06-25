#!/usr/bin/env python3
"""Experiment 111: L4 verifier runtime instrumentation design (Phase 21J)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.safety.l4_verifier_runtime_instrumentation_design import (  # noqa: E402
    DEFAULT_EXP111_REPORT,
    EXPERIMENT_111_ID,
    run_exp111_l4_verifier_runtime_instrumentation_design,
    validate_exp111_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 111: L4 verifier runtime instrumentation design",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP111_REPORT)
    args = parser.parse_args()

    report = run_exp111_l4_verifier_runtime_instrumentation_design()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp111_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"design_outcome={report.get('design_outcome')}")
    print(f"runtime_hooks={len(report.get('runtime_hooks') or [])}")
    print(f"instrumentation_points={len(report.get('instrumentation_points') or [])}")
    print(
        "stage_3_dry_run_design_authorized="
        f"{report.get('design_decision', {}).get('stage_3_dry_run_design_authorized')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
