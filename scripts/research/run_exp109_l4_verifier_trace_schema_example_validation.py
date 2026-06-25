#!/usr/bin/env python3
"""Experiment 109: L4 verifier trace schema example validation (Phase 21H).

Schema-example validation and trace-only execution — no runtime instrumentation.
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

from exactkv.safety.l4_verifier_trace_schema_example_validation import (  # noqa: E402
    DEFAULT_EXP109_REPORT,
    EXPERIMENT_109_ID,
    run_exp109_l4_verifier_trace_schema_example_validation,
    validate_exp109_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 109: L4 verifier trace schema example validation",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP109_REPORT)
    args = parser.parse_args()

    report = run_exp109_l4_verifier_trace_schema_example_validation()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp109_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"validation_outcome={report.get('validation_outcome')}")
    print(f"examples_passed={report.get('examples_passed')}/{report.get('total_examples')}")
    print(f"schema_coverage={report.get('schema_correctness_coverage')}")
    print(f"allowed_next_phase={report.get('allowed_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
