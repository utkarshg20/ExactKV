#!/usr/bin/env python3
"""Experiment 108: L4 verifier evidence trace schema scaffold (Phase 21G).

Schema validation and dry-run conversion only — no runtime instrumentation.
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

from exactkv.safety.l4_verifier_evidence_trace_schema_scaffold import (  # noqa: E402
    DEFAULT_EXP108_REPORT,
    EXPERIMENT_108_ID,
    run_exp108_l4_verifier_evidence_trace_schema_scaffold,
    validate_exp108_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 108: L4 verifier evidence trace schema scaffold",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP108_REPORT)
    args = parser.parse_args()

    report = run_exp108_l4_verifier_evidence_trace_schema_scaffold()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp108_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"scaffold_decision={report.get('scaffold_decision')}")
    print(f"valid_examples={report.get('valid_examples')}/{report.get('total_examples')}")
    print(f"converted_examples={report.get('converted_examples')}")
    print(f"runtime_instrumentation_authorized={report.get('runtime_instrumentation_authorized')}")
    print(f"allowed_next_phase={report.get('allowed_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
