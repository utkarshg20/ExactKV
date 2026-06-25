#!/usr/bin/env python3
"""Experiment 107: L4 verifier evidence trace schema design (Phase 21F).

Schema design specification only — no runtime instrumentation.
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

from exactkv.safety.l4_verifier_evidence_trace_schema_design import (  # noqa: E402
    DEFAULT_EXP107_REPORT,
    EXPERIMENT_107_ID,
    run_exp107_l4_verifier_evidence_trace_schema_design,
    validate_exp107_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 107: L4 verifier evidence trace schema design",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP107_REPORT)
    args = parser.parse_args()

    report = run_exp107_l4_verifier_evidence_trace_schema_design()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp107_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    decision = report.get("design_decision") or {}
    print(f"outcome={decision.get('outcome')}")
    print(f"runtime_instrumentation_authorized={report.get('runtime_instrumentation_authorized')}")
    print(f"runtime_commit_authorized={report.get('runtime_commit_authorized')}")
    print(f"allowed_next_phase={report.get('allowed_next_phase')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
