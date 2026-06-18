#!/usr/bin/env python3
"""Experiment 089: integration design review (Phase 17D).

Claim-safe review of integration levels, gates, and risks after Phase 17.
Design phase only — no runtime changes or new experiments.
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

from exactkv.demo.integration_design_review import (  # noqa: E402
    DEFAULT_EXP089_REPORT,
    EXPERIMENT_089_ID,
    run_exp089_integration_design_review,
    validate_exp089_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 089: integration design review (Phase 17D)",
    )
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP089_REPORT)
    args = parser.parse_args()

    report = run_exp089_integration_design_review(root=args.root)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp089_report(report)
    if errors:
        print("Report validation errors:", errors, file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"status={report.get('status')}")
    print(f"implemented_level={report.get('current_implemented_level')}")
    print(f"recommended_next={report.get('recommended_next_phase')}")
    print(
        f"docs={len(report.get('source_docs_found', []))} found "
        f"{len(report.get('source_docs_missing', []))} missing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
