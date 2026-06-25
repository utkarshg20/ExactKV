#!/usr/bin/env python3
"""Experiment 116: L4 instability regime extraction (Phase 21O).

Post-hoc analysis over Exp 115 stress panel report only. No inference.
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

from exactkv.analysis.l4_instability_regime_extractor import (  # noqa: E402
    DEFAULT_EXP115_REPORT,
    DEFAULT_EXP116_REPORT,
    EXPERIMENT_116_ID,
    run_exp116_instability_regime_extraction,
    validate_exp116_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 116: instability regime extraction (Phase 21O)",
    )
    parser.add_argument(
        "--exp115-input",
        type=Path,
        default=DEFAULT_EXP115_REPORT,
        help="Path to Exp 115 stress panel JSON report.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP116_REPORT)
    args = parser.parse_args()

    report = run_exp116_instability_regime_extraction(exp115_path=args.exp115_input)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    validation = validate_exp116_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"experiment_id={EXPERIMENT_116_ID}")
    print(f"status={report['status']}")
    print(f"source_total_cells={report['source_total_cells']}")
    print(f"regime_coverage={report['regime_coverage']}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
