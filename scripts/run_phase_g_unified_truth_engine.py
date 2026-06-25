#!/usr/bin/env python3
"""Phase G unified truth + divergence authority engine runner.

Consumes Phase A, D, F, and leaderboard JSON from disk. No inference.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.engine.unified_truth_engine import (  # noqa: E402
    DEFAULT_KERNEL_CONSISTENCY_REPORT,
    DEFAULT_LEADERBOARD_INPUT,
    DEFAULT_PHASE_A_INPUT,
    DEFAULT_PHASE_D_INPUT,
    DEFAULT_PHASE_F_INPUT,
    DEFAULT_PHASE_G_TRUTH_REPORT,
    DEFAULT_UNIFIED_DIVERGENCE_MAP,
    PHASE_G_ID,
    run_phase_g_unified_truth_engine,
    validate_phase_g_report,
    write_phase_g_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase G unified truth authority engine")
    parser.add_argument("--phase-a-input", type=Path, default=DEFAULT_PHASE_A_INPUT)
    parser.add_argument("--phase-d-input", type=Path, default=DEFAULT_PHASE_D_INPUT)
    parser.add_argument("--phase-f-input", type=Path, default=DEFAULT_PHASE_F_INPUT)
    parser.add_argument("--leaderboard-input", type=Path, default=DEFAULT_LEADERBOARD_INPUT)
    parser.add_argument("--output-truth", type=Path, default=DEFAULT_PHASE_G_TRUTH_REPORT)
    parser.add_argument("--output-kernel", type=Path, default=DEFAULT_KERNEL_CONSISTENCY_REPORT)
    parser.add_argument("--divergence-map", type=Path, default=DEFAULT_UNIFIED_DIVERGENCE_MAP)
    args = parser.parse_args()

    report = run_phase_g_unified_truth_engine(
        phase_a_path=args.phase_a_input,
        phase_d_path=args.phase_d_input,
        phase_f_path=args.phase_f_input,
        leaderboard_path=args.leaderboard_input,
        divergence_map_path=args.divergence_map,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    paths = write_phase_g_outputs(
        report,
        truth_path=args.output_truth,
        kernel_path=args.output_kernel,
    )

    validation = validate_phase_g_report(report)
    print(f"phase_id={PHASE_G_ID}")
    print(f"status={report['status']}")
    print(f"unified_records={report['source_totals']['unified_records']}")
    print(f"kernel_consistent={report['kernel_consistency']['overall_consistent']}")
    print(f"failure_regimes={report['failure_regime_counts']}")
    for name, path in paths.items():
        print(f"wrote_{name}={path}")
    print(f"wrote_divergence_map={args.divergence_map}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
