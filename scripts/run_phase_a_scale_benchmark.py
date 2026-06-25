#!/usr/bin/env python3
"""Phase A scale benchmarking runner — multi-model KV compression truth layer.

Generates reports/phaseA_benchmark.json and reports/phaseA_benchmark.md.
No ExactKVGenerator modifications. Trace-only evaluation.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.benchmarks.phase_a_scale_benchmark import (  # noqa: E402
    DEFAULT_PHASE_A_MARKDOWN,
    DEFAULT_PHASE_A_REPORT,
    PHASE_A_ID,
    run_phase_a_scale_benchmark,
    validate_phase_a_report,
    write_phase_a_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase A scale benchmarking runner")
    parser.add_argument(
        "--deterministic-mode",
        action="store_true",
        help="Hash-seeded synthetic cells (no GPU / no HF download)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (e.g. cpu, cuda)")
    parser.add_argument("--dtype", default="float32", help="Torch dtype")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached HF weights",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_PHASE_A_REPORT,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_PHASE_A_MARKDOWN,
    )
    parser.add_argument(
        "--exp116-input",
        type=Path,
        default=None,
        help="Optional Exp 116 report for instability score integration",
    )
    args = parser.parse_args()

    report = run_phase_a_scale_benchmark(
        deterministic_mode=args.deterministic_mode,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        exp116_path=args.exp116_input or DEFAULT_PHASE_A_REPORT.parent / "experiment_116_instability_regime_analysis.json",
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    json_path, md_path = write_phase_a_outputs(
        report,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )

    validation = validate_phase_a_report(report)
    print(f"phase_id={PHASE_A_ID}")
    print(f"status={report['status']}")
    print(f"models_evaluated={len(report.get('models_evaluated') or [])}")
    print(f"total_cells={report.get('total_cells')}")
    print(f"exactkv_failures={report.get('exactkv_failures')}")
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
