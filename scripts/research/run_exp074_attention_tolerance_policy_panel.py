#!/usr/bin/env python3
"""Experiment 074: offline attention tolerance policy panel (Phase 16I)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.tolerance_policy import (  # noqa: E402
    DEFAULT_EXP074_REPORT,
    DEFAULT_REPORT_PATHS,
    AttentionTolerancePolicy,
    run_exp074_panel,
    validate_exp074_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 074 attention tolerance policy panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP074_REPORT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--include-optional-models", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-existing-reports", action="store_true")
    args = parser.parse_args()

    report_paths = None
    if args.skip_existing_reports:
        report_paths = {k: v for k, v in DEFAULT_REPORT_PATHS.items() if v.is_file()}

    report = run_exp074_panel(
        report_paths=report_paths,
        policy=AttentionTolerancePolicy(dtype_name=args.dtype),
        include_optional_models=args.include_optional_models,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp074_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 074: {report['status']} loaded={report['reports_loaded']} "
        f"missing={report['reports_missing']} evaluated={report['total_cells_evaluated']}"
    )
    print(
        f"  strict_pass={report['strict_numeric_pass_cells']} "
        f"depth_aware_pass={report['strict_fail_depth_aware_pass_cells']} "
        f"fr_accum={report['local_alignment_pass_free_running_accumulation_cells']} "
        f"topk_drift={report['topk_agrees_numeric_drift_present_cells']}"
    )
    if args.include_optional_models:
        print(
            f"  optional_loaded={report['optional_models_loaded']} "
            f"optional_blocked={len(report['optional_models_blocked'])}"
        )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "diagnostic_complete", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
