#!/usr/bin/env python3
"""Experiment 114: L4 minimal runtime coupling layer (Phase 21M).

First real inference-driven trace-only verification panel. No L4 commit.
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

from exactkv.safety.l4_runtime_coupling import (  # noqa: E402
    DEFAULT_EXP114_REPORT,
    DEFAULT_MODEL_ID,
    EXPERIMENT_114_ID,
    run_exp114_l4_minimal_runtime_coupling_panel,
    validate_exp114_panel_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 114: L4 minimal runtime coupling layer (Phase 21M)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--compressors",
        nargs="+",
        default=["noop", "int8"],
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP114_REPORT)
    args = parser.parse_args()

    report = run_exp114_l4_minimal_runtime_coupling_panel(
        model_name=args.model_id,
        max_prompts=args.max_prompts,
        compressors=args.compressors,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    validation = validate_exp114_panel_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"experiment_id={EXPERIMENT_114_ID}")
    print(f"status={report['status']}")
    print(f"panel_outcome={report['panel_outcome']}")
    print(f"trace_records_total={report['trace_records_total']}")
    print(f"model_experiments_run={report['model_experiments_run']}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
