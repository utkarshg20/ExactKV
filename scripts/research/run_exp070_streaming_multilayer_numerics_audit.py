#!/usr/bin/env python3
"""Experiment 070: streaming multi-layer numerical boundary audit (Phase 16E)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.hf_multilayer_probe import (  # noqa: E402
    DEFAULT_EXP070_REPORT,
    DEFAULT_MODEL_ID,
    run_exp070_probe,
    validate_exp070_report,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exp 070 streaming multi-layer numerics audit"
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP070_REPORT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-token-lengths", default="64,128,256")
    parser.add_argument("--prefix-layer-counts", default="1,2,4")
    parser.add_argument("--chunk-sizes", default="8,16,32,64")
    parser.add_argument("--accumulator-modes", default="default,float32,float64")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    report = run_exp070_probe(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        target_token_lengths=_parse_int_list(args.target_token_lengths),
        prefix_layer_counts=_parse_int_list(args.prefix_layer_counts),
        chunk_sizes=_parse_int_list(args.chunk_sizes),
        accumulator_modes=_parse_str_list(args.accumulator_modes),
        max_prompts=args.max_prompts,
        local_files_only=args.local_files_only,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp070_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 070: {report['status']} model_load={report.get('model_load_succeeded')} "
        f"cells={report['total_cells']} success={report['successful_cells']} "
        f"strict_fail={report['failed_cells_under_strict_tolerance']} "
        f"rec_fail={report['failed_cells_under_recommended_tolerance']}"
    )
    print(
        f"  phase16d_status={report['phase16d_failure_status_after_audit']} "
        f"reproduced={report['phase16d_failure_reproduced']}"
    )
    print(
        f"  policy={report['tolerance_policy_recommendation'].get('policy')} "
        f"algorithm_change={report['algorithm_change_made']}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "pass_with_recommended_tolerance", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
