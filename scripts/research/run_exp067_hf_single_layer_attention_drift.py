#!/usr/bin/env python3
"""Experiment 067: HF single-layer attention-drift probe (Phase 16B).

Offline Q/K/V extraction from a real HF layer; not ExactKV generation integration.
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

from exactkv.attention.hf_single_layer_probe import (  # noqa: E402
    DEFAULT_EXP067_REPORT,
    DEFAULT_MODEL_ID,
    run_exp067_probe,
    validate_exp067_report,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 067 HF single-layer attention drift")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP067_REPORT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--layers", default="", help="Comma-separated layer indices")
    parser.add_argument("--chunk-sizes", default="16,32,64")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--projection-only-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    layers = _parse_int_list(args.layers) if args.layers else None
    chunk_sizes = _parse_int_list(args.chunk_sizes)

    report = run_exp067_probe(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        layer_indices=layers,
        chunk_sizes=chunk_sizes,
        max_prompts=args.max_prompts,
        local_files_only=args.local_files_only,
        allow_projection_only=args.projection_only_ok,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp067_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 067: {report['status']} model_load={report.get('model_load_succeeded')} "
        f"cells={report['total_cells']} success={report['successful_cells']} "
        f"blocked={report['blocked_cells']} stream_pass={report['streaming_vs_materialized_pass_cells']}"
    )
    print(f"  max_streaming_vs_materialized={report['max_streaming_vs_materialized_error']:.6g}")
    drift = report["full_vs_streaming_drift_summary"]
    print(
        f"  full_vs_streaming max={drift['max_abs_error']:.6g} "
        f"mean={drift['mean_abs_error']:.6g}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
