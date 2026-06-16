#!/usr/bin/env python3
"""Experiment 071: offline full-prefix logit drift smoke (Phase 16F)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.attention.hf_full_replay_probe import (  # noqa: E402
    DEFAULT_EXP071_REPORT,
    DEFAULT_MODEL_ID,
    run_exp071_probe,
    validate_exp071_report,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 071 full-prefix logit drift smoke")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP071_REPORT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-token-lengths", default="32,64,128")
    parser.add_argument("--chunk-sizes", default="16,32,64")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--accumulator-mode", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-parity-fail",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    report = run_exp071_probe(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        target_token_lengths=_parse_int_list(args.target_token_lengths),
        chunk_sizes=_parse_int_list(args.chunk_sizes),
        max_prompts=args.max_prompts,
        accumulator_mode=args.accumulator_mode,
        local_files_only=args.local_files_only,
        allow_parity_fail=args.allow_parity_fail,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp071_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 071: {report['status']} model_load={report.get('model_load_succeeded')} "
        f"cells={report['total_cells']} success={report['successful_cells']} "
        f"blocked={report['blocked_cells']} parity_pass={report['full_model_parity_pass_cells']} "
        f"stream_pass={report['streaming_vs_materialized_pass_cells']} "
        f"top1_changed={report['compressed_top1_changed_cells']}"
    )
    print(
        f"  max_sm_logit={report['max_streaming_vs_materialized_logit_error']:.6g} "
        f"max_sm_hidden={report['max_streaming_vs_materialized_hidden_error']:.6g}"
    )
    print(
        f"  layers={report.get('num_layers_replayed')} "
        f"longest_context={report.get('longest_context_tested')}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
