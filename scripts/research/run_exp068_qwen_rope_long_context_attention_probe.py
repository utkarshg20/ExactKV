#!/usr/bin/env python3
"""Experiment 068: Qwen RoPE/GQA long-context single-layer attention probe (Phase 16C).

Offline HF Q/K/V extraction with RoPE and long-context chunking — not ExactKV generation.
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
    DEFAULT_EXP068_REPORT,
    DEFAULT_MODEL_ID,
    run_exp068_probe,
    validate_exp068_report,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exp 068 Qwen RoPE/GQA long-context single-layer attention probe"
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP068_REPORT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-token-lengths", default="64,128,256")
    parser.add_argument("--layers", default="", help="Comma-separated layer indices")
    parser.add_argument("--chunk-sizes", default="16,32,64")
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--projection-only-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attempt-layer-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    layers = _parse_int_list(args.layers) if args.layers else None
    target_lengths = _parse_int_list(args.target_token_lengths)
    chunk_sizes = _parse_int_list(args.chunk_sizes)

    report = run_exp068_probe(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        target_token_lengths=target_lengths,
        layer_indices=layers,
        chunk_sizes=chunk_sizes,
        max_prompts=args.max_prompts,
        local_files_only=args.local_files_only,
        allow_projection_only=args.projection_only_ok,
        attempt_layer_parity=args.attempt_layer_parity,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp068_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 068: {report['status']} model_load={report.get('model_load_succeeded')} "
        f"cells={report['total_cells']} success={report['successful_cells']} "
        f"blocked={report['blocked_cells']} stream_pass={report['streaming_vs_materialized_pass_cells']}"
    )
    print(f"  extraction_modes={report.get('extraction_mode_counts')}")
    print(f"  rope_status={report.get('rope_status_counts')}")
    print(f"  gqa_status={report.get('gqa_status_counts')}")
    print(f"  longest_context={report.get('longest_context_tested')} max_chunks={report.get('max_num_chunks')}")
    print(f"  max_streaming_vs_materialized={report['max_streaming_vs_materialized_error']:.6g}")
    print(f"  best_theoretical_reduction={report.get('best_theoretical_streaming_reduction')}")
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
