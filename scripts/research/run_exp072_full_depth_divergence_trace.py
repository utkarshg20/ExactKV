#!/usr/bin/env python3
"""Experiment 072: full-depth streaming/materialized divergence trace (Phase 16G)."""
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
    DEFAULT_EXP072_REPORT,
    DEFAULT_MODEL_ID,
    run_exp072_probe,
    validate_exp072_report,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 072 full-depth divergence trace")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP072_REPORT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-token-lengths", default="32,64")
    parser.add_argument("--chunk-sizes", default="16,32,64")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--accumulator-mode", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    report = run_exp072_probe(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        target_token_lengths=_parse_int_list(args.target_token_lengths),
        chunk_sizes=_parse_int_list(args.chunk_sizes),
        max_prompts=args.max_prompts,
        accumulator_mode=args.accumulator_mode,
        local_files_only=args.local_files_only,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp072_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 072: {report['status']} model_load={report.get('model_load_succeeded')} "
        f"cells={report['total_cells']} success={report['successful_cells']} "
        f"blocked={report['blocked_cells']} phase16f_repro={report['phase16f_failure_reproduced']}"
    )
    print(f"  root_causes={report.get('root_cause_counts')}")
    print(
        f"  tf_attn_max={report['teacher_forced_local_error_summary'].get('max_attn_context_error'):.6g} "
        f"fr_post_mlp_max={report['free_running_error_summary'].get('max_post_mlp_error'):.6g}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "diagnostic_complete", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
