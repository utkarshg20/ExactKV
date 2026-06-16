#!/usr/bin/env python3
"""Experiment 073: Qwen-family offline divergence panel (Phase 16H)."""
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
    DEFAULT_EXP073_REPORT,
    DEFAULT_MODEL_IDS_073,
    OPTIONAL_MODEL_IDS_073,
    run_exp073_probe,
    validate_exp073_report,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 073 Qwen-family divergence panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP073_REPORT)
    parser.add_argument("--model-ids", default=",".join(DEFAULT_MODEL_IDS_073))
    parser.add_argument("--include-optional-models", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-token-lengths", default="32,64")
    parser.add_argument("--chunk-sizes", default="16,64")
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--accumulator-mode", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]
    if args.include_optional_models:
        for mid in OPTIONAL_MODEL_IDS_073:
            if mid not in model_ids:
                model_ids.append(mid)

    report = run_exp073_probe(
        model_ids=model_ids,
        device=args.device,
        dtype=args.dtype,
        target_token_lengths=_parse_int_list(args.target_token_lengths),
        chunk_sizes=_parse_int_list(args.chunk_sizes),
        max_prompts=args.max_prompts,
        accumulator_mode=args.accumulator_mode,
        local_files_only=args.local_files_only,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    errors = validate_exp073_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report.get("panel_summary", {})
    print(
        f"Exp 073: {report['status']} loaded={summary.get('models_loaded')} "
        f"blocked={summary.get('models_blocked')} cells={report['total_cells']} "
        f"success={report['successful_cells']}"
    )
    print(f"  classifications={report.get('model_level_classifications')}")
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "diagnostic_complete", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
