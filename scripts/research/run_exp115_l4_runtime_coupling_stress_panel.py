#!/usr/bin/env python3
"""Experiment 115: L4 runtime coupling stress panel (Phase 21N).

Multi-model trace consistency stress expansion over Exp 114. Trace-only; no commit.
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

from exactkv.safety.l4_runtime_coupling_stress_panel import (  # noqa: E402
    DEFAULT_EXP115_REPORT,
    DEFAULT_STRESS_COMPRESSORS,
    DEFAULT_STRESS_MAX_NEW_TOKENS,
    DEFAULT_STRESS_MODELS,
    EXPERIMENT_115_ID,
    run_exp115_l4_runtime_coupling_stress_panel,
    validate_exp115_stress_panel_report,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 115: L4 runtime coupling stress panel (Phase 21N)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_STRESS_MODELS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument(
        "--max-new-tokens-values",
        default=",".join(str(x) for x in DEFAULT_STRESS_MAX_NEW_TOKENS),
    )
    parser.add_argument(
        "--compressors",
        nargs="+",
        default=list(DEFAULT_STRESS_COMPRESSORS),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--deterministic-mode",
        action="store_true",
        help="Use deterministic synthetic traces (no HF download).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXP115_REPORT)
    args = parser.parse_args()

    report = run_exp115_l4_runtime_coupling_stress_panel(
        models=args.models,
        max_prompts=args.max_prompts,
        compressors=args.compressors,
        max_new_tokens_values=_parse_int_list(args.max_new_tokens_values),
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        deterministic_mode=args.deterministic_mode,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    validation = validate_exp115_stress_panel_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"experiment_id={EXPERIMENT_115_ID}")
    print(f"status={report['status']}")
    print(f"total_cells={report['total_cells']}")
    print(f"verifier_stability_score={report['verifier_stability_score']:.4f}")
    print(f"proposal_instability_rate={report['proposal_instability_rate']:.4f}")
    if not validation.valid:
        print("validation_errors:", validation.errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
