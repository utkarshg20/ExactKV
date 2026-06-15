#!/usr/bin/env python3
"""Experiment 062: vLLM container/CUDA-13 environment feasibility (Phase 15C-env).

Runs in the native Python of a vLLM-compatible image/container.
Does **not** install vLLM or modify ExactKV default runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.integrations.vllm_probe import (  # noqa: E402
    DEFAULT_EXP062_REPORT,
    DEFAULT_SMOKE_MODEL,
    probe_vllm_container_feasibility,
    recommend_next_step_after_container_probe,
    validate_exp062_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 062 vLLM container feasibility")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP062_REPORT)
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL)
    parser.add_argument(
        "--environment-label",
        default=os.environ.get("VLLM_ENVIRONMENT_LABEL", ""),
        help="Human-readable environment label (e.g. RunPod vLLM template happy_blush_scallop)",
    )
    parser.add_argument(
        "--skip-generation-smoke",
        action="store_true",
        help="Skip tiny vLLM generation smoke",
    )
    args = parser.parse_args()

    report = probe_vllm_container_feasibility(
        run_generation_smoke=not args.skip_generation_smoke,
        smoke_model_id=args.model,
        environment_label=args.environment_label,
    )
    report["recommended_next_step"] = recommend_next_step_after_container_probe(
        status=str(report["status"]),
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp062_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 062: {report['status']} vllm={report['vllm_version']} "
        f"cuda={report['cuda_available']}"
    )
    print(f"  python={report['python_executable']}")
    print(f"  torch={report['torch_version']} cuda_runtime={report.get('cuda_runtime', '')}")
    print(
        f"  llm={report['llm_class_importable']} "
        f"sampling={report['sampling_params_importable']}"
    )
    print(
        f"  generation_smoke={report['generation_smoke_passed']} "
        f"attempted={report['generation_smoke_attempted']}"
    )
    if report.get("generated_text_preview"):
        print(f"  preview={report['generated_text_preview']!r}")
    if report["blockers"]:
        print(f"  blockers={report['blockers']}")
    print(f"  next={report['recommended_next_step']}")
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
