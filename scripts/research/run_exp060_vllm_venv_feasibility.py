#!/usr/bin/env python3
"""Experiment 060: isolated vLLM venv feasibility (Phase 15B).

Uses system Python to orchestrate probe in `/workspace/ExactKV/.venv-vllm` only.
Does **not** install vLLM into system Python.
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

import transformers

from exactkv.integrations.vllm_probe import (  # noqa: E402
    DEFAULT_EXP060_REPORT,
    DEFAULT_SMOKE_MODEL,
    DEFAULT_VLLM_VENV_PYTHON,
    SYSTEM_PYTHON,
    run_vllm_venv_feasibility,
    validate_exp060_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 060 vLLM venv feasibility")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP060_REPORT)
    parser.add_argument("--venv-python", type=Path, default=DEFAULT_VLLM_VENV_PYTHON)
    parser.add_argument("--system-python", type=Path, default=SYSTEM_PYTHON)
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL)
    parser.add_argument(
        "--skip-generation-smoke",
        action="store_true",
        help="Skip tiny vLLM generation smoke in venv",
    )
    args = parser.parse_args()

    result = run_vllm_venv_feasibility(
        venv_python=args.venv_python,
        system_python=args.system_python,
        run_generation_smoke=not args.skip_generation_smoke,
        smoke_model_id=args.model,
    )
    report = result.to_report_dict()
    report["transformers_version"] = transformers.__version__
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp060_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 060: {report['status']} vllm_importable={report['vllm_importable']} "
        f"venv_cuda={report['venv_cuda_available']}"
    )
    print(f"  system_torch={report['system_torch_version']} venv_torch={report['venv_torch_version']}")
    print(f"  install_success={report['install_success']} vllm_version={report['vllm_version']}")
    print(
        f"  generation_smoke={report['generation_smoke_passed']} "
        f"attempted={report['generation_smoke_attempted']}"
    )
    if report["blockers"]:
        print(f"  blockers={report['blockers']}")
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
