#!/usr/bin/env python3
"""Experiment 059: vLLM feasibility probe (Phase 15A).

Install-safe environment probe only. Does **not** install vLLM or modify torch.
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
    DEFAULT_EXP059_REPORT,
    DEFAULT_SMOKE_MODEL,
    probe_vllm_availability,
    validate_exp059_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 059 vLLM feasibility probe")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP059_REPORT)
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL)
    parser.add_argument(
        "--skip-generation-smoke",
        action="store_true",
        help="Skip optional vLLM generation smoke even if importable",
    )
    args = parser.parse_args()

    result = probe_vllm_availability(
        run_generation_smoke=not args.skip_generation_smoke,
        smoke_model_id=args.model,
    )
    report = result.to_report_dict()
    report["transformers_version"] = transformers.__version__
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp059_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 059: {report['status']} vllm_importable={report['vllm_importable']} "
        f"cuda={report['cuda_available']}"
    )
    if report["vllm_importable"]:
        print(f"  vllm_version={report['vllm_version']}")
        print(f"  LLM={report['llm_class_importable']} SamplingParams={report['sampling_params_importable']}")
        print(f"  generation_smoke={report['generation_smoke_passed']} attempted={report['generation_smoke_attempted']}")
    else:
        print(f"  import_error={report['import_error']}")
    if report["blockers"]:
        print(f"  blockers={report['blockers']}")
    print(f"  kv_cache_access_status={report['kv_cache_access_status']}")
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
