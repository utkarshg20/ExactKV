#!/usr/bin/env python3
"""Experiment 065: idle-GPU vLLM object-level KV/cache probe (Phase 15E).

Requires an idle GPU (no auto-started vLLM server). Does **not** implement
ExactKV↔vLLM integration.
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

from exactkv.integrations.vllm_probe import DEFAULT_SMOKE_MODEL  # noqa: E402
from exactkv.integrations.vllm_kv_visibility import (  # noqa: E402
    DEFAULT_EXP065_REPORT,
    recommend_next_step_after_idle_probe,
    run_idle_vllm_object_kv_probe,
    validate_exp065_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 065 idle-GPU vLLM object KV probe")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP065_REPORT)
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL)
    parser.add_argument(
        "--environment-note",
        default=os.environ.get(
            "VLLM_ENVIRONMENT_LABEL",
            "vLLM CUDA-13 idle GPU pod (no auto-started serve)",
        ),
    )
    parser.add_argument(
        "--stop-server",
        action="store_true",
        help="Stop likely vLLM/OpenAI server before probe (only if stop succeeds)",
    )
    args = parser.parse_args()

    result = run_idle_vllm_object_kv_probe(
        environment_note=args.environment_note,
        stop_server=args.stop_server,
        smoke_model_id=args.model,
    )
    report = result.to_report_dict()
    report["recommended_next_step"] = recommend_next_step_after_idle_probe(
        status=str(report["status"]),
        kv_status=str(report["kv_cache_visibility_status"]),
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp065_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 065: {report['status']} kv={report['kv_cache_visibility_status']} "
        f"raw_kv={report['raw_kv_export_status']} vllm={report['vllm_version']}"
    )
    print(f"  python={report['python_executable']} torch={report['torch_version']}")
    print(f"  gpu_before={report['gpu_memory_before']}")
    print(f"  gpu_after={report['gpu_memory_after']}")
    print(
        f"  server_running={report['running_server_detected']} "
        f"stopped={report['stopped_processes']}"
    )
    print(
        f"  llm_init={report['llm_object_initialized']} "
        f"smoke={report['generation_smoke_passed']} "
        f"attempted={report['generation_smoke_attempted']}"
    )
    if report.get("generated_text_preview"):
        print(f"  preview={report['generated_text_preview']!r}")
    print(f"  engine_attrs={report['visible_engine_attrs'][:8]}")
    print(f"  executor_attrs={report['visible_model_executor_attrs'][:8]}")
    print(f"  cache_attrs={report['visible_cache_attrs'][:8]}")
    print(f"  cache_config={report['cache_config_summary']}")
    print(f"  adapter_path={report['possible_adapter_path']}")
    if report["blockers"]:
        print(f"  blockers={report['blockers']}")
    print(f"  next={report['recommended_next_step']}")
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
