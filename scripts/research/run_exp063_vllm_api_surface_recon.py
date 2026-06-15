#!/usr/bin/env python3
"""Experiment 063: vLLM API surface and KV-cache visibility recon (Phase 15C).

Runs in the native Python of a vLLM-compatible image. Does **not** implement
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
from exactkv.integrations.vllm_surface_recon import (  # noqa: E402
    DEFAULT_EXP063_REPORT,
    run_vllm_surface_recon,
    recommend_next_step_after_surface_recon,
    validate_exp063_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 063 vLLM API surface recon")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP063_REPORT)
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL)
    parser.add_argument(
        "--environment-note",
        default=os.environ.get(
            "VLLM_ENVIRONMENT_LABEL",
            "RunPod vLLM template happy_blush_scallop",
        ),
    )
    parser.add_argument(
        "--allow-llm-init",
        action="store_true",
        help="Instantiate LLM for object-level recon even if GPU appears busy",
    )
    parser.add_argument(
        "--stop-template-server",
        action="store_true",
        help="Stop likely vLLM/OpenAI server process before recon (disposable pods only)",
    )
    args = parser.parse_args()

    result = run_vllm_surface_recon(
        environment_note=args.environment_note,
        allow_llm_init=args.allow_llm_init,
        stop_template_server=args.stop_template_server,
        smoke_model_id=args.model,
    )
    report = result.to_report_dict()
    report["recommended_next_step"] = recommend_next_step_after_surface_recon(
        status=str(report["status"]),
        kv_status=str(report["kv_cache_access_status"]),
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp063_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 063: {report['status']} kv={report['kv_cache_access_status']} "
        f"vllm={report['vllm_version']}"
    )
    print(f"  python={report['python_executable']} torch={report['torch_version']}")
    print(f"  gpu_mem={report['gpu_memory_summary']}")
    print(f"  server_running={report['running_server_detected']} stopped={report['stopped_processes']}")
    print(
        f"  llm_import={report['llm_class_importable']} "
        f"llm_init={report['llm_object_initialized']} "
        f"smoke={report['generation_smoke_passed']} attempted={report['generation_smoke_attempted']}"
    )
    print(f"  cache_surfaces={report['visible_cache_surfaces']}")
    print(f"  adapter_path={report['possible_adapter_path']}")
    if report["blockers"]:
        print(f"  blockers={report['blockers']}")
    print(f"  next={report['recommended_next_step']}")
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
