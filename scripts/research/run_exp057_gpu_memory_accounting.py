#!/usr/bin/env python3
"""Experiment 057: GPU memory accounting diagnostic (Phase 14B).

Diagnostic CUDA memory observations for explicit experimental restored-verifier
runtime. Requires Exp 056 CUDA exactness gate. **Not** a memory savings claim.
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

import torch
import transformers

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL  # noqa: E402
from exactkv.cache.restored_verifier_runner import DEFAULT_SMOKE_PROMPT_IDS  # noqa: E402
from exactkv.metrics.gpu_memory_accounting import (  # noqa: E402
    DEFAULT_EXP056_REPORT,
    DEFAULT_EXP057_REPORT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MEMORY_PROMPT_IDS,
    check_exp056_exactness_gate,
    run_gpu_memory_accounting,
    validate_exp057_report,
)
from exactkv.runtime.experimental import (  # noqa: E402
    report_to_exp056_json,
    run_cuda_restored_verifier_runtime_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 057 GPU memory accounting diagnostic")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP057_REPORT)
    parser.add_argument("--exp056-report", type=Path, default=DEFAULT_EXP056_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument(
        "--prompt-ids",
        default=",".join(DEFAULT_MEMORY_PROMPT_IDS),
        help="Comma-separated prompt ids (2–4 recommended)",
    )
    parser.add_argument(
        "--run-exp056-if-missing",
        action="store_true",
        help="Run Exp 056 gate when report is missing and CUDA is available",
    )
    args = parser.parse_args()

    exp056_ok, exp056_blockers = check_exp056_exactness_gate(args.exp056_report)
    if not exp056_ok and args.run_exp056_if_missing and torch.cuda.is_available():
        if any("not found" in b.lower() for b in exp056_blockers):
            print("Exp 056 report missing — running CUDA gate first...")
            gate = run_cuda_restored_verifier_runtime_gate(model_id=args.model)
            gate_report = report_to_exp056_json(gate, model_id=args.model)
            args.exp056_report.parent.mkdir(parents=True, exist_ok=True)
            args.exp056_report.write_text(
                json.dumps(gate_report, indent=2), encoding="utf-8"
            )
            if gate.status != "pass" or gate.exactkv_failures > 0:
                print("Exp 056 gate failed — refusing Exp 057")
                return 1
            exp056_ok, exp056_blockers = check_exp056_exactness_gate(args.exp056_report)

    if not exp056_ok:
        print("Exp 057 blocked — Exp 056 exactness gate not passed:")
        for b in exp056_blockers:
            print(f"  - {b}")
        blocked = run_gpu_memory_accounting(
            model_id=args.model,
            exp056_report_path=args.exp056_report,
            require_exp056_gate=True,
        )
        report = blocked.to_dict()
        report["transformers_version"] = transformers.__version__
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    prompt_ids = [x.strip() for x in args.prompt_ids.split(",") if x.strip()]
    result = run_gpu_memory_accounting(
        model_id=args.model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
        exp056_report_path=args.exp056_report,
    )
    report = result.to_dict()
    report["transformers_version"] = transformers.__version__
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp057_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 057: {report['status']} dtypes={report['dtype_configs']} "
        f"exact={result.token_exact_match_count}/{result.total_cells} "
        f"measurements={len(result.measurements)}"
    )
    for label in (
        "model_loaded",
        "full_greedy",
        "kv_capture_store_reload",
        "restored_verifier_runtime",
    ):
        entry = report.get(label)
        if entry:
            print(
                f"  {label}: peak_alloc={entry['peak_allocated_bytes']} "
                f"peak_reserved={entry['peak_reserved_bytes']}"
            )
    print(f"  full_kv_payload_bytes={result.full_kv_payload_bytes}")
    print(f"  stored_kv_payload_bytes={result.stored_kv_payload_bytes}")
    print(f"Wrote {args.json_out}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
