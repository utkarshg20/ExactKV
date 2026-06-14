#!/usr/bin/env python3
"""Experiment 056: CUDA restored-verifier runtime gate (Phase 14A).

CUDA float16/bfloat16 exactness via explicit ``run_experimental_restored_verifier()``.
**Not** default runtime. Requires CUDA for real evidence.
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

from exactkv.cache.offline_verifier import DEFAULT_MODEL  # noqa: E402
from exactkv.cache.restored_verifier_runner import DEFAULT_SMOKE_PROMPT_IDS  # noqa: E402
from exactkv.runtime.experimental import (  # noqa: E402
    DEFAULT_CUDA_GATE_DRAFT_LEN,
    DEFAULT_CUDA_GATE_MAX_NEW_TOKENS,
    EXPERIMENT_056_ID,
    report_to_exp056_json,
    run_cuda_restored_verifier_runtime_gate,
    validate_exp056_report,
)

DEFAULT_JSON = _ROOT / "reports" / "experiment_056_cuda_restored_verifier_runtime_gate.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 056 CUDA restored-verifier runtime gate")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_CUDA_GATE_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_CUDA_GATE_DRAFT_LEN)
    parser.add_argument(
        "--prompt-ids",
        default=",".join(DEFAULT_SMOKE_PROMPT_IDS),
        help="Comma-separated prompt ids",
    )
    args = parser.parse_args()

    prompt_ids = [x.strip() for x in args.prompt_ids.split(",") if x.strip()]
    result = run_cuda_restored_verifier_runtime_gate(
        model_id=args.model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
    )
    json_report = report_to_exp056_json(
        result,
        model_id=args.model,
        prompt_count=len(prompt_ids),
    )
    json_report["transformers_version"] = transformers.__version__
    json_report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp056_report(json_report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    if not result.cuda_available:
        print(f"Exp 056: blocked — CUDA unavailable (cuda={torch.cuda.is_available()})")
    else:
        print(
            f"Exp 056: {json_report['status']} dtypes={json_report['dtype_configs']} "
            f"exact={result.token_exact_match_count}/{result.total_cells} "
            f"draft_div={result.draft_divergence_count}"
        )
    print(f"Wrote {args.json_out}")
    return 0 if result.status in ("pass", "blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
