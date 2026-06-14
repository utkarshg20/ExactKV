#!/usr/bin/env python3
"""Experiment 052: Restored-verifier runner smoke (Phase 12G).

Small smoke using the consolidated isolated runner API.
**Not** wired into ``ExactKVGenerator`` defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import transformers

from exactkv.cache.restored_verifier_runner import (  # noqa: E402
    DEFAULT_SMOKE_COMPRESSORS,
    DEFAULT_SMOKE_DRAFT_LEN,
    DEFAULT_SMOKE_MAX_NEW_TOKENS,
    DEFAULT_SMOKE_PROMPT_IDS,
    EXPERIMENT_052_ID,
    EXP052_CLAIM_NOTE,
    check_phase12f_exactness_gate,
    default_smoke_config,
    report_to_exp052_json,
    run_restored_verifier,
    validate_exp052_report,
    validate_exactness_gate,
)
from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS, DEFAULT_MODEL  # noqa: E402
from exactkv.cache.offline_verifier import VERIFIER_SOURCE  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_052_restored_verifier_runner_smoke.json"


def _blocked_report(reason: str, *, gate_reason: str = "") -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_052_ID,
        "status": "blocked",
        "model": DEFAULT_MODEL,
        "device": "unknown",
        "dtype": "unknown",
        "transformers_version": transformers.__version__,
        "prompt_count": 0,
        "storage_backend": "none",
        "storage_backends": [],
        "compressor_names": list(DEFAULT_SMOKE_COMPRESSORS),
        "draft_len": DEFAULT_SMOKE_DRAFT_LEN,
        "max_new_tokens": DEFAULT_SMOKE_MAX_NEW_TOKENS,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [],
        "total_cells": 0,
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "mean_acceptance": 0.0,
        "draft_divergence_count": 0,
        "first_divergences": [],
        "restore_blockers": [reason],
        "draft_blockers": [],
        "verification_blockers": [],
        "blockers": {
            "restore_blockers": [reason],
            "draft_blockers": [],
            "verification_blockers": [],
        },
        "phase12f_gate": gate_reason,
        "claim_note": EXP052_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 052 restored-verifier runner smoke")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_SMOKE_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_SMOKE_DRAFT_LEN)
    parser.add_argument(
        "--prompt-ids",
        default=",".join(DEFAULT_SMOKE_PROMPT_IDS),
        help="Comma-separated prompt ids",
    )
    parser.add_argument(
        "--compressors",
        default=",".join(DEFAULT_SMOKE_COMPRESSORS),
        help="Comma-separated compressor names",
    )
    parser.add_argument("--include-file-backend", action="store_true")
    parser.add_argument("--tmpdir", type=Path, default=_ROOT / "reports" / "exp052_kv_files")
    args = parser.parse_args()

    allowed, gate_reason = check_phase12f_exactness_gate(
        _ROOT / "reports" / "experiment_051_offline_verifier_cuda_drift_panel.json"
    )
    if not allowed:
        report = _blocked_report(
            f"Phase 12G blocked: {gate_reason}",
            gate_reason=gate_reason,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Exp 052: blocked — {gate_reason}")
        return 1

    prompt_ids = [x.strip() for x in args.prompt_ids.split(",") if x.strip()]
    compressors = [x.strip() for x in args.compressors.split(",") if x.strip()]
    config = default_smoke_config(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        prompt_ids=prompt_ids,
        compressor_names=compressors,
        draft_len=args.draft_len,
        max_new_tokens=args.max_new_tokens,
        file_storage_root=str(args.tmpdir),
    )

    extra_backends = ["file_kv_storage"] if args.include_file_backend else None

    try:
        report = run_restored_verifier(config, extra_backends=extra_backends)
    except Exception as exc:  # noqa: BLE001
        blocked = _blocked_report(f"runner failed: {type(exc).__name__}: {exc}", gate_reason=gate_reason)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        print(f"Exp 052: blocked — {exc}")
        return 1

    json_report = report_to_exp052_json(report)
    json_report["transformers_version"] = transformers.__version__
    json_report["phase12f_gate"] = gate_reason
    json_report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp052_report(json_report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))
    gate_errors = validate_exactness_gate(report)
    if gate_errors and report.exactkv_failures > 0:
        json_report["status"] = "failed"
        json_report["exactness_gate_errors"] = gate_errors

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    print(
        f"Exp 052: {json_report['status']} exact={report.token_exact_match_count}/"
        f"{report.total_cells} draft_div={report.draft_divergence_count} "
        f"gate={gate_reason}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report.exactkv_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
