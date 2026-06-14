#!/usr/bin/env python3
"""Experiment 054: Experimental restored-verifier runtime smoke (Phase 13A).

Explicit opt-in experimental runtime wrapper — **not** default ExactKVGenerator.
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

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL, FORBIDDEN_CLAIMS  # noqa: E402
from exactkv.cache.offline_verifier import VERIFIER_SOURCE  # noqa: E402
from exactkv.runtime.experimental import (  # noqa: E402
    EXPERIMENT_054_ID,
    EXP054_CLAIM_NOTE,
    default_experimental_smoke_config,
    report_to_exp054_json,
    run_experimental_restored_verifier,
    validate_exp054_report,
)

DEFAULT_JSON = _ROOT / "reports" / "experiment_054_experimental_restored_verifier_runtime.json"


def _blocked_report(reason: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_054_ID,
        "status": "blocked",
        "runtime_mode": "restored_verifier_offline",
        "enabled": True,
        "model": DEFAULT_MODEL,
        "device": "unknown",
        "dtype": "unknown",
        "transformers_version": transformers.__version__,
        "prompt_count": 0,
        "storage_backends": [],
        "compressor_names": [],
        "draft_len": 0,
        "draft_lens": [],
        "max_new_tokens": 0,
        "verifier_source": VERIFIER_SOURCE,
        "total_cells": 0,
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "mean_acceptance": 0.0,
        "draft_divergence_count": 0,
        "restore_blockers": [reason],
        "draft_blockers": [],
        "verification_blockers": [],
        "blockers": {
            "restore_blockers": [reason],
            "draft_blockers": [],
            "verification_blockers": [],
        },
        "runner_called": False,
        "claim_note": EXP054_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "message": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 054 experimental restored-verifier runtime")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--draft-len", type=int, default=4)
    parser.add_argument("--disabled", action="store_true", help="Run with enabled=False smoke")
    args = parser.parse_args()

    if args.disabled:
        from exactkv.runtime.experimental import ExperimentalRestoredVerifierConfig

        result = run_experimental_restored_verifier(ExperimentalRestoredVerifierConfig.disabled())
        json_report = report_to_exp054_json(result)
        json_report["transformers_version"] = transformers.__version__
        json_report["generated_at"] = datetime.now(timezone.utc).isoformat()
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
        print(f"Exp 054: disabled runner_called={result.runner_called}")
        print(f"Wrote {args.json_out}")
        return 0

    config = default_experimental_smoke_config(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        draft_lens=[args.draft_len],
    )

    try:
        result = run_experimental_restored_verifier(config)
    except Exception as exc:  # noqa: BLE001
        blocked = _blocked_report(f"experimental runtime failed: {type(exc).__name__}: {exc}")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        print(f"Exp 054: blocked — {exc}")
        return 1

    json_report = report_to_exp054_json(result)
    json_report["transformers_version"] = transformers.__version__
    json_report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp054_report(json_report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    report = result.runner_report
    if report is not None:
        print(
            f"Exp 054: {json_report['status']} enabled={result.enabled} "
            f"exact={report.token_exact_match_count}/{report.total_cells} "
            f"draft_div={report.draft_divergence_count}"
        )
    else:
        print(f"Exp 054: {json_report['status']} enabled={result.enabled}")
    print(f"Wrote {args.json_out}")
    return 0 if result.status in ("pass", "disabled") else 1


if __name__ == "__main__":
    raise SystemExit(main())
