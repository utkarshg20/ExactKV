#!/usr/bin/env python3
"""Experiment 053: Runner-backed restored-verifier drift panel (Phase 12H).

Exp 050-style drift panel via ``run_restored_verifier()`` only — no duplicated loop.
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

import torch
import transformers

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL, FORBIDDEN_CLAIMS  # noqa: E402
from exactkv.cache.offline_verifier import VERIFIER_SOURCE  # noqa: E402
from exactkv.cache.restored_verifier_runner import (  # noqa: E402
    DEFAULT_PANEL_DRAFT_LENS,
    DEFAULT_PANEL_MAX_NEW_TOKENS,
    EXPERIMENT_053_ID,
    EXP053_CLAIM_NOTE,
    default_panel_config,
    default_panel_prompt_ids,
    report_to_exp053_json,
    run_restored_verifier,
    validate_exp053_report,
)

DEFAULT_JSON = _ROOT / "reports" / "experiment_053_restored_verifier_runner_panel.json"


def _blocked_report(reason: str, *, device: str = "unknown", dtype: str = "unknown") -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_053_ID,
        "status": "blocked",
        "config": {},
        "model": DEFAULT_MODEL,
        "device": device,
        "dtype": dtype,
        "transformers_version": transformers.__version__,
        "prompt_count": 0,
        "storage_backends": [],
        "compressor_names": [],
        "draft_len_values": list(DEFAULT_PANEL_DRAFT_LENS),
        "max_new_tokens": DEFAULT_PANEL_MAX_NEW_TOKENS,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [],
        "total_cells": 0,
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "mean_acceptance": 0.0,
        "draft_divergence_count": 0,
        "semantic_divergence_count": 0,
        "no_real_drift_observed": True,
        "first_divergences": [],
        "restore_blockers": [reason],
        "draft_blockers": [],
        "verification_blockers": [],
        "blockers": {
            "restore_blockers": [reason],
            "draft_blockers": [],
            "verification_blockers": [],
        },
        "claim_note": EXP053_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 053 runner-backed drift panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_PANEL_MAX_NEW_TOKENS)
    parser.add_argument("--full-panel", action="store_true", help="Use full 12-prompt drift panel")
    parser.add_argument("--include-file-backend", action="store_true")
    parser.add_argument("--cuda", action="store_true", help="Run on CUDA (skip cleanly if unavailable)")
    parser.add_argument("--tmpdir", type=Path, default=_ROOT / "reports" / "exp053_kv_files")
    args = parser.parse_args()

    device = "cuda" if args.cuda else args.device
    dtype = "float16" if args.cuda else args.dtype

    if args.cuda and not torch.cuda.is_available():
        report = _blocked_report("CUDA unavailable — experiment skipped", device="cuda", dtype=dtype)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Exp 053: blocked — CUDA unavailable")
        return 0

    prompt_ids = default_panel_prompt_ids(full_panel=args.full_panel)
    config = default_panel_config(
        model_id=args.model,
        device=device,
        dtype=dtype,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        file_storage_root=str(args.tmpdir),
    )

    extra_backends = ["file_kv_storage"] if args.include_file_backend else None

    try:
        report = run_restored_verifier(
            config,
            experiment_id=EXPERIMENT_053_ID,
            extra_backends=extra_backends,
        )
    except Exception as exc:  # noqa: BLE001
        blocked = _blocked_report(f"runner failed: {type(exc).__name__}: {exc}", device=device, dtype=dtype)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        print(f"Exp 053: blocked — {exc}")
        return 1

    json_report = report_to_exp053_json(report)
    json_report["transformers_version"] = transformers.__version__
    json_report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp053_report(json_report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    drift_note = (
        "no_real_drift_observed"
        if report.no_real_drift_observed
        else f"draft_div={report.draft_divergence_count}"
    )
    print(
        f"Exp 053: {json_report['status']} cells={report.total_cells} "
        f"exact={report.token_exact_match_count} failures={report.exactkv_failures} "
        f"mean_accept={report.mean_acceptance:.3f} {drift_note}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report.exactkv_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
