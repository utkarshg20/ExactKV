#!/usr/bin/env python3
"""Experiment 049: Offline verifier restore with lossy draft (Phase 12D).

Lossy compressor draft tokens verified against reloaded full-KV payloads in an
isolated loop. **Not** wired into ``ExactKVGenerator`` defaults.
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

from exactkv.cache.offline_verifier import (  # noqa: E402
    DEFAULT_DRAFT_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    EXPERIMENT_049_ID,
    FORBIDDEN_CLAIMS,
    OFFLINE_LOSSY_CLAIM_NOTE,
    VERIFIER_SOURCE,
    default_lossy_compressors,
    default_offline_prompts,
    run_offline_lossy_verifier_cell,
    validate_exp049_report,
)
from exactkv.cache.storage import FileKVStorageBackend, InMemoryKVStorageBackend  # noqa: E402
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_049_offline_verifier_lossy_draft.json"


def _blocked_report(reason: str) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_049_ID,
        "status": "blocked",
        "model": DEFAULT_MODEL,
        "device": "unknown",
        "dtype": "unknown",
        "transformers_version": transformers.__version__,
        "prompt_count": 0,
        "storage_backends": [],
        "compressor_names": [],
        "draft_len": DEFAULT_DRAFT_LEN,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [],
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "accepted_prefix_lengths": [],
        "mean_acceptance": 0.0,
        "first_divergences": [],
        "restore_blockers": [reason],
        "draft_blockers": [],
        "verification_blockers": [],
        "claim_note": OFFLINE_LOSSY_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 049 offline verifier lossy draft")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--tmpdir", type=Path, default=_ROOT / "reports" / "exp049_kv_files")
    args = parser.parse_args()

    prompts = default_offline_prompts()
    compressors = default_lossy_compressors()
    if not compressors:
        report = _blocked_report("no lossy compressors available in registry")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Exp 049: blocked — no compressors")
        return 0

    cells: list[dict[str, Any]] = []
    restore_blockers: list[str] = []
    draft_blockers: list[str] = []
    verification_blockers: list[str] = []
    backends_tested: list[str] = []
    first_divergences: list[dict[str, Any]] = []
    all_accepted_prefix_lengths: list[list[int]] = []
    per_cell_mean_acceptance: list[float] = []
    exact_matches = 0
    exactkv_failures = 0

    try:
        runtime = ModelRuntime(args.model, device=args.device, dtype=args.dtype)
    except Exception as exc:  # noqa: BLE001
        report = _blocked_report(f"model load failed: {type(exc).__name__}: {exc}")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Exp 049: blocked — {exc}")
        return 0

    backends: list[tuple[str, Any]] = [
        ("in_memory_kv_storage", InMemoryKVStorageBackend()),
    ]
    args.tmpdir.mkdir(parents=True, exist_ok=True)
    backends.append(("file_kv_storage", FileKVStorageBackend(args.tmpdir)))

    for backend_name, backend in backends:
        if backend_name not in backends_tested:
            backends_tested.append(backend_name)
        for compressor_name in compressors:
            for entry in prompts:
                result = run_offline_lossy_verifier_cell(
                    runtime,
                    prompt_id=entry["prompt_id"],
                    prompt=entry["prompt"],
                    backend=backend,
                    compressor_name=compressor_name,
                    max_new_tokens=args.max_new_tokens,
                    draft_len=args.draft_len,
                )
                row = result.to_dict()
                cells.append(row)
                all_accepted_prefix_lengths.append(result.accepted_prefix_lengths)
                per_cell_mean_acceptance.append(result.mean_acceptance)
                if result.restore_blocker:
                    restore_blockers.append(
                        f"{backend_name}/{compressor_name}/{entry['prompt_id']}: "
                        f"{result.restore_blocker}"
                    )
                if result.draft_blocker:
                    draft_blockers.append(
                        f"{backend_name}/{compressor_name}/{entry['prompt_id']}: "
                        f"{result.draft_blocker}"
                    )
                if result.verification_blocker:
                    verification_blockers.append(
                        f"{backend_name}/{compressor_name}/{entry['prompt_id']}: "
                        f"{result.verification_blocker}"
                    )
                if result.token_exact_match:
                    exact_matches += 1
                else:
                    exactkv_failures += 1
                    if result.first_divergence_idx is not None:
                        first_divergences.append(
                            {
                                "prompt_id": entry["prompt_id"],
                                "backend_name": backend_name,
                                "compressor_name": compressor_name,
                                "first_divergence_idx": result.first_divergence_idx,
                            }
                        )

    aggregate_mean = (
        sum(per_cell_mean_acceptance) / len(per_cell_mean_acceptance)
        if per_cell_mean_acceptance
        else 0.0
    )

    report = {
        "experiment_id": EXPERIMENT_049_ID,
        "status": "pass" if exactkv_failures == 0 else "failed",
        "model": args.model,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "transformers_version": transformers.__version__,
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "draft_len": args.draft_len,
        "storage_backends": backends_tested,
        "compressor_names": compressors,
        "verifier_source": VERIFIER_SOURCE,
        "cells": cells,
        "exactkv_failures": exactkv_failures,
        "token_exact_match_count": exact_matches,
        "accepted_prefix_lengths": all_accepted_prefix_lengths,
        "mean_acceptance": aggregate_mean,
        "per_cell_mean_acceptance": per_cell_mean_acceptance,
        "first_divergences": first_divergences,
        "restore_blockers": restore_blockers,
        "draft_blockers": draft_blockers,
        "verification_blockers": verification_blockers,
        "claim_note": OFFLINE_LOSSY_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    errors = validate_exp049_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Exp 049: {report['status']} exact={exact_matches} failures={exactkv_failures} "
        f"mean_accept={aggregate_mean:.3f} compressors={compressors}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
