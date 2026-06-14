#!/usr/bin/env python3
"""Experiment 051: Offline verifier CUDA drift panel (Phase 12F).

CUDA float16/bfloat16 exactness for reloaded full-KV verifier drift panel.
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

from exactkv.cache.offline_verifier import (  # noqa: E402
    DEFAULT_DRIFT_DRAFT_LENS,
    DEFAULT_DRIFT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    EXPERIMENT_051_ID,
    FORBIDDEN_CLAIMS,
    OFFLINE_CUDA_DRIFT_CLAIM_NOTE,
    VERIFIER_SOURCE,
    configure_cuda_determinism,
    default_cuda_drift_prompts,
    default_drift_stress_compressors,
    resolve_cuda_drift_dtype_configs,
    run_offline_cuda_drift_cell,
    validate_exp051_report,
)
from exactkv.cache.storage import FileKVStorageBackend, InMemoryKVStorageBackend  # noqa: E402
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_051_offline_verifier_cuda_drift_panel.json"


def _blocked_report(reason: str, *, cuda_available: bool = False) -> dict[str, Any]:
    configs = resolve_cuda_drift_dtype_configs()
    dtype_supported = {c.dtype: c.dtype_supported for c in configs}
    skipped = [c.to_dict() for c in configs if c.status == "skipped"]
    return {
        "experiment_id": EXPERIMENT_051_ID,
        "status": "blocked",
        "model": DEFAULT_MODEL,
        "device": "cuda" if cuda_available else "unknown",
        "dtype": "none",
        "transformers_version": transformers.__version__,
        "prompt_count": 0,
        "storage_backends": [],
        "compressor_names": [],
        "draft_len_values": list(DEFAULT_DRIFT_DRAFT_LENS),
        "max_new_tokens": DEFAULT_DRIFT_MAX_NEW_TOKENS,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [],
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "accepted_prefix_lengths": [],
        "mean_acceptance": 0.0,
        "draft_divergence_count": 0,
        "semantic_divergence_count": 0,
        "first_divergences": [],
        "cuda_available": cuda_available,
        "dtype_supported": dtype_supported,
        "skipped_configs": skipped,
        "restore_blockers": [reason] if reason else [],
        "draft_blockers": [],
        "verification_blockers": [],
        "exactness_blockers": [],
        "claim_note": OFFLINE_CUDA_DRIFT_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 051 CUDA offline drift panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_DRIFT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--draft-lens",
        default=",".join(str(x) for x in DEFAULT_DRIFT_DRAFT_LENS),
        help="Comma-separated draft lengths (default: 4,8)",
    )
    parser.add_argument("--full-panel", action="store_true", help="Use full 12-prompt panel")
    parser.add_argument("--no-file-backend", action="store_true", help="Skip FileKVStorageBackend")
    parser.add_argument("--tmpdir", type=Path, default=_ROOT / "reports" / "exp051_kv_files")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        report = _blocked_report("CUDA unavailable — experiment skipped")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Exp 051: blocked — CUDA unavailable")
        return 0

    configure_cuda_determinism()
    draft_lens = [int(x.strip()) for x in args.draft_lens.split(",") if x.strip()]
    prompts = default_cuda_drift_prompts(full_panel=args.full_panel)
    compressors = default_drift_stress_compressors()
    configs = resolve_cuda_drift_dtype_configs()

    if not compressors:
        report = _blocked_report("no drift compressors available in registry", cuda_available=True)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Exp 051: blocked — no compressors")
        return 0

    cells: list[dict[str, Any]] = []
    restore_blockers: list[str] = []
    draft_blockers: list[str] = []
    verification_blockers: list[str] = []
    exactness_blockers: list[str] = []
    backends_tested: list[str] = []
    first_divergences: list[dict[str, Any]] = []
    all_accepted_prefix_lengths: list[list[int]] = []
    per_cell_mean_acceptance: list[float] = []
    total_draft_divergence = 0
    total_semantic_divergence = 0
    exact_matches = 0
    exactkv_failures = 0
    tested_dtypes: list[str] = []

    backends: list[tuple[str, Any]] = [
        ("in_memory_kv_storage", InMemoryKVStorageBackend()),
    ]
    if not args.no_file_backend:
        args.tmpdir.mkdir(parents=True, exist_ok=True)
        backends.append(("file_kv_storage", FileKVStorageBackend(args.tmpdir)))

    for cfg in configs:
        if cfg.status == "skipped":
            continue
        try:
            runtime = ModelRuntime(args.model, device=cfg.device, dtype=cfg.dtype)
        except Exception as exc:  # noqa: BLE001
            cfg.status = "skipped"
            cfg.skip_reason = f"model load failed: {type(exc).__name__}: {exc}"
            restore_blockers.append(f"{cfg.dtype}: {cfg.skip_reason}")
            continue

        cfg.status = "tested"
        tested_dtypes.append(cfg.dtype)
        for backend_name, backend in backends:
            if backend_name not in backends_tested:
                backends_tested.append(backend_name)
            for draft_len in draft_lens:
                for compressor_name in compressors:
                    for entry in prompts:
                        result = run_offline_cuda_drift_cell(
                            runtime,
                            prompt_id=entry["prompt_id"],
                            prompt=entry["prompt"],
                            category=entry["category"],
                            backend=backend,
                            compressor_name=compressor_name,
                            draft_len=draft_len,
                            dtype=cfg.dtype,
                            max_new_tokens=args.max_new_tokens,
                        )
                        row = result.to_dict()
                        cells.append(row)
                        all_accepted_prefix_lengths.append(result.accepted_prefix_lengths)
                        per_cell_mean_acceptance.append(result.mean_acceptance)
                        total_draft_divergence += result.draft_divergence_count
                        total_semantic_divergence += result.semantic_divergence_count
                        if result.restore_blocker:
                            restore_blockers.append(
                                f"{cfg.dtype}/{backend_name}/{compressor_name}/dl{draft_len}/"
                                f"{entry['prompt_id']}: {result.restore_blocker}"
                            )
                        if result.draft_blocker:
                            draft_blockers.append(
                                f"{cfg.dtype}/{backend_name}/{compressor_name}/dl{draft_len}/"
                                f"{entry['prompt_id']}: {result.draft_blocker}"
                            )
                        if result.verification_blocker:
                            verification_blockers.append(
                                f"{cfg.dtype}/{backend_name}/{compressor_name}/dl{draft_len}/"
                                f"{entry['prompt_id']}: {result.verification_blocker}"
                            )
                        if result.exactness_blocker:
                            exactness_blockers.append(
                                f"{cfg.dtype}/{backend_name}/{compressor_name}/dl{draft_len}/"
                                f"{entry['prompt_id']}: {result.exactness_blocker}"
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
                                        "draft_len": draft_len,
                                        "dtype": cfg.dtype,
                                        "first_divergence_idx": result.first_divergence_idx,
                                    }
                                )

        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not tested_dtypes:
        report = _blocked_report(
            "no CUDA dtype configs could be tested",
            cuda_available=True,
        )
        report["skipped_configs"] = [c.to_dict() for c in configs]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Exp 051: blocked — no dtype configs tested")
        return 0

    aggregate_mean = (
        sum(per_cell_mean_acceptance) / len(per_cell_mean_acceptance)
        if per_cell_mean_acceptance
        else 0.0
    )
    dtype_supported = {c.dtype: c.dtype_supported for c in configs}
    skipped = [c.to_dict() for c in configs if c.status == "skipped"]

    report = {
        "experiment_id": EXPERIMENT_051_ID,
        "status": "pass" if exactkv_failures == 0 else "failed",
        "model": args.model,
        "device": "cuda",
        "dtype": ",".join(tested_dtypes),
        "transformers_version": transformers.__version__,
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "draft_len_values": draft_lens,
        "storage_backends": backends_tested,
        "compressor_names": compressors,
        "verifier_source": VERIFIER_SOURCE,
        "cells": cells,
        "exactkv_failures": exactkv_failures,
        "token_exact_match_count": exact_matches,
        "accepted_prefix_lengths": all_accepted_prefix_lengths,
        "mean_acceptance": aggregate_mean,
        "per_cell_mean_acceptance": per_cell_mean_acceptance,
        "draft_divergence_count": total_draft_divergence,
        "semantic_divergence_count": total_semantic_divergence,
        "first_divergences": first_divergences,
        "cuda_available": True,
        "dtype_supported": dtype_supported,
        "skipped_configs": skipped,
        "restore_blockers": restore_blockers,
        "draft_blockers": draft_blockers,
        "verification_blockers": verification_blockers,
        "exactness_blockers": exactness_blockers,
        "claim_note": OFFLINE_CUDA_DRIFT_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    errors = validate_exp051_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Exp 051: {report['status']} exact={exact_matches} failures={exactkv_failures} "
        f"dtypes={tested_dtypes} draft_div={total_draft_divergence}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
