#!/usr/bin/env python3
"""Experiment 046: Full-KV restore smoke (Phase 12A).

Real HF ``past_key_values`` capture → ``KVStorageBackend`` → reload → continuation
equivalence on a tiny prompt panel. **Not** wired into ``ExactKVGenerator``.
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

from exactkv.cache.hf_kv_restore import (  # noqa: E402
    CLAIM_NOTE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    EXPERIMENT_046_ID,
    FORBIDDEN_CLAIMS,
    capture_prefill_kv,
    default_smoke_prompts,
    run_restore_equivalence_for_prompt,
    validate_exp046_report,
)
from exactkv.cache.storage import FileKVStorageBackend, InMemoryKVStorageBackend  # noqa: E402
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_046_full_kv_restore_smoke.json"


def _blocked_report(reason: str) -> dict[str, Any]:
    report = {
        "experiment_id": EXPERIMENT_046_ID,
        "status": "blocked",
        "model": DEFAULT_MODEL,
        "device": "unknown",
        "dtype": "unknown",
        "prompt_count": 0,
        "storage_backends_tested": [],
        "cache_format_detected": "unknown",
        "layer_count": 0,
        "shape_summary": "",
        "token_exact_match_count": 0,
        "failures_count": 0,
        "per_prompt": [],
        "restore_blockers": [reason],
        "claim_note": CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_exp046_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 046 full-KV restore smoke")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--tmpdir", type=Path, default=_ROOT / "reports" / "exp046_kv_files")
    args = parser.parse_args()

    prompts = default_smoke_prompts()
    per_prompt: list[dict[str, Any]] = []
    restore_blockers: list[str] = []
    backends_tested: list[str] = []
    cache_format_detected = "unknown"
    layer_count = 0
    shape_summary = ""
    exact_matches = 0
    failures = 0

    try:
        runtime = ModelRuntime(args.model, device=args.device, dtype=args.dtype)
    except Exception as exc:  # noqa: BLE001
        report = _blocked_report(f"model load failed: {type(exc).__name__}: {exc}")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Exp 046: blocked — {exc}")
        return 0

    backends: list[tuple[str, Any]] = [
        ("in_memory_kv_storage", InMemoryKVStorageBackend()),
    ]
    args.tmpdir.mkdir(parents=True, exist_ok=True)
    backends.append(("file_kv_storage", FileKVStorageBackend(args.tmpdir)))

    for backend_name, backend in backends:
        backends_tested.append(backend_name)
        for entry in prompts:
            result = run_restore_equivalence_for_prompt(
                runtime,
                prompt_id=entry["prompt_id"],
                prompt=entry["prompt"],
                backend=backend,
                max_new_tokens=args.max_new_tokens,
            )
            row = result.to_dict()
            row["backend_name"] = backend_name
            per_prompt.append(row)
            if result.restore_blocker:
                restore_blockers.append(
                    f"{backend_name}/{entry['prompt_id']}: {result.restore_blocker}"
                )
                failures += 1
            elif result.token_exact_match:
                exact_matches += 1
            else:
                failures += 1
            if result.cache_format != "unknown":
                cache_format_detected = result.cache_format
            if not shape_summary and not result.restore_blocker:
                capture = capture_prefill_kv(runtime, entry["prompt"])
                layer_count = capture.cache_summary.layer_count
                shape_summary = capture.cache_summary.shape_summary

    report = {
        "experiment_id": EXPERIMENT_046_ID,
        "status": "pass" if failures == 0 else "failed",
        "model": args.model,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "prompt_count": len(prompts) * len(backends),
        "storage_backends_tested": backends_tested,
        "cache_format_detected": cache_format_detected,
        "layer_count": layer_count,
        "shape_summary": shape_summary,
        "max_new_tokens": args.max_new_tokens,
        "token_exact_match_count": exact_matches,
        "failures_count": failures,
        "per_prompt": per_prompt,
        "restore_blockers": restore_blockers,
        "claim_note": CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    errors = validate_exp046_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Exp 046: {report['status']} exact={exact_matches} failures={failures} "
        f"format={cache_format_detected}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
