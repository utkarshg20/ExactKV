#!/usr/bin/env python3
"""Experiment 047: Full-KV restore panel hardening (Phase 12B).

Multi-prompt HF ``past_key_values`` capture → ``KVStorageBackend`` → reload → continuation
equivalence across device/dtype variants. **Not** wired into ``ExactKVGenerator``.
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

from exactkv.cache.hf_kv_restore import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PANEL_MAX_NEW_TOKENS,
    EXPERIMENT_047_ID,
    FORBIDDEN_CLAIMS,
    PANEL_CLAIM_NOTE,
    default_panel_prompts,
    resolve_panel_device_dtype_configs,
    run_restore_panel_cell,
    validate_exp047_report,
)
from exactkv.cache.storage import FileKVStorageBackend, InMemoryKVStorageBackend  # noqa: E402
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_047_full_kv_restore_panel.json"


def _blocked_report(reason: str) -> dict[str, Any]:
    configs = resolve_panel_device_dtype_configs()
    report = {
        "experiment_id": EXPERIMENT_047_ID,
        "status": "blocked",
        "model": DEFAULT_MODEL,
        "transformers_version": transformers.__version__,
        "total_cells": 0,
        "passed_cells": 0,
        "failed_cells": 0,
        "skipped_cells": 0,
        "storage_backends_tested": [],
        "device_dtype_configs_tested": [c.to_dict() for c in configs],
        "cache_formats_detected": [],
        "aggregate_exactness": {
            "token_exact_match_count": 0,
            "failures_count": 0,
        },
        "per_cell": [],
        "restore_blockers": [reason],
        "claim_note": PANEL_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    errors = validate_exp047_report(report)
    if errors:
        report["schema_validation_errors"] = errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 047 full-KV restore panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_PANEL_MAX_NEW_TOKENS)
    parser.add_argument("--tmpdir", type=Path, default=_ROOT / "reports" / "exp047_kv_files")
    args = parser.parse_args()

    prompts = default_panel_prompts()
    configs = resolve_panel_device_dtype_configs()
    per_cell: list[dict[str, Any]] = []
    restore_blockers: list[str] = []
    backends_tested: list[str] = []
    cache_formats: set[str] = set()

    passed = 0
    failed = 0
    skipped = 0
    exact_matches = 0

    backends: list[tuple[str, Any]] = [
        ("in_memory_kv_storage", InMemoryKVStorageBackend()),
    ]
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
            if cfg.required:
                report = _blocked_report(cfg.skip_reason)
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"Exp 047: blocked — {cfg.skip_reason}")
                return 0
            restore_blockers.append(f"{cfg.device}/{cfg.dtype}: {cfg.skip_reason}")
            continue

        cfg.status = "tested"
        for backend_name, backend in backends:
            if backend_name not in backends_tested:
                backends_tested.append(backend_name)
            for entry in prompts:
                result = run_restore_panel_cell(
                    runtime,
                    prompt_id=entry["prompt_id"],
                    prompt=entry["prompt"],
                    category=entry["category"],
                    backend=backend,
                    max_new_tokens=args.max_new_tokens,
                )
                row = result.to_dict()
                per_cell.append(row)
                if result.cell_status == "passed":
                    passed += 1
                    if result.token_exact_match:
                        exact_matches += 1
                elif result.cell_status == "failed":
                    failed += 1
                    if result.restore_blocker:
                        restore_blockers.append(
                            f"{cfg.device}/{cfg.dtype}/{backend_name}/"
                            f"{entry['prompt_id']}: {result.restore_blocker}"
                        )
                elif result.cell_status == "skipped":
                    skipped += 1
                if result.cache_format != "unknown":
                    cache_formats.add(result.cache_format)

    skipped_configs = sum(1 for c in configs if c.status == "skipped")
    total_cells = passed + failed + skipped

    report = {
        "experiment_id": EXPERIMENT_047_ID,
        "status": "pass" if failed == 0 and passed > 0 else ("failed" if failed else "blocked"),
        "model": args.model,
        "transformers_version": transformers.__version__,
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "total_cells": total_cells,
        "passed_cells": passed,
        "failed_cells": failed,
        "skipped_cells": skipped,
        "skipped_device_dtype_configs": skipped_configs,
        "storage_backends_tested": backends_tested,
        "device_dtype_configs_tested": [c.to_dict() for c in configs],
        "cache_formats_detected": sorted(cache_formats),
        "aggregate_exactness": {
            "token_exact_match_count": exact_matches,
            "failures_count": failed,
        },
        "per_cell": per_cell,
        "restore_blockers": restore_blockers,
        "claim_note": PANEL_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    errors = validate_exp047_report(report)
    if errors:
        raise ValueError("; ".join(errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Exp 047: {report['status']} cells={total_cells} exact={exact_matches} "
        f"failed={failed} skipped={skipped} formats={sorted(cache_formats)}"
    )
    print(f"Wrote {args.json_out}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
