#!/usr/bin/env python3
"""Experiment 058: expanded GPU memory accounting panel (Phase 14C).

Broader exactness-gated memory panel across prompts, compressors, draft lengths,
dtypes, and storage backends. **Not** a memory savings claim.
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

import transformers

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL  # noqa: E402
from exactkv.cache.restored_verifier_runner import DEFAULT_SMOKE_PROMPT_IDS  # noqa: E402
from exactkv.metrics.gpu_memory_accounting import (  # noqa: E402
    DEFAULT_EXP056_REPORT,
    DEFAULT_EXP057_REPORT,
    DEFAULT_EXP058_FILE_ROOT,
    DEFAULT_EXP058_REPORT,
    DEFAULT_EXPANDED_COMPRESSORS,
    DEFAULT_EXPANDED_DRAFT_LENS,
    DEFAULT_EXPANDED_PROMPT_IDS,
    DEFAULT_EXPANDED_STORAGE_BACKENDS,
    DEFAULT_MAX_NEW_TOKENS,
    check_exp056_exactness_gate,
    run_expanded_gpu_memory_panel,
    validate_exp058_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 058 expanded GPU memory panel")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_EXP058_REPORT)
    parser.add_argument("--exp056-report", type=Path, default=DEFAULT_EXP056_REPORT)
    parser.add_argument("--exp057-report", type=Path, default=DEFAULT_EXP057_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--prompt-ids",
        default=",".join(DEFAULT_EXPANDED_PROMPT_IDS),
    )
    parser.add_argument(
        "--draft-lens",
        default=",".join(str(x) for x in DEFAULT_EXPANDED_DRAFT_LENS),
    )
    parser.add_argument(
        "--storage-backends",
        default=",".join(DEFAULT_EXPANDED_STORAGE_BACKENDS),
    )
    parser.add_argument(
        "--compressors",
        default=",".join(DEFAULT_EXPANDED_COMPRESSORS),
    )
    parser.add_argument("--file-storage-root", type=Path, default=DEFAULT_EXP058_FILE_ROOT)
    args = parser.parse_args()

    exp056_ok, exp056_blockers = check_exp056_exactness_gate(args.exp056_report)
    if not exp056_ok:
        print("Exp 058 blocked — Exp 056 exactness gate not passed:")
        for b in exp056_blockers:
            print(f"  - {b}")
        blocked = run_expanded_gpu_memory_panel(
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
    draft_lens = [int(x.strip()) for x in args.draft_lens.split(",") if x.strip()]
    storage_backends = [x.strip() for x in args.storage_backends.split(",") if x.strip()]
    compressors = [x.strip() for x in args.compressors.split(",") if x.strip()]

    result = run_expanded_gpu_memory_panel(
        model_id=args.model,
        prompt_ids=prompt_ids,
        compressor_names=compressors,
        draft_lens=draft_lens,
        storage_backends=storage_backends,
        max_new_tokens=args.max_new_tokens,
        exp056_report_path=args.exp056_report,
        exp057_report_path=args.exp057_report,
        file_storage_root=args.file_storage_root,
    )
    report = result.to_dict()
    report["transformers_version"] = transformers.__version__
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    schema_errors = validate_exp058_report(report)
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Exp 058: {report['status']} slices={len(result.slices)} "
        f"exact={result.token_exact_match_count}/{result.total_cells} "
        f"dtypes={result.dtype_configs}"
    )
    agg = result.aggregate_peak_stats
    for label in ("full_greedy", "restored_verifier_runtime", "kv_capture_store_reload"):
        stats = agg.get(label, {})
        if stats.get("count"):
            print(
                f"  {label}: min={stats['min']} max={stats['max']} mean={stats['mean']:.0f}"
            )
    for note in result.stability_notes:
        print(f"  note: {note}")
    if result.blockers:
        print(f"  blockers: {result.blockers}")
    print(f"Wrote {args.json_out}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
