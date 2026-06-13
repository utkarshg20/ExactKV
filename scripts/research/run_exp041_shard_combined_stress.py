#!/usr/bin/env python3
"""Experiment 041: Shard external-drafter combined stress (stream_bits=4, 128 tok).

Single bounded configuration — NOT Shard integration, NOT default registry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.external.shard_combined_stress import (  # noqa: E402
    COMBINED_SHARD_SETTINGS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    build_combined_report,
    build_exp040_comparison_summary,
    summarize_combined_from_prompt_results,
)
from exactkv.external.shard_probe import resolve_shard_repo_path, try_import_shard  # noqa: E402
from exactkv.external.shard_stress_panel import DEFAULT_DRAFT_LEN, build_stress_prompt_panel  # noqa: E402
from scripts.research.run_exp040_shard_ablation import run_setting  # noqa: E402
from scripts.research.run_exp039_shard_stress_panel import (  # noqa: E402
    ShardDraftSession,
    _check_torch_transformers,
    _default_dtype,
    _ensure_shard_on_path,
)
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_041_shard_combined_stress.json"


def run_combined_stress(
    *,
    model_name: str,
    draft_len: int,
    dtype: str,
    device: str,
    cache_cls: Any,
    enable_llama_fused_attention: Any,
    per_category: int,
    max_prompts: int,
) -> dict[str, Any]:
    panel = build_stress_prompt_panel(per_category=per_category, max_prompts=max_prompts)
    setting = {
        "setting_name": "combined_stream_bits_4_128tok",
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "shard_settings": dict(COMBINED_SHARD_SETTINGS),
    }

    dep_err = _check_torch_transformers()
    if dep_err:
        return build_combined_report(
            combined_status="restricted_no_go",
            blocked_reason=dep_err,
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result={"prompt_count": len(panel), "exactkv_failures": None},
            notes=[dep_err],
            recommendation="restricted_no_go",
        )

    try:
        verifier_runtime = ModelRuntime(model_name=model_name, device=device, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        return build_combined_report(
            combined_status="restricted_no_go",
            blocked_reason=f"verifier model load blocked: {exc}",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result={"prompt_count": len(panel), "exactkv_failures": None},
            notes=[str(exc)],
            recommendation="restricted_no_go",
        )

    try:
        shard_session = ShardDraftSession.load(
            model_name,
            dtype=dtype,
            cache_cls=cache_cls,
            enable_llama_fused_attention=enable_llama_fused_attention,
            shard_settings=COMBINED_SHARD_SETTINGS,
        )
    except Exception as exc:  # noqa: BLE001
        return build_combined_report(
            combined_status="restricted_no_go",
            blocked_reason=f"Shard model load failed: {exc}",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result={"prompt_count": len(panel), "exactkv_failures": None},
            notes=[str(exc)],
            recommendation="restricted_no_go",
        )

    row = run_setting(
        setting=setting,
        panel=panel,
        verifier_runtime=verifier_runtime,
        shard_session=shard_session,
        draft_len=draft_len,
    )
    prompt_results = row.pop("prompt_results", [])

    if row.get("status") != "pass":
        result = summarize_combined_from_prompt_results(
            prompt_results,
            status=row.get("status", "restricted_no_go"),
            blocked_reason=row.get("blocked_reason", ""),
        )
        return build_combined_report(
            combined_status="restricted_no_go",
            blocked_reason=row.get("blocked_reason", "alignment failed"),
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result=result,
            notes=["Combined stress did not complete alignment on all prompts."],
            recommendation="restricted_no_go",
        )

    result = summarize_combined_from_prompt_results(prompt_results, status="pass")
    report = build_combined_report(
        combined_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used=model_name,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        draft_len=draft_len,
        shard_settings=COMBINED_SHARD_SETTINGS,
        result=result,
        notes=[
            "Combined: stream_bits=4 + max_new_tokens=128 on Exp039 32-prompt panel.",
            "Lossy streaming (4-bit); stream_bits=8 is documented lossless in Shard.",
            "Draft divergence is useful signal; exactkv_failures counts verifier mismatches.",
            "External Shard README metrics are not ExactKV results.",
        ],
        recommendation=_recommendation(result),
    )
    return report


def _recommendation(result: dict[str, Any]) -> str:
    vs = build_exp040_comparison_summary(result)
    if result.get("exactkv_failures", 0) > 0:
        return "restricted_go_verify_harness"
    if vs.get("increased_vs_all_exp040_singles"):
        return "stop_shard_bounded_probe_complete"
    if vs.get("increased_vs_length_128_or_stream_bits_4"):
        return "stop_shard_bounded_probe_complete"
    return "move_to_spectralquant_or_archive_shard"


def run_combined_job(
    *,
    try_run: bool,
    model_name: str,
    draft_len: int,
    device: str,
    dtype: str | None,
    per_category: int,
    max_prompts: int,
) -> dict[str, Any]:
    repo_path = resolve_shard_repo_path()
    generated_at = datetime.now(timezone.utc).isoformat()

    if repo_path is None:
        report = build_combined_report(
            combined_status="blocked",
            blocked_reason="blocked: Shard repo not provided (set SHARD_REPO_PATH)",
            shard_repo_path_present=False,
            shard_import_success=False,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result={"prompt_count": 0, "exactkv_failures": None},
            notes=["Export SHARD_REPO_PATH=/path/to/shard."],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    import_result = try_import_shard(repo_path)
    if not import_result.success:
        report = build_combined_report(
            combined_status="blocked",
            blocked_reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result={"prompt_count": 0, "exactkv_failures": None},
            notes=[import_result.reason],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    if not try_run:
        planned = build_stress_prompt_panel(per_category=per_category, max_prompts=max_prompts)
        report = build_combined_report(
            combined_status="blocked",
            blocked_reason="blocked: combined stress not executed (pass --try-run)",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result={"prompt_count": len(planned), "exactkv_failures": None},
            notes=[f"Planned prompts: {len(planned)}.", "Setting: stream_bits=4, max_new_tokens=128."],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    _ensure_shard_on_path(repo_path)
    import_result = try_import_shard(repo_path)
    if not import_result.success or import_result.cache_cls is None:
        report = build_combined_report(
            combined_status="blocked",
            blocked_reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            model_used=model_name,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            draft_len=draft_len,
            shard_settings=COMBINED_SHARD_SETTINGS,
            result={"prompt_count": 0, "exactkv_failures": None},
            notes=[import_result.reason],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    resolved_dtype = dtype or _default_dtype()
    report = run_combined_stress(
        model_name=model_name,
        draft_len=draft_len,
        dtype=resolved_dtype,
        device=device,
        cache_cls=import_result.cache_cls,
        enable_llama_fused_attention=import_result.enable_llama_fused_attention,
        per_category=per_category,
        max_prompts=max_prompts,
    )
    report["generated_at"] = generated_at
    report["shard_repo_path"] = str(repo_path)
    report["dtype"] = resolved_dtype
    return report


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 041 — Shard combined stress")
    parser.add_argument("--try-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=os.environ.get("SHARD_PROBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--device", default=os.environ.get("SHARD_PROBE_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.environ.get("SHARD_PROBE_DTYPE"))
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=48)
    args = parser.parse_args()

    report = run_combined_job(
        try_run=args.try_run,
        model_name=args.model,
        draft_len=args.draft_len,
        device=args.device,
        dtype=args.dtype,
        per_category=args.per_category,
        max_prompts=args.max_prompts,
    )
    write_json_report(report, args.json_out)

    status = report["combined_status"]
    print(f"Shard combined stress: {status}")
    reason = report.get("blocked_reason") or ""
    if reason:
        print(reason)
    if status == "pass":
        print(
            f"prompts={report.get('prompt_count')} "
            f"divergence={report.get('divergence_count')} "
            f"rate={report.get('divergence_rate')} "
            f"mean_prefix={report.get('accepted_prefix_mean')} "
            f"exactkv_failures={report.get('exactkv_failures')}"
        )
        vs = report.get("combined_vs_exp040") or {}
        print(f"increased_vs_exp040_singles={vs.get('increased_vs_all_exp040_singles')}")
    print(f"recommendation={report.get('recommendation')}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
