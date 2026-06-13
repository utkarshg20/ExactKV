#!/usr/bin/env python3
"""Experiment 040: Shard external-drafter lossiness and length ablation.

Bounded ablation — NOT Shard integration, NOT default registry.
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

from exactkv.external.shard_ablation import (  # noqa: E402
    DEFAULT_MODEL,
    build_ablation_grid,
    build_ablation_report,
    summarize_setting_result,
)
from exactkv.external.shard_probe import (  # noqa: E402
    check_tokenizer_alignment,
    compare_token_sequences,
    prompt_ids_comparable,
    resolve_shard_repo_path,
    try_import_shard,
)
from exactkv.external.shard_stress_panel import (  # noqa: E402
    DEFAULT_DRAFT_LEN,
    build_stress_prompt_panel,
    classify_divergence_kind,
)
from exactkv.research.external_drafter_probe import (  # noqa: E402
    run_external_drafter_probe,
    trajectory_token_agreement,
)
from exactkv.runtime.generation import generate_full_greedy  # noqa: E402
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402
from scripts.research.run_exp039_shard_stress_panel import (  # noqa: E402
    ShardDraftSession,
    _check_torch_transformers,
    _default_dtype,
    _ensure_shard_on_path,
    _enrich_comparison,
)

DEFAULT_JSON = _ROOT / "reports" / "experiment_040_shard_external_ablation.json"


def run_setting(
    *,
    setting: dict[str, Any],
    panel: list[dict[str, str]],
    verifier_runtime: ModelRuntime,
    shard_session: ShardDraftSession,
    draft_len: int,
) -> dict[str, Any]:
    setting_name = setting["setting_name"]
    max_new_tokens = int(setting["max_new_tokens"])
    shard_settings = dict(setting["shard_settings"])
    shard_session.shard_settings = shard_settings

    prompt_results: list[dict[str, Any]] = []
    for entry in panel:
        prompt = entry["prompt"]
        hf_full = generate_full_greedy(verifier_runtime, prompt, max_new_tokens)
        verifier_ids = hf_full.generated_ids.squeeze(0).tolist()
        hf_prompt_ids = verifier_runtime.tokenizer.encode(prompt, add_special_tokens=False)

        shard_prompt_ids, shard_draft_ids, shard_err = shard_session.draft_ids(
            prompt,
            max_new_tokens=max_new_tokens,
        )

        alignment = check_tokenizer_alignment(
            verifier_runtime.tokenizer,
            verifier_runtime.tokenizer,
            prompt,
            generated_ids=verifier_ids,
        )
        prompt_aligned = prompt_ids_comparable(
            hf_prompt_ids, shard_prompt_ids, verifier_runtime.tokenizer
        )
        token_alignment_pass = alignment["alignment_pass"] and prompt_aligned and shard_err is None
        blocked = shard_err is not None or not token_alignment_pass

        comparison: dict[str, Any] | None = None
        external_probe: dict[str, Any] | None = None
        exactkv_failure = False

        if token_alignment_pass and shard_draft_ids:
            comparison = compare_token_sequences(verifier_ids, shard_draft_ids)
            comparison = _enrich_comparison(
                comparison,
                tokenizer=verifier_runtime.tokenizer,
                draft_ids=shard_draft_ids,
                verifier_ids=verifier_ids,
            )
            ext = run_external_drafter_probe(
                verifier_runtime,
                prompt,
                shard_draft_ids,
                draft_len=draft_len,
                max_new_tokens=max_new_tokens,
                token_alignment_safe=True,
            )
            external_probe = ext.to_dict()
            external_probe["metric_class"] = "external_probe_hf_verifier"
            external_probe["not_exactkv_compressor_acceptance"] = True
            external_probe["trajectory"] = trajectory_token_agreement(
                verifier_ids, shard_draft_ids
            )
            committed = ext.committed_output_ids
            compare_len = min(len(committed), len(verifier_ids))
            exactkv_failure = committed[:compare_len] != verifier_ids[:compare_len]
        else:
            if shard_err:
                alignment["shard_error"] = shard_err
            if not prompt_aligned:
                alignment["shard_prompt_ids"] = shard_prompt_ids

        prompt_results.append(
            {
                "prompt_id": entry["prompt_id"],
                "category": entry.get("category", entry.get("panel_category")),
                "panel_category": entry.get("panel_category", entry.get("category")),
                "prompt": prompt,
                "blocked": blocked,
                "tokenizer_alignment": alignment,
                "token_alignment_pass": token_alignment_pass,
                "comparison": comparison,
                "external_probe_verification": external_probe,
                "exactkv_failure": exactkv_failure,
            }
        )

    aligned = [r for r in prompt_results if r.get("token_alignment_pass") and r.get("comparison")]
    if not aligned:
        return summarize_setting_result(
            setting_name=setting_name,
            max_new_tokens=max_new_tokens,
            shard_settings=shard_settings,
            prompt_results=prompt_results,
            status="restricted_no_go",
            blocked_reason="alignment failed on all prompts for this setting",
        )

    return summarize_setting_result(
        setting_name=setting_name,
        max_new_tokens=max_new_tokens,
        shard_settings=shard_settings,
        prompt_results=prompt_results,
        status="pass",
    )


def run_ablation(
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
    settings, skipped = build_ablation_grid()
    panel = build_stress_prompt_panel(per_category=per_category, max_prompts=max_prompts)

    dep_err = _check_torch_transformers()
    if dep_err:
        return build_ablation_report(
            ablation_status="restricted_no_go",
            blocked_reason=dep_err,
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            draft_len=draft_len,
            prompt_count=len(panel),
            settings_tested=[s["setting_name"] for s in settings],
            settings_skipped=skipped,
            setting_results=[],
            notes=[dep_err],
            recommendation="restricted_no_go",
        )

    try:
        verifier_runtime = ModelRuntime(model_name=model_name, device=device, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        return build_ablation_report(
            ablation_status="restricted_no_go",
            blocked_reason=f"verifier model load blocked: {exc}",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            draft_len=draft_len,
            prompt_count=len(panel),
            settings_tested=[s["setting_name"] for s in settings],
            settings_skipped=skipped,
            setting_results=[],
            notes=[str(exc)],
            recommendation="restricted_no_go",
        )

    try:
        shard_session = ShardDraftSession.load(
            model_name,
            dtype=dtype,
            cache_cls=cache_cls,
            enable_llama_fused_attention=enable_llama_fused_attention,
            shard_settings=dict(settings[0]["shard_settings"]),
        )
    except Exception as exc:  # noqa: BLE001
        return build_ablation_report(
            ablation_status="restricted_no_go",
            blocked_reason=f"Shard model load failed: {exc}",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            draft_len=draft_len,
            prompt_count=len(panel),
            settings_tested=[s["setting_name"] for s in settings],
            settings_skipped=skipped,
            setting_results=[],
            notes=[str(exc)],
            recommendation="restricted_no_go",
        )

    setting_results: list[dict[str, Any]] = []
    for setting in settings:
        print(f"--- setting {setting['setting_name']} max_new_tokens={setting['max_new_tokens']} ---")
        row = run_setting(
            setting=setting,
            panel=panel,
            verifier_runtime=verifier_runtime,
            shard_session=shard_session,
            draft_len=draft_len,
        )
        # Drop bulky per-prompt payloads from top-level JSON (keep summary + examples).
        slim = {k: v for k, v in row.items() if k != "prompt_results"}
        setting_results.append(slim)
        print(
            f"  divergence={row.get('divergence_count')} "
            f"rate={row.get('divergence_rate')} "
            f"mean_prefix={row.get('accepted_prefix_mean')} "
            f"exactkv_failures={row.get('exactkv_failures')}"
        )

    total_exactkv_failures = sum(r.get("exactkv_failures") or 0 for r in setting_results)
    pass_rows = [r for r in setting_results if r.get("status") == "pass"]
    baseline = next((r for r in pass_rows if r["setting_name"] == "baseline_64tok"), None)
    lossy = next((r for r in pass_rows if r["setting_name"] == "stream_bits_4"), None)

    if total_exactkv_failures > 0:
        recommendation = "restricted_go_verify_harness"
    elif lossy and baseline and (lossy.get("divergence_count", 0) > baseline.get("divergence_count", 0)):
        recommendation = "expand_shard_lossy_ablation"
    elif pass_rows:
        recommendation = "expand_shard_or_try_spectralquant"
    else:
        recommendation = "restricted_no_go"

    notes = [
        "Reuses Exp039 32-prompt panel across settings.",
        "stream_bits=8 is documented-lossless streaming — not labeled lossy compression.",
        "Draft divergence is useful signal; exactkv_failures counts verifier output mismatches.",
        "External Shard README metrics are not ExactKV results.",
    ]

    return build_ablation_report(
        ablation_status="pass" if pass_rows else "restricted_no_go",
        blocked_reason="" if pass_rows else "no setting completed successfully",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used=model_name,
        draft_len=draft_len,
        prompt_count=len(panel),
        settings_tested=[s["setting_name"] for s in settings],
        settings_skipped=skipped,
        setting_results=setting_results,
        notes=notes,
        recommendation=recommendation,
    )


def run_ablation_job(
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
    settings, skipped = build_ablation_grid()

    if repo_path is None:
        report = build_ablation_report(
            ablation_status="blocked",
            blocked_reason="blocked: Shard repo not provided (set SHARD_REPO_PATH)",
            shard_repo_path_present=False,
            shard_import_success=False,
            model_used=model_name,
            draft_len=draft_len,
            prompt_count=0,
            settings_tested=[s["setting_name"] for s in settings],
            settings_skipped=skipped,
            setting_results=[],
            notes=["Export SHARD_REPO_PATH=/path/to/shard."],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    import_result = try_import_shard(repo_path)
    if not import_result.success:
        report = build_ablation_report(
            ablation_status="blocked",
            blocked_reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            model_used=model_name,
            draft_len=draft_len,
            prompt_count=0,
            settings_tested=[s["setting_name"] for s in settings],
            settings_skipped=skipped,
            setting_results=[],
            notes=[import_result.reason],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    if not try_run:
        planned = build_stress_prompt_panel(per_category=per_category, max_prompts=max_prompts)
        report = build_ablation_report(
            ablation_status="blocked",
            blocked_reason="blocked: ablation not executed (pass --try-run)",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            draft_len=draft_len,
            prompt_count=len(planned),
            settings_tested=[s["setting_name"] for s in settings],
            settings_skipped=skipped,
            setting_results=[],
            notes=[f"Planned settings: {len(settings)}.", f"Planned prompts: {len(planned)}."],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    _ensure_shard_on_path(repo_path)
    import_result = try_import_shard(repo_path)
    if not import_result.success or import_result.cache_cls is None:
        report = build_ablation_report(
            ablation_status="blocked",
            blocked_reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            model_used=model_name,
            draft_len=draft_len,
            prompt_count=0,
            settings_tested=[s["setting_name"] for s in settings],
            settings_skipped=skipped,
            setting_results=[],
            notes=[import_result.reason],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    resolved_dtype = dtype or _default_dtype()
    report = run_ablation(
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
    parser = argparse.ArgumentParser(description="Experiment 040 — Shard ablation")
    parser.add_argument("--try-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=os.environ.get("SHARD_PROBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--device", default=os.environ.get("SHARD_PROBE_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.environ.get("SHARD_PROBE_DTYPE"))
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=48)
    args = parser.parse_args()

    report = run_ablation_job(
        try_run=args.try_run,
        model_name=args.model,
        draft_len=args.draft_len,
        device=args.device,
        dtype=args.dtype,
        per_category=args.per_category,
        max_prompts=args.max_prompts,
    )
    write_json_report(report, args.json_out)

    status = report["ablation_status"]
    print(f"Shard ablation: {status}")
    reason = report.get("blocked_reason") or ""
    if reason:
        print(reason)
    for row in report.get("setting_results") or []:
        print(
            f"  {row.get('setting_name')}: div={row.get('divergence_count')} "
            f"rate={row.get('divergence_rate')} mean_prefix={row.get('accepted_prefix_mean')} "
            f"exactkv_failures={row.get('exactkv_failures')}"
        )
    print(f"recommendation={report.get('recommendation')}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
