#!/usr/bin/env python3
"""Experiment 039: Shard external-drafter stress panel (Phase 10B2).

Bounded stress panel — NOT Shard integration, NOT default registry.
HF full-KV greedy verifier remains authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.external.shard_probe import (  # noqa: E402
    DEFAULT_MODEL,
    check_tokenizer_alignment,
    compare_token_sequences,
    prompt_ids_comparable,
    resolve_shard_repo_path,
    try_import_shard,
)
from exactkv.external.shard_stress_panel import (  # noqa: E402
    DEFAULT_DRAFT_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SHARD_SETTINGS,
    build_panel_report,
    build_stress_prompt_panel,
    classify_divergence_kind,
)
from exactkv.research.external_drafter_probe import (  # noqa: E402
    run_external_drafter_probe,
    trajectory_token_agreement,
)
from exactkv.runtime.generation import generate_full_greedy  # noqa: E402
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_039_shard_external_stress_panel.json"


def _default_dtype() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "float16"
    except ImportError:
        pass
    return "float32"


def _check_torch_transformers() -> str | None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return f"missing dependency: {exc}"
    return None


def _ensure_shard_on_path(repo_path: Path) -> None:
    for candidate in (repo_path / "src", repo_path):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


@dataclass
class ShardDraftSession:
    model: Any
    tokenizer: Any
    cache_cls: Any
    shard_settings: dict[str, Any]

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        dtype: str,
        cache_cls: Any,
        enable_llama_fused_attention: Any,
        shard_settings: dict[str, Any],
    ) -> ShardDraftSession:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        enable_llama_fused_attention(model)
        return cls(
            model=model,
            tokenizer=tokenizer,
            cache_cls=cache_cls,
            shard_settings=shard_settings,
        )

    def draft_ids(self, prompt: str, *, max_new_tokens: int) -> tuple[list[int], list[int], str | None]:
        try:
            import torch

            cache = self.cache_cls.from_model(
                self.model,
                k_target_cr=float(self.shard_settings.get("k_target_cr", 16.0)),
            )
            cache._streaming = bool(self.shard_settings.get("streaming", True))
            cache._stream_bits = int(self.shard_settings.get("stream_bits", 8))
            cache._stream_qjl = bool(self.shard_settings.get("stream_qjl", False))

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[1])
            prompt_ids = inputs["input_ids"][0].tolist()

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    past_key_values=cache,
                )
            draft_ids = out[0, prompt_len:].tolist()[:max_new_tokens]
            return prompt_ids, draft_ids, None
        except Exception as exc:  # noqa: BLE001
            return [], [], f"Shard draft generation failed: {exc}"


def _decode_token(tokenizer: Any, token_id: int | None) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:  # noqa: BLE001
        return ""


def _enrich_comparison(
    comparison: dict[str, Any],
    *,
    tokenizer: Any,
    draft_ids: list[int],
    verifier_ids: list[int],
) -> dict[str, Any]:
    div_idx = comparison.get("first_divergence_index")
    draft_tok = comparison.get("draft_token_id")
    ver_tok = comparison.get("verifier_token_id")
    draft_text = _decode_token(tokenizer, draft_tok)
    ver_text = _decode_token(tokenizer, ver_tok)
    prefix_len = comparison.get("accepted_prefix_length", 0) or 0
    comparison = {
        **comparison,
        "draft_token_text": draft_text,
        "verifier_token_text": ver_text,
        "decoded_draft_prefix": tokenizer.decode(draft_ids[:prefix_len], skip_special_tokens=False),
        "decoded_verifier_prefix": tokenizer.decode(verifier_ids[:prefix_len], skip_special_tokens=False),
        "divergence_kind": classify_divergence_kind(
            draft_text=draft_text,
            verifier_text=ver_text,
            draft_token_id=draft_tok,
            verifier_token_id=ver_tok,
        )
        if div_idx is not None
        else "none",
    }
    return comparison


def run_stress_panel(
    *,
    model_name: str,
    max_new_tokens: int,
    draft_len: int,
    dtype: str,
    device: str,
    cache_cls: Any,
    enable_llama_fused_attention: Any,
    shard_settings: dict[str, Any],
    per_category: int,
    max_prompts: int,
) -> dict[str, Any]:
    dep_err = _check_torch_transformers()
    if dep_err:
        return build_panel_report(
            panel_status="restricted_no_go",
            blocked_reason=dep_err,
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=[],
            notes=[dep_err],
            recommendation="restricted_no_go",
        )

    try:
        verifier_runtime = ModelRuntime(model_name=model_name, device=device, dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        return build_panel_report(
            panel_status="restricted_no_go",
            blocked_reason=f"verifier model load blocked: {exc}",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=[],
            notes=[
                "Llama model access may require HF_TOKEN and Meta license acceptance.",
                str(exc),
            ],
            recommendation="restricted_no_go",
        )

    try:
        shard_session = ShardDraftSession.load(
            model_name,
            dtype=dtype,
            cache_cls=cache_cls,
            enable_llama_fused_attention=enable_llama_fused_attention,
            shard_settings=shard_settings,
        )
    except Exception as exc:  # noqa: BLE001
        return build_panel_report(
            panel_status="restricted_no_go",
            blocked_reason=f"Shard model load failed: {exc}",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=[],
            notes=[str(exc)],
            recommendation="restricted_no_go",
        )

    panel = build_stress_prompt_panel(per_category=per_category, max_prompts=max_prompts)
    prompt_results: list[dict[str, Any]] = []
    alignment_pass_count = 0

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
            alignment_pass_count += 1
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
                "hf_verifier": {
                    "generated_token_ids": verifier_ids,
                    "output_text": hf_full.output_text,
                    "prompt_token_ids": hf_prompt_ids,
                },
                "shard_external_drafter": {
                    "generated_token_ids": shard_draft_ids,
                    "prompt_token_ids": shard_prompt_ids,
                    "uses_compressed_kv_cache": True,
                    "not_exactkv_integrated_compressor": True,
                    "error": shard_err,
                },
                "tokenizer_alignment": alignment,
                "token_alignment_pass": token_alignment_pass,
                "comparison": comparison,
                "external_probe_verification": external_probe,
                "exactkv_failure": exactkv_failure,
            }
        )

    if alignment_pass_count == 0:
        return build_panel_report(
            panel_status="restricted_no_go",
            blocked_reason="tokenizer or Shard draft alignment failed on all stress prompts",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=prompt_results,
            notes=[
                "No prompt achieved safe ID comparison between Shard draft and HF verifier.",
            ],
            recommendation="restricted_no_go",
        )

    divergence_count = sum(
        1
        for r in prompt_results
        if r.get("comparison") and r["comparison"].get("first_divergence_index") is not None
    )
    exactkv_failures = sum(1 for r in prompt_results if r.get("exactkv_failure"))

    if exactkv_failures > 0:
        recommendation = "restricted_go_verify_harness"
    elif divergence_count > 0:
        recommendation = "restricted_go_with_divergence"
    else:
        recommendation = "restricted_go_no_divergence_in_panel"

    notes = [
        "Shard used as external compressed-KV draft source only.",
        "HF full-KV greedy verifier is authoritative.",
        f"Aligned prompts: {alignment_pass_count}/{len(panel)}.",
        "Draft divergence is reported separately from exactkv_failures.",
        "External Shard README metrics are not ExactKV results.",
    ]
    if divergence_count == 0:
        notes.append(
            "No divergence observed in this bounded panel; stronger/longer/compression-specific settings may be needed."
        )

    return build_panel_report(
        panel_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used=model_name,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        shard_settings=shard_settings,
        tokenizer_alignment_pass=alignment_pass_count == len(panel),
        prompt_results=prompt_results,
        notes=notes,
        recommendation=recommendation,
    )


def run_panel_job(
    *,
    try_run: bool,
    json_out: Path,
    model_name: str,
    max_new_tokens: int,
    draft_len: int,
    device: str,
    dtype: str | None,
    shard_settings: dict[str, Any],
    per_category: int,
    max_prompts: int,
) -> dict[str, Any]:
    repo_path = resolve_shard_repo_path()
    generated_at = datetime.now(timezone.utc).isoformat()

    if repo_path is None:
        report = build_panel_report(
            panel_status="blocked",
            blocked_reason="blocked: Shard repo not provided (set SHARD_REPO_PATH)",
            shard_repo_path_present=False,
            shard_import_success=False,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=[],
            notes=["Export SHARD_REPO_PATH=/path/to/shard clone of krish1905/shard."],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    import_result = try_import_shard(repo_path)
    if not import_result.success:
        report = build_panel_report(
            panel_status="blocked",
            blocked_reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=[],
            notes=[import_result.reason],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    if not try_run:
        planned = build_stress_prompt_panel(per_category=per_category, max_prompts=max_prompts)
        report = build_panel_report(
            panel_status="blocked",
            blocked_reason="blocked: stress panel not executed (pass --try-run)",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=[],
            notes=[
                f"Planned prompt count: {len(planned)}.",
                "Shard import succeeded; model load skipped without --try-run.",
            ],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        report["planned_prompt_count"] = len(planned)
        return report

    _ensure_shard_on_path(repo_path)
    import_result = try_import_shard(repo_path)
    if not import_result.success or import_result.cache_cls is None:
        report = build_panel_report(
            panel_status="blocked",
            blocked_reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            model_used=model_name,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            shard_settings=shard_settings,
            tokenizer_alignment_pass=False,
            prompt_results=[],
            notes=[import_result.reason],
            recommendation="blocked",
        )
        report["generated_at"] = generated_at
        return report

    resolved_dtype = dtype or _default_dtype()
    report = run_stress_panel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        dtype=resolved_dtype,
        device=device,
        cache_cls=import_result.cache_cls,
        enable_llama_fused_attention=import_result.enable_llama_fused_attention,
        shard_settings=shard_settings,
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
    parser = argparse.ArgumentParser(
        description="Experiment 039 — Shard external-drafter stress panel",
    )
    parser.add_argument("--try-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--model", default=os.environ.get("SHARD_PROBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--device", default=os.environ.get("SHARD_PROBE_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.environ.get("SHARD_PROBE_DTYPE"))
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=48)
    parser.add_argument(
        "--stream-bits",
        type=int,
        default=int(os.environ.get("SHARD_STREAM_BITS", DEFAULT_SHARD_SETTINGS["stream_bits"])),
    )
    parser.add_argument(
        "--k-target-cr",
        type=float,
        default=float(os.environ.get("SHARD_K_TARGET_CR", DEFAULT_SHARD_SETTINGS["k_target_cr"])),
    )
    args = parser.parse_args()

    shard_settings = {
        **DEFAULT_SHARD_SETTINGS,
        "stream_bits": args.stream_bits,
        "k_target_cr": args.k_target_cr,
    }

    report = run_panel_job(
        try_run=args.try_run,
        json_out=args.json_out,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
        device=args.device,
        dtype=args.dtype,
        shard_settings=shard_settings,
        per_category=args.per_category,
        max_prompts=args.max_prompts,
    )
    write_json_report(report, args.json_out)

    status = report["panel_status"]
    print(f"Shard stress panel: {status}")
    reason = report.get("blocked_reason") or ""
    if reason:
        print(reason)
    if status == "pass":
        print(
            f"model={report.get('model_used')} "
            f"prompts={report.get('prompt_count')} "
            f"max_new_tokens={report.get('max_new_tokens')} "
            f"divergence_count={report.get('divergence_count')} "
            f"exactkv_failures={report.get('exactkv_failures')}"
        )
        dist = report.get("accepted_prefix_distribution") or {}
        if dist.get("histogram"):
            print(f"accepted_prefix_histogram={dist['histogram']}")
        if report.get("no_divergence_observed"):
            print("no_divergence_observed=true")
        examples = report.get("divergence_examples") or []
        if examples:
            print(f"divergence_examples={len(examples)}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
