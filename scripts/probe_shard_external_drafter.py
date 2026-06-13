#!/usr/bin/env python3
"""Experiment 038: Shard external-drafter feasibility probe (Phase 10B).

Restricted feasibility only — NOT default registry, NOT KVCompressor backend,
NOT production Shard integration. HF full-KV greedy path is authoritative.

No speedup, memory savings, serving, or external benchmark claims.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.external.shard_probe import (  # noqa: E402
    DEFAULT_DRAFT_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    MINIMAL_PROBE_PROMPTS,
    blocked_report,
    build_report,
    compare_token_sequences,
    check_tokenizer_alignment,
    prompt_ids_comparable,
    resolve_shard_repo_path,
    restricted_no_go_report,
    try_import_shard,
)
from exactkv.research.external_drafter_probe import (  # noqa: E402
    run_external_drafter_probe,
    trajectory_token_agreement,
)
from exactkv.runtime.generation import generate_full_greedy  # noqa: E402
from exactkv.runtime.model_runtime import ModelRuntime  # noqa: E402

DEFAULT_JSON = _ROOT / "reports" / "experiment_038_shard_external_drafter_probe.json"


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


def load_shard_draft_ids(
    model_name: str,
    prompt: str,
    *,
    max_new_tokens: int,
    dtype: str,
    cache_cls: Any,
    enable_llama_fused_attention: Any,
) -> tuple[list[int], list[int], str | None]:
    try:
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
        cache = cache_cls.from_model(model)
        cache._streaming = True
        cache._stream_bits = 8

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        prompt_ids = inputs["input_ids"][0].tolist()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                past_key_values=cache,
            )
        draft_ids = out[0, prompt_len:].tolist()[:max_new_tokens]
        return prompt_ids, draft_ids, None
    except Exception as exc:  # noqa: BLE001 — feasibility probe reports blockers
        return [], [], f"Shard draft generation failed: {exc}"


def run_probe_attempt(
    *,
    model_name: str,
    max_new_tokens: int,
    draft_len: int,
    dtype: str,
    cache_cls: Any,
    enable_llama_fused_attention: Any,
    device: str,
) -> dict[str, Any]:
    dep_err = _check_torch_transformers()
    if dep_err:
        return restricted_no_go_report(
            reason=dep_err,
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            notes=[dep_err],
        )

    try:
        verifier_runtime = ModelRuntime(
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    except Exception as exc:  # noqa: BLE001
        return restricted_no_go_report(
            reason=f"verifier model load blocked: {exc}",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            notes=[
                "Llama model access may require HF_TOKEN and Meta license acceptance.",
                str(exc),
            ],
        )

    prompt_results: list[dict[str, Any]] = []
    alignment_pass_count = 0
    exactkv_failures = 0

    for entry in MINIMAL_PROBE_PROMPTS:
        prompt = entry["prompt"]
        hf_full = generate_full_greedy(verifier_runtime, prompt, max_new_tokens)
        verifier_ids = hf_full.generated_ids.squeeze(0).tolist()
        hf_prompt_ids = verifier_runtime.tokenizer.encode(
            prompt, add_special_tokens=False
        )

        shard_prompt_ids, shard_draft_ids, shard_err = load_shard_draft_ids(
            model_name,
            prompt,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            cache_cls=cache_cls,
            enable_llama_fused_attention=enable_llama_fused_attention,
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
        token_alignment_pass = (
            alignment["alignment_pass"]
            and prompt_aligned
            and shard_err is None
        )
        if shard_err:
            token_alignment_pass = False

        comparison: dict[str, Any] | None = None
        external_probe: dict[str, Any] | None = None
        if token_alignment_pass and shard_draft_ids is not None and shard_err is None:
            comparison = compare_token_sequences(verifier_ids, shard_draft_ids)
            traj = trajectory_token_agreement(verifier_ids, shard_draft_ids)
            if not comparison["exact_match"]:
                exactkv_failures += 1
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
            external_probe["trajectory"] = traj
            alignment_pass_count += 1
        else:
            if shard_err:
                alignment["shard_error"] = shard_err
            if not prompt_aligned:
                alignment["shard_prompt_ids"] = shard_prompt_ids

        prompt_results.append(
            {
                "prompt_id": entry["prompt_id"],
                "category": entry["category"],
                "prompt": prompt,
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
            }
        )

    if alignment_pass_count == 0:
        return restricted_no_go_report(
            reason="tokenizer or Shard draft alignment failed on all probe prompts",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            prompt_results=prompt_results,
            notes=[
                "No prompt achieved safe ID-by-ID comparison between Shard draft and HF verifier.",
                "Shard remains external-drafter feasibility only.",
            ],
        )

    recommendation = "restricted_go"
    if exactkv_failures > 0:
        recommendation = "restricted_go_with_divergence"
    elif alignment_pass_count < len(MINIMAL_PROBE_PROMPTS):
        recommendation = "restricted_go_partial_alignment"

    return build_report(
        probe_status="pass",
        blocked_reason="",
        shard_repo_path_present=True,
        shard_import_success=True,
        model_used=model_name,
        tokenizer_alignment_pass=alignment_pass_count == len(MINIMAL_PROBE_PROMPTS),
        prompt_count=len(prompt_results),
        exactkv_failures=exactkv_failures,
        accepted_prefix_lengths=[
            r["comparison"]["accepted_prefix_length"]
            for r in prompt_results
            if r.get("comparison")
        ],
        first_divergence_indices=[
            r["comparison"]["first_divergence_index"]
            for r in prompt_results
            if r.get("comparison")
        ],
        prompt_results=prompt_results,
        notes=[
            "Shard used as external compressed-KV draft source only.",
            "HF full-KV greedy verifier is authoritative.",
            f"Aligned prompts: {alignment_pass_count}/{len(MINIMAL_PROBE_PROMPTS)}.",
            "External Shard README metrics are not ExactKV results.",
        ],
        recommendation=recommendation,
    )


def run_probe(
    *,
    try_run: bool,
    json_out: Path,
    model_name: str,
    max_new_tokens: int,
    draft_len: int,
    device: str,
    dtype: str | None,
) -> dict[str, Any]:
    repo_path = resolve_shard_repo_path()
    generated_at = datetime.now(timezone.utc).isoformat()

    if repo_path is None:
        report = blocked_report(
            reason="blocked: Shard repo not provided (set SHARD_REPO_PATH)",
            shard_repo_path_present=False,
            notes=[
                "Export SHARD_REPO_PATH=/path/to/shard clone of krish1905/shard.",
                "Re-run with --try-run after clone and dependencies are installed.",
            ],
        )
        report["generated_at"] = generated_at
        return report

    import_result = try_import_shard(repo_path)
    if not import_result.success:
        report = blocked_report(
            reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            notes=[import_result.reason],
        )
        report["generated_at"] = generated_at
        return report

    if not try_run:
        report = blocked_report(
            reason="blocked: probe not executed (pass --try-run to load model and compare tokens)",
            shard_repo_path_present=True,
            shard_import_success=True,
            model_used=model_name,
            notes=[
                "Shard import succeeded; model load skipped without --try-run.",
                "Use: SHARD_REPO_PATH=... python3 scripts/probe_shard_external_drafter.py --try-run",
            ],
        )
        report["generated_at"] = generated_at
        return report

    _ensure_shard_on_path(repo_path)
    import_result = try_import_shard(repo_path)
    if not import_result.success or import_result.cache_cls is None:
        report = blocked_report(
            reason=f"blocked: {import_result.reason}",
            shard_repo_path_present=True,
            shard_import_success=False,
            model_used=model_name,
        )
        report["generated_at"] = generated_at
        return report

    resolved_dtype = dtype or _default_dtype()
    report = run_probe_attempt(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        dtype=resolved_dtype,
        cache_cls=import_result.cache_cls,
        enable_llama_fused_attention=import_result.enable_llama_fused_attention,
        device=device,
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
        description="Experiment 038 — Shard external-drafter feasibility probe",
    )
    parser.add_argument(
        "--try-run",
        action="store_true",
        help="Load Llama model and run Shard vs full-KV token comparison",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON,
    )
    parser.add_argument("--model", default=os.environ.get("SHARD_PROBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--draft-len", type=int, default=DEFAULT_DRAFT_LEN)
    parser.add_argument("--device", default=os.environ.get("SHARD_PROBE_DEVICE", "auto"))
    parser.add_argument("--dtype", default=os.environ.get("SHARD_PROBE_DTYPE"))
    args = parser.parse_args()

    report = run_probe(
        try_run=args.try_run,
        json_out=args.json_out,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
        device=args.device,
        dtype=args.dtype,
    )
    write_json_report(report, args.json_out)

    status = report["probe_status"]
    reason = report.get("blocked_reason") or ""
    print(f"Shard external-drafter probe: {status}")
    if reason:
        print(reason)
    if status == "pass":
        print(
            f"model={report.get('model_used')} "
            f"prompts={report.get('prompt_count')} "
            f"exactkv_failures={report.get('exactkv_failures')}"
        )
        lengths = report.get("accepted_prefix_lengths") or []
        divs = report.get("first_divergence_indices") or []
        if lengths:
            print(f"accepted_prefix_lengths={lengths}")
        if divs:
            print(f"first_divergence_indices={divs}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
