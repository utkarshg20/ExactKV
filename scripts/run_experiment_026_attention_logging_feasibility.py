#!/usr/bin/env python3
"""Experiment 026: true attention logging feasibility (V12 Phase 6).

Attention capture is diagnostic only; does not modify ExactKV verification.
No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.analysis.attention_logging import (
    EXP026_COMPRESSORS,
    FALLBACK_ATTENTION_MODEL,
    assert_attention_artifact_safe,
    collect_environment_meta,
    evaluate_feasibility,
    load_exp026_prompt_subset,
    load_probe_model,
    probe_prefill_attention,
    probe_verifier_single_step_attention,
)
from exactkv.compressors import get_compressor
from exactkv.metrics.exactness import token_exact_match
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

MODEL_QWEN = "Qwen/Qwen2.5-0.5B"
EXPERIMENT_CLASS = "v12_attention_logging_feasibility"
MAX_NEW_TOKENS = 8
DRAFT_LEN = 4

_FORBIDDEN = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


def _assert_no_forbidden(obj: Any, path: str = "artifact") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")


def _default_device_dtype() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "float32"


def _run_exactkv_exactness_smoke(
    runtime: ModelRuntime,
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Confirm ExactKV cells still exact on subset; attention probe does not change core."""
    results: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    for prompt_entry in prompts:
        for comp_name in EXP026_COMPRESSORS:
            if comp_name not in cache:
                cache[comp_name] = get_compressor(comp_name)
            comp = cache[comp_name]
            prompt = prompt_entry["prompt"]
            full = generate_full_greedy(runtime, prompt, MAX_NEW_TOKENS)
            ekv = ExactKVGenerator(runtime, comp, draft_len=DRAFT_LEN).generate(
                prompt, MAX_NEW_TOKENS
            )
            ok = token_exact_match(full.generated_ids, ekv.output_ids)
            results.append({
                "prompt_id": prompt_entry["prompt_id"],
                "compressor_name": comp_name,
                "exactkv_failure": not ok,
            })
    failures = sum(1 for r in results if r["exactkv_failure"])
    return {
        "cells": len(results),
        "exactkv_failures": failures,
        "results": results,
    }


def run_feasibility(
    *,
    device: str,
    dtype: str,
    include_fallback: bool,
) -> dict[str, Any]:
    prompts = load_exp026_prompt_subset()
    env = collect_environment_meta()
    env["device_requested"] = device
    env["dtype_requested"] = dtype

    attempts: list[dict[str, Any]] = []
    probe_prompt = prompts[0]["prompt"]
    probe_prompt_id = prompts[0]["prompt_id"]

    # --- A: Qwen default backend ---
    print("Attempt A: Qwen default backend ...", flush=True)
    try:
        model_a, tok_a, dev_a, impl_a = load_probe_model(
            MODEL_QWEN, device=device, dtype=dtype, attn_implementation=None
        )
        rec = probe_prefill_attention(
            model_a,
            tok_a,
            dev_a,
            probe_prompt,
            attempt_label="A_qwen_default_prefill",
            attn_implementation=impl_a,
        )
        rec["model_name"] = MODEL_QWEN
        rec["prompt_id"] = probe_prompt_id
        attempts.append(rec)
        del model_a
    except Exception as exc:
        attempts.append({
            "attempt": "A_qwen_default_prefill",
            "model_name": MODEL_QWEN,
            "weights_obtained": False,
            "error": f"model_load_failed: {exc}",
        })

    # --- B: Qwen eager attention ---
    print("Attempt B: Qwen eager attention ...", flush=True)
    try:
        model_b, tok_b, dev_b, impl_b = load_probe_model(
            MODEL_QWEN,
            device=device,
            dtype=dtype,
            attn_implementation="eager",
        )
        rec = probe_prefill_attention(
            model_b,
            tok_b,
            dev_b,
            probe_prompt,
            attempt_label="B_qwen_eager_prefill",
            attn_implementation=impl_b,
        )
        rec["model_name"] = MODEL_QWEN
        rec["prompt_id"] = probe_prompt_id
        attempts.append(rec)
        del model_b
    except Exception as exc:
        attempts.append({
            "attempt": "B_qwen_eager_prefill",
            "model_name": MODEL_QWEN,
            "weights_obtained": False,
            "error": f"model_load_failed: {exc}",
        })

    # --- C: Prefill-only + verifier single-step via ModelRuntime (eager if B worked) ---
    print("Attempt C: verifier single-step attention ...", flush=True)
    eager_for_runtime = any(
        a.get("attempt") == "B_qwen_eager_prefill" and a.get("weights_obtained")
        for a in attempts
    )
    try:
        # ModelRuntime lacks attn_implementation; load eager model separately for C
        # using direct probe on all 6 prompts if eager prefill succeeded.
        runtime = ModelRuntime(MODEL_QWEN, device=device, dtype=dtype)
        for pe in prompts:
            rec = probe_verifier_single_step_attention(
                runtime,
                pe["prompt"],
                attempt_label="C_qwen_verifier_single_step_default_runtime",
            )
            rec["model_name"] = MODEL_QWEN
            rec["prompt_id"] = pe["prompt_id"]
            rec["note"] = (
                "ModelRuntime default load; may use sdpa without output_attentions"
            )
            attempts.append(rec)
    except Exception as exc:
        attempts.append({
            "attempt": "C_qwen_verifier_single_step",
            "model_name": MODEL_QWEN,
            "weights_obtained": False,
            "error": f"runtime_probe_failed: {exc}",
        })

    # C2: eager prefill on all prompts if B succeeded
    if eager_for_runtime:
        print("Attempt C2: Qwen eager prefill on full 6-prompt panel ...", flush=True)
        try:
            model_c2, tok_c2, dev_c2, impl_c2 = load_probe_model(
                MODEL_QWEN,
                device=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            for pe in prompts:
                rec = probe_prefill_attention(
                    model_c2,
                    tok_c2,
                    dev_c2,
                    pe["prompt"],
                    attempt_label="C2_qwen_eager_prefill_panel",
                    attn_implementation=impl_c2,
                )
                rec["model_name"] = MODEL_QWEN
                rec["prompt_id"] = pe["prompt_id"]
                attempts.append(rec)
            del model_c2
        except Exception as exc:
            attempts.append({
                "attempt": "C2_qwen_eager_prefill_panel",
                "weights_obtained": False,
                "error": str(exc),
            })

    # --- D: Fallback model plumbing check ---
    if include_fallback:
        print(f"Attempt D: fallback model {FALLBACK_ATTENTION_MODEL} ...", flush=True)
        try:
            model_d, tok_d, dev_d, impl_d = load_probe_model(
                FALLBACK_ATTENTION_MODEL,
                device=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            rec = probe_prefill_attention(
                model_d,
                tok_d,
                dev_d,
                "Hello world",
                attempt_label="D_fallback_gpt2_eager_plumbing",
                attn_implementation=impl_d,
            )
            rec["model_name"] = FALLBACK_ATTENTION_MODEL
            rec["prompt_id"] = "plumbing_only"
            rec["note"] = (
                "Attention plumbing check only; not an ExactKV/Qwen result"
            )
            attempts.append(rec)
            del model_d
        except Exception as exc:
            attempts.append({
                "attempt": "D_fallback_gpt2_eager_plumbing",
                "model_name": FALLBACK_ATTENTION_MODEL,
                "weights_obtained": False,
                "error": str(exc),
                "note": "Plumbing check only",
            })

    # Exactness smoke (core unchanged)
    print("ExactKV exactness smoke on panel ...", flush=True)
    runtime_smoke = ModelRuntime(MODEL_QWEN, device=device, dtype=dtype)
    exactness = _run_exactkv_exactness_smoke(runtime_smoke, prompts)

    verdict = evaluate_feasibility(attempts)
    return {
        "manifest": {
            "experiment": "026_attention_logging_feasibility",
            "experiment_class": EXPERIMENT_CLASS,
            "artifact_type": "attention_feasibility",
            "models_tested": list({
                a.get("model_name", MODEL_QWEN) for a in attempts
            }),
            "prompt_count": len(prompts),
            "compressors_in_exactness_smoke": list(EXP026_COMPRESSORS),
            "environment": env,
        },
        "prompts": [
            {"prompt_id": p["prompt_id"], "v10_suite": p["v10_suite"]}
            for p in prompts
        ],
        "attempts": attempts,
        "exactness_smoke": exactness,
        "verdict": verdict,
        "note": (
            "Attention logging feasibility only. Weights are not fabricated. "
            "Diagnostic use only; does not change ExactKV verification."
        ),
    }


def _fmt_bool(v: bool | None) -> str:
    if v is None:
        return "—"
    return "yes" if v else "no"


def generate_markdown_report(artifact: dict[str, Any]) -> str:
    manifest = artifact["manifest"]
    env = manifest["environment"]
    verdict = artifact["verdict"]
    attempts = artifact["attempts"]
    exactness = artifact["exactness_smoke"]

    lines = [
        "# Experiment 026: True Attention Logging Feasibility",
        "",
        "_Generated by "
        "`scripts/run_experiment_026_attention_logging_feasibility.py`. "
        "V12 Phase 6 — attention logging feasibility only._",
        "",
        "> This is **attention logging feasibility only**.",
        "> **Attention weights are not fabricated.** If absent, this report states so.",
        "> This does **not** change ExactKV verification.",
        "> ExactKV does **not** claim speed, memory, latency, throughput, runtime, "
        "tokens/sec, active GPU memory, production serving, or model accuracy improvement.",
        "> Attention findings, if any, are **diagnostic only**.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Determine whether true attention weights can be logged safely for a tiny "
        "ExactKV divergence-forensics subset, or document a precise no-go.",
        "",
        "## 2. Why this follows Experiments 019 and 025",
        "",
        "Experiment 019 deferred true attention logging (sdpa / `output_attentions` "
        "blocker). Experiment 025 showed repair policies on the full suite but did not "
        "resolve per-head forensics. Phase 6 tests eager, prefill-only, and verifier "
        "paths without changing ExactKV semantics.",
        "",
        "## 3. Environment",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| torch | {env.get('torch_version', '—')} |",
        f"| transformers | {env.get('transformers_version', '—')} |",
        f"| CUDA available | {env.get('cuda_available', '—')} |",
        f"| GPU | {env.get('gpu_device_name', '— (CPU)')} |",
        f"| device / dtype | {env.get('device_requested', '—')} / "
        f"{env.get('dtype_requested', '—')} |",
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 "
        "scripts/run_experiment_026_attention_logging_feasibility.py",
        "```",
        "",
        "## 4. Models/backends tested",
        "",
        "| Model | Backend / attempt |",
        "|---|---|",
    ]
    seen: set[str] = set()
    for a in attempts:
        key = f"{a.get('model_name', '?')}:{a.get('attempt', '?')}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| `{a.get('model_name', '—')}` | `{a.get('attempt', '—')}` "
            f"({a.get('attn_implementation', '—')}) |"
        )

    lines.extend([
        "",
        "## 5. Prompt subset",
        "",
        "| Prompt ID | Suite |",
        "|---|---|",
    ])
    for p in artifact["prompts"]:
        lines.append(f"| `{p['prompt_id']}` | `{p['v10_suite']}` |")

    lines.extend([
        "",
        "## 6. Attention logging attempts",
        "",
        "| Attempt | Weights? | Phase | Error / note |",
        "|---|---|---|---|",
    ])
    for a in attempts:
        err = a.get("error") or a.get("note") or "—"
        lines.append(
            f"| `{a.get('attempt', '—')}` | {_fmt_bool(a.get('weights_obtained'))} | "
            f"{a.get('phase', '—')} | {err} |"
        )

    any_weights = verdict.get("any_weights_obtained", False)
    lines.extend([
        "",
        "## 7. Whether true attention weights were obtained",
        "",
        f"**{'Yes (at least one attempt)' if any_weights else 'No'}** — "
        f"see attempt table above.",
        "",
    ])

    if any_weights:
        lines.extend([
            "## 8. Attention tensor shapes and summaries",
            "",
        ])
        for a in attempts:
            if not a.get("weights_obtained"):
                continue
            s = a.get("attention_summary") or {}
            lt = s.get("last_token_summary") or {}
            lines.append(f"### `{a.get('attempt')}` — `{a.get('prompt_id', '—')}`")
            lines.append("")
            lines.append(f"- Layers: {s.get('num_layers', '—')}")
            lines.append(f"- Last layer shape: `{s.get('last_layer_shape', '—')}`")
            lines.append(f"- Heads (last layer): {s.get('num_heads_last_layer', '—')}")
            lines.append(
                f"- Last-token mass to recent {lt.get('recent_k', '?')} tokens: "
                f"{lt.get('mass_to_recent_tokens', '—')}"
            )
            lines.append(
                f"- Last-token mass to early {lt.get('early_k', '?')} tokens: "
                f"{lt.get('mass_to_early_tokens', '—')}"
            )
            lines.append(f"- Entropy: {lt.get('entropy', '—')}")
            lines.append("")
    else:
        lines.extend([
            "## 8. Attention tensor shapes and summaries",
            "",
            "_No attention weights obtained; section intentionally empty._",
            "",
        ])

    lines.extend([
        "## 9. Blocker table (if applicable)",
        "",
        "| Attempt | Blocker |",
        "|---|---|",
    ])
    for a in attempts:
        if a.get("weights_obtained"):
            lines.append(f"| `{a.get('attempt')}` | — (success) |")
        else:
            lines.append(
                f"| `{a.get('attempt')}` | {a.get('error', 'unknown')} |"
            )

    lines.extend([
        "",
        "## 10. Whether attention can be used in future divergence analysis",
        "",
    ])
    if verdict.get("qwen_prefill_weights"):
        lines.append(
            "Yes — **restricted prefill-only** snapshots on Qwen with eager attention "
            "can supplement logit/KV-layer autopsy. Not for acceptance decisions."
        )
    elif verdict.get("any_weights_obtained"):
        lines.append(
            "Only on fallback plumbing model; **not** for primary Qwen ExactKV forensics "
            "until Qwen path unblocks."
        )
    else:
        lines.append(
            "**No** — defer per-head divergence analysis; continue logit and per-layer KV "
            "error forensics only."
        )

    lines.extend([
        "",
        "## 11. What this proves",
        "",
        "- Whether `output_attentions=True` is viable on the primary model/backends.",
        "- ExactKV exactness smoke on the 6-prompt × 4-compressor panel remains intact "
        f"(`exactkv_failures == {exactness.get('exactkv_failures', '—')}`).",
        "",
        "## 12. What this does not prove",
        "",
        "- Production attention-gated compression or Sparse V.",
        "- Speed, memory, or accuracy improvement.",
        "- Full decode-trace attention for every ExactKV verification round.",
        "",
        "## 13. Limitations",
        "",
        "- Tiny 6-prompt panel only.",
        "- Prefill or single verifier step — not full compressed-KV draft attention.",
        "- ModelRuntime does not expose `attn_implementation`; eager tests use separate loads.",
        "",
        "## 14. Go/no-go recommendation",
        "",
        f"**`{verdict.get('recommendation', '—')}`**",
        "",
        verdict.get("detail", ""),
        "",
        "## 15. VeriCache attribution",
        "",
        "ExactKV draft-verify-commit loop inspired by "
        "[VeriCache](https://arxiv.org/abs/2605.17613). Attention logging is an "
        "optional diagnostic layer only; it does not change the verification algorithm.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 026 attention feasibility")
    default_device, default_dtype = _default_device_dtype()
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--dtype", default=default_dtype)
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=_ROOT / "reports")
    parser.add_argument("--docs-dir", type=Path, default=_ROOT / "docs")
    args = parser.parse_args()

    artifact = run_feasibility(
        device=args.device,
        dtype=args.dtype,
        include_fallback=not args.skip_fallback,
    )
    assert_attention_artifact_safe(artifact)
    _assert_no_forbidden(artifact)

    json_path = args.reports_dir / "experiment_026_attention_logging_feasibility.json"
    md_path = args.docs_dir / "EXPERIMENT_026_ATTENTION_LOGGING_FEASIBILITY.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    md_path.write_text(generate_markdown_report(artifact), encoding="utf-8")

    v = artifact["verdict"]
    print(f"\nExperiment 026 complete", flush=True)
    print(f"  any_weights_obtained: {v.get('any_weights_obtained')}", flush=True)
    print(f"  recommendation: {v.get('recommendation')}", flush=True)
    print(f"  exactkv_failures (smoke): {artifact['exactness_smoke']['exactkv_failures']}")
    print(f"  JSON: {json_path}", flush=True)
    print(f"  Report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
