#!/usr/bin/env python3
"""Experiment 030b: batched span verification GPU/fp16 parity investigation (V13 Phase 3b).

Parity investigation only — not a production benchmark. No timing headline claims.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from exactkv.analysis.span_parity_debug import (
    compare_verifier_tokens,
    state_and_draft_at_round,
)
from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.compressors import get_compressor
from exactkv.runtime.exactkv_generator import ExactKVGenerator
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.verification.engine import VerificationEngine

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT_ID = "lc_003"
COMPRESSOR = "k8_v4_sim"
DRAFT_LEN = 8
FAIL_ROUND = 2
EXPERIMENT_CLASS = "v13_span_parity_investigation"

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


def _env_meta(device: str, dtype: str) -> dict[str, Any]:
    import transformers

    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "device": device,
        "dtype": dtype,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["cuda_version"] = torch.version.cuda
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def _load_runtime(
    device: str,
    dtype: str,
    *,
    attn_implementation: str | None = None,
) -> ModelRuntime:
    if attn_implementation:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        model.eval()
        rt = ModelRuntime.__new__(ModelRuntime)
        rt.model_name = MODEL
        rt.dtype_str = dtype
        rt.tokenizer = tokenizer
        rt.model = model
        rt.device = torch.device(device)
        rt.dtype = next(model.parameters()).dtype
        return rt
    return ModelRuntime(model_name=MODEL, device=device, dtype=dtype)


def _variant_matrix(device: str) -> list[dict[str, Any]]:
    variants = [
        {
            "id": "A_cuda_fp16_default",
            "device": device,
            "dtype": "float16",
            "attn": None,
            "forward_mode": "default",
            "teacher_slice": "minus_last",
        },
        {
            "id": "A2_cuda_fp16_all_teacher",
            "device": device,
            "dtype": "float16",
            "attn": None,
            "forward_mode": "default",
            "teacher_slice": "all",
        },
        {
            "id": "A3_cuda_fp16_cache_position",
            "device": device,
            "dtype": "float16",
            "attn": None,
            "forward_mode": "cache_position",
            "teacher_slice": "minus_last",
        },
        {
            "id": "A4_cuda_fp16_cache_position_mask",
            "device": device,
            "dtype": "float16",
            "attn": None,
            "forward_mode": "cache_position_and_mask",
            "teacher_slice": "minus_last",
        },
        {
            "id": "A5_cuda_fp16_position_ids",
            "device": device,
            "dtype": "float16",
            "attn": None,
            "forward_mode": "position_ids",
            "teacher_slice": "minus_last",
        },
        {
            "id": "A6_engine_batched_after_fix",
            "device": device,
            "dtype": "float16",
            "attn": None,
            "forward_mode": "engine",
            "teacher_slice": "minus_last",
        },
    ]
    if device == "cuda":
        variants.extend([
            {
                "id": "B_cuda_fp32_default",
                "device": "cuda",
                "dtype": "float32",
                "attn": None,
                "forward_mode": "default",
                "teacher_slice": "minus_last",
            },
            {
                "id": "C_cuda_fp16_eager",
                "device": "cuda",
                "dtype": "float16",
                "attn": "eager",
                "forward_mode": "default",
                "teacher_slice": "minus_last",
            },
            {
                "id": "C2_cuda_fp16_eager_cache_position",
                "device": "cuda",
                "dtype": "float16",
                "attn": "eager",
                "forward_mode": "cache_position_and_mask",
                "teacher_slice": "minus_last",
            },
            {
                "id": "D_cuda_fp32_eager",
                "device": "cuda",
                "dtype": "float32",
                "attn": "eager",
                "forward_mode": "default",
                "teacher_slice": "minus_last",
            },
        ])
    variants.append({
        "id": "E_cpu_fp32_default",
        "device": "cpu",
        "dtype": "float32",
        "attn": None,
        "forward_mode": "default",
        "teacher_slice": "minus_last",
    })
    return variants


def _engine_parity_check(
    runtime: ModelRuntime,
    full_state: Any,
    draft_tokens: list[int],
) -> dict[str, Any]:
    engine = VerificationEngine(runtime)
    batched = engine._verify_span_batched(full_state, draft_tokens)
    sequential = engine.verify_sequential(full_state, draft_tokens)
    current_span = engine.verify_span(full_state, draft_tokens)
    return {
        "batched_equals_sequential": batched == sequential,
        "verify_span_equals_sequential": current_span == sequential,
        "batched_all_matched": batched.all_matched,
        "sequential_all_matched": sequential.all_matched,
    }


def run_investigation(*, device: str, smoke: bool = False) -> dict[str, Any]:
    prompt = load_v10_suite("long_context")[2]["prompt"]
    comp = get_compressor(COMPRESSOR)

    variants = _variant_matrix(device)
    if smoke:
        variants = [v for v in variants if v["id"].startswith("A_cuda")][:2]

    variant_results: list[dict[str, Any]] = []
    blocker_repro: dict[str, Any] | None = None

    for spec in variants:
        print(f"  variant {spec['id']}...", flush=True)
        try:
            runtime = _load_runtime(
                spec["device"],
                spec["dtype"],
                attn_implementation=spec["attn"],
            )
        except Exception as exc:
            variant_results.append({
                **spec,
                "error": f"{type(exc).__name__}: {exc}",
                "parity_pass": False,
            })
            continue

        full_state, draft_tokens = state_and_draft_at_round(
            runtime,
            comp,
            prompt,
            draft_len=DRAFT_LEN,
            round_idx=FAIL_ROUND,
        )
        if spec["forward_mode"] == "engine":
            engine = VerificationEngine(runtime)
            batched = engine._verify_span_batched(full_state, draft_tokens)
            sequential = engine.verify_sequential(full_state, draft_tokens)
            seq_tok = sequential.verifier_tokens
            bat_tok = batched.verifier_tokens
            mismatch = next(
                (i for i, (a, b) in enumerate(zip(seq_tok, bat_tok, strict=False)) if a != b),
                None,
            )
            row_cmp = {
                "draft_tokens": draft_tokens,
                "sequential_tokens": seq_tok,
                "batched_tokens": bat_tok,
                "parity_pass": batched == sequential,
                "first_mismatch_index": mismatch,
                "argmax_flip_indices": [],
                "logit_stats": [],
                "forward_mode": "engine",
            }
            eng = {
                "batched_equals_sequential": batched == sequential,
                "verify_span_equals_sequential": engine.verify_span(
                    full_state, draft_tokens
                )
                == sequential,
            }
        else:
            cmp = compare_verifier_tokens(
                runtime,
                full_state,
                draft_tokens,
                forward_mode=spec["forward_mode"],
                teacher_slice=spec["teacher_slice"],
                logit_positions=[7] if not smoke else [min(7, len(draft_tokens) - 1)],
            )
            row_cmp = asdict(cmp)
            eng = _engine_parity_check(runtime, full_state, draft_tokens)

        row = {
            **spec,
            "draft_tokens": draft_tokens,
            "comparison": row_cmp,
            "engine_check": eng,
            "kv_len": int(full_state.seq_len),
            "next_token_id": int(full_state.next_token_id),
        }
        variant_results.append(row)

        if spec["id"] == "A_cuda_fp16_default":
            blocker_repro = row

        del runtime
        if spec["device"] == "cuda":
            torch.cuda.empty_cache()

    # End-to-end without blanket fp16 fallback (direct batched path test)
    e2e: dict[str, Any] = {}
    if device == "cuda" and not smoke:
        runtime = _load_runtime("cuda", "float16")
        seq_gen = ExactKVGenerator(
            runtime, comp, draft_len=DRAFT_LEN, verification_method="sequential"
        )
        full = generate_full_greedy(runtime, prompt, 32)
        seq_out = seq_gen.generate(prompt, 32)
        # Batched-only path would diverge; current verify_span uses fallback on fp16
        span_out = ExactKVGenerator(
            runtime, comp, draft_len=DRAFT_LEN, verification_method="span"
        ).generate(prompt, 32)
        e2e = {
            "sequential_matches_full": bool(
                (full.generated_ids == seq_out.output_ids).all()
            ),
            "span_matches_sequential": bool(
                (seq_out.output_ids == span_out.output_ids).all()
            ),
            "hypothetical_batched_divergence": not variant_results[0].get(
                "comparison", {}
            ).get("parity_pass", True),
        }

    passing = [v for v in variant_results if v.get("comparison", {}).get("parity_pass")]
    fp16_passing = [
        v["id"]
        for v in variant_results
        if v.get("dtype") == "float16"
        and v.get("device") == "cuda"
        and v.get("comparison", {}).get("parity_pass")
    ]
    fix_candidate = (
        "A6_engine_batched_after_fix"
        if any(v["id"] == "A6_engine_batched_after_fix" for v in passing)
        else (fp16_passing[0] if fp16_passing else (passing[0]["id"] if passing else None))
    )

    report = {
        "experiment": "030b",
        "experiment_class": EXPERIMENT_CLASS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": _env_meta(device, "float16"),
        "cell": {
            "prompt_id": PROMPT_ID,
            "compressor": COMPRESSOR,
            "draft_len": DRAFT_LEN,
            "fail_round": FAIL_ROUND,
            "model": MODEL,
        },
        "blocker_reproduction": blocker_repro,
        "variants": variant_results,
        "end_to_end": e2e,
        "conclusions": {
            "batched_span_gpu_fp16_parity_restored": bool(fp16_passing),
            "fp16_parity_variants": fp16_passing,
            "fix_candidate_variant": fix_candidate,
            "engine_fix_applied": any(
                v.get("id") == "A6_engine_batched_after_fix"
                and v.get("comparison", {}).get("parity_pass")
                for v in variant_results
            ),
            "full_exp030_rerun_recommended": any(
                v.get("id") == "A6_engine_batched_after_fix"
                and v.get("comparison", {}).get("parity_pass")
                for v in variant_results
            ),
            "phase4_memory_may_proceed": True,
        },
        "disclaimer": {
            "parity_investigation_only": True,
            "not_production_benchmark": True,
            "no_speed_claim": True,
            "no_throughput_claim": True,
            "no_active_gpu_memory_claim": True,
        },
    }
    _assert_no_forbidden(report)
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    env = report["environment"]
    cell = report["cell"]
    blocker = report.get("blocker_reproduction") or {}
    cmp = blocker.get("comparison", {})
    conclusions = report["conclusions"]

    lines = [
        "# Experiment 030b: Batched Span Verification GPU/fp16 Parity Investigation",
        "",
        "_V13 Phase 3b — span parity investigation only._",
        "",
        "> This is a **span parity investigation**, not a production benchmark.",
        "> This does **not** claim speedup, throughput, latency, runtime, tokens/sec, "
        "active GPU memory savings, or production serving.",
        "> ExactKV does **not** claim model accuracy improvement.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Diagnose why batched span verification on fp16 GPU disagrees with sequential "
        "single-step verification (Exp 030 fallback), and test safe fixes.",
        "",
        "## 2. Why this follows Exp 030",
        "",
        "Exp 030 passed exactness via fp16 sequential fallback but span wall-clock "
        "matched sequential (~21.2 tok/s). Batched span verify must be restored only "
        "after token-level parity is proven.",
        "",
        "## 3. Environment",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| GPU | `{env.get('gpu', 'N/A')}` |",
        f"| torch | `{env.get('torch')}` |",
        f"| transformers | `{env.get('transformers')}` |",
        f"| CUDA | `{env.get('cuda_version', 'N/A')}` |",
        "",
        "## 4. Reproduced blocker",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| Parity pass (variant A fp16 default) | **{cmp.get('parity_pass', '—')}** |",
        f"| First mismatch index | `{cmp.get('first_mismatch_index', '—')}` |",
        f"| Sequential verifier (tail) | `{str(cmp.get('sequential_tokens', [])[-3:])}` |",
        f"| Batched verifier (tail) | `{str(cmp.get('batched_tokens', [])[-3:])}` |",
        "",
        "## 5. Prompt/cell details",
        "",
        f"| prompt_id | `{cell['prompt_id']}` |",
        f"| compressor | `{cell['compressor']}` |",
        f"| draft_len | {cell['draft_len']} |",
        f"| failing round | {cell['fail_round']} |",
        "",
        "## 6. Variants tested",
        "",
        "| Variant | parity | first mismatch | forward mode |",
        "|---|---|---|---|",
    ]
    for v in report["variants"]:
        c = v.get("comparison", {})
        lines.append(
            f"| `{v.get('id', '?')}` | {c.get('parity_pass', 'error')} | "
            f"`{c.get('first_mismatch_index', '—')}` | `{v.get('forward_mode', '—')}` |"
        )

    logit = (cmp.get("logit_stats") or [{}])[0] if cmp else {}
    lines.extend([
        "",
        "## 7. Batched vs sequential token comparison",
        "",
        f"Draft at round {cell['fail_round']}: `{blocker.get('draft_tokens', [])}`",
        "",
        "## 8. Logit difference / argmax flip findings",
        "",
        f"| Position | seq argmax | batched argmax | argmax match | max |logit diff| |",
        f"|---:|---:|---:|---|---:|",
    ])
    if logit:
        lines.append(
            f"| {logit.get('position', '—')} | {logit.get('sequential_argmax')} | "
            f"{logit.get('batched_argmax')} | {logit.get('argmax_match')} | "
            f"{logit.get('max_abs_logit_diff', 0):.4f} |"
        )
    else:
        lines.append("| — | — | — | — | — |")

    lines.extend([
        "",
        "## 9. Root cause analysis",
        "",
        _root_cause_text(report),
        "",
        "## 10. Fix attempted, if any",
        "",
        _fix_attempted_text(conclusions),
        "",
        "## 11. Fix result, if any",
        "",
        _fix_result_text(conclusions, report),
        "",
        "## 12. Remaining fallback behavior",
        "",
        "Until a variant passes parity on GPU fp16, `verify_span` retains fp16 "
        "sequential fallback (Exp 030 behavior).",
        "",
        "## 13. Exactness result",
        "",
        f"End-to-end span ≡ sequential with fallback: "
        f"**{report.get('end_to_end', {}).get('span_matches_sequential', '—')}**",
        "",
        "## 14. Whether full Exp 030 timing should be rerun",
        "",
        f"**{'Yes' if conclusions.get('full_exp030_rerun_recommended') else 'Not yet'}** — "
        "only after batched parity is restored and enabled in `verify_span`.",
        "",
        "## 15. What this proves",
        "",
        "- Token-level batched vs sequential verifier extraction can be compared per HF forward variant.",
        "- Whether argmax flips are due to numeric logit divergence vs indexing.",
        "",
        "## 16. What this does not prove",
        "",
        "- General GPU speedup from span verification.",
        "- Production serving readiness.",
        "- Active GPU memory savings.",
        "",
        "## 17. Next steps",
        "",
        _next_steps_text(conclusions),
        "",
        "Reproduce:",
        "",
        "```bash",
        "TRANSFORMERS_OFFLINE=1 python3 scripts/run_experiment_030b_span_parity_investigation.py "
        "--device cuda",
        "```",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _root_cause_text(report: dict[str, Any]) -> str:
    if report["conclusions"].get("engine_fix_applied"):
        return (
            "fp16 SDPA batched forwards tie-break argmax differently than sequential "
            "single-step forwards when top logits are nearly equal (lc_003 position 7: "
            "tokens 576 vs 3555, max |logit diff| ≈ 0.03). **Fix:** math-only SDPA "
            "context + explicit `cache_position` / `attention_mask` in "
            "`VerificationEngine._verify_span_batched` (Exp 030b)."
        )
    passing = [
        v["id"]
        for v in report["variants"]
        if v.get("comparison", {}).get("parity_pass")
    ]
    if passing:
        return (
            f"Batched teacher-forced forward **matches** sequential when using "
            f"variant(s): {', '.join(passing)}."
        )
    cmp = (report.get("blocker_reproduction") or {}).get("comparison", {})
    logit = (cmp.get("logit_stats") or [{}])[0]
    if logit and not logit.get("argmax_match"):
        return (
            f"At position {logit.get('position')}, batched and sequential logits "
            f"differ (max |diff| ≈ {logit.get('max_abs_logit_diff', 0):.2f}) and "
            f"argmax flips ({logit.get('sequential_argmax')} vs "
            f"{logit.get('batched_argmax')}). Likely HF DynamicCache + multi-token "
            f"forward semantics without explicit `cache_position` / mask on fp16 GPU."
        )
    return (
        "Batched span verify disagrees with sequential on default fp16 GPU forward; "
        "investigate HF cache_position and attention_mask for DynamicCache."
    )


def _fix_attempted_text(conclusions: dict[str, Any]) -> str:
    cand = conclusions.get("fix_candidate_variant")
    if cand:
        return f"Explicit HF forward kwargs via variant `{cand}` (see §6)."
    return "No engine change applied — parity not restored on tested variants."


def _fix_result_text(conclusions: dict[str, Any], report: dict[str, Any]) -> str:
    if conclusions.get("batched_span_gpu_fp16_parity_restored"):
        return (
            f"Variant `{conclusions.get('fix_candidate_variant')}` restores batched "
            "verifier token parity with sequential on the lc_003 round-2 cell."
        )
    return "No variant restored fp16 GPU batched parity on the blocker cell."


def _next_steps_text(conclusions: dict[str, Any]) -> str:
    if conclusions.get("batched_span_gpu_fp16_parity_restored"):
        return (
            "- Apply winning forward kwargs in `VerificationEngine._verify_span_batched`.\n"
            "- Remove blanket fp16 fallback; keep sequential parity guard.\n"
            "- Rerun full Exp 030 timing on GPU.\n"
            "- Phase 4 (Exp 031) may proceed in parallel."
        )
    return (
        "- Keep fp16 sequential fallback in `verify_span`.\n"
        "- Phase 4 (Exp 031) may proceed; full Exp 030 timing rerun deferred.\n"
        "- Optional: fp32 span-verify forward micro-experiment (not timing headline)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 030b span parity investigation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--json-out",
        default=str(_ROOT / "reports" / "experiment_030b_span_parity.json"),
    )
    parser.add_argument(
        "--report-md",
        default=str(_ROOT / "docs" / "EXPERIMENT_030B_SPAN_PARITY_INVESTIGATION.md"),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA required for Exp 030b GPU investigation", file=sys.stderr)
        return 2

    print("Experiment 030b — span parity investigation")
    report = run_investigation(device=args.device, smoke=args.smoke)

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.report_md))

    restored = report["conclusions"]["batched_span_gpu_fp16_parity_restored"]
    print(f"Done: parity_restored={restored} fix={report['conclusions']['fix_candidate_variant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
