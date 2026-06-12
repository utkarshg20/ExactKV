"""Attention logging feasibility helpers for Experiment 026 (V12 Phase 6).

Prefill-only or auxiliary forward passes for diagnostic attention capture.
Does **not** modify ExactKV generation or verification logic.
Does **not** fabricate attention weights.
No timing, throughput, latency, speedup, or active_gpu_kv_bytes fields.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from exactkv.benchmarks.v10_prompts import load_v10_suite
from exactkv.runtime.model_runtime import ModelRuntime, _hf_dtype_kwarg
from exactkv.runtime.prefill import prefill_to_full_state

FORBIDDEN_ATTENTION_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

EXP026_SUITES = (
    ("long_context", 2),
    ("retrieval_copy", 2),
    ("tool_json", 2),
)

EXP026_COMPRESSORS = (
    "noop",
    "int8",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
)

# Plumbing check only — not an ExactKV/Qwen production result.
FALLBACK_ATTENTION_MODEL = "gpt2"


def assert_attention_artifact_safe(obj: Any, path: str = "artifact") -> None:
    if isinstance(obj, dict):
        hits = FORBIDDEN_ATTENTION_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            assert_attention_artifact_safe(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_attention_artifact_safe(item, f"{path}[{i}]")


def load_exp026_prompt_subset() -> list[dict[str, Any]]:
    """Deterministic 6-prompt panel: 2× long_context, 2× retrieval_copy, 2× tool_json."""
    out: list[dict[str, Any]] = []
    for suite, n in EXP026_SUITES:
        rows = load_v10_suite(suite)
        rows.sort(key=lambda r: r["prompt_id"])
        for row in rows[:n]:
            entry = dict(row)
            entry["v10_panel"] = "exp026_attention"
            out.append(entry)
    return out


def attention_tensor_shape(attn: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in attn.shape)


def summarize_last_token_attention(
    attn: torch.Tensor,
    *,
    recent_k: int = 4,
    early_k: int = 4,
) -> dict[str, Any]:
    """Summarize last-query-row attention on a single layer tensor.

    Expected prefill shape: ``[batch, heads, seq, seq]`` or ``[batch, seq, seq]``.
    Returns diagnostic scalars only — not used for acceptance decisions.
    """
    t = attn.detach().float()
    if t.dim() == 4:
        # Average over heads for a single summary vector.
        row = t[0].mean(dim=0)[-1]
    elif t.dim() == 3:
        row = t[0][-1]
    else:
        raise ValueError(f"Unexpected attention rank {t.dim()}")

    row = row.clamp(min=0.0)
    total = float(row.sum().item())
    if total <= 0:
        return {
            "seq_len": int(row.numel()),
            "total_mass": total,
            "recent_k": recent_k,
            "early_k": early_k,
            "mass_to_recent_tokens": 0.0,
            "mass_to_early_tokens": 0.0,
            "entropy": None,
        }

    probs = row / row.sum()
    seq_len = int(probs.numel())
    rk = min(recent_k, seq_len)
    ek = min(early_k, seq_len)
    recent_mass = float(probs[-rk:].sum().item())
    early_mass = float(probs[:ek].sum().item())
    entropy = 0.0
    for p in probs.tolist():
        if p > 1e-12:
            entropy -= p * math.log(p)
    return {
        "seq_len": seq_len,
        "total_mass": total,
        "recent_k": rk,
        "early_k": ek,
        "mass_to_recent_tokens": recent_mass,
        "mass_to_early_tokens": early_mass,
        "entropy": entropy,
    }


def _describe_attention_stack(attentions: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    layers = len(attentions)
    first = attentions[0]
    last = attentions[-1]
    first_shape = attention_tensor_shape(first)
    last_shape = attention_tensor_shape(last)
    num_heads = None
    if len(last_shape) == 4:
        num_heads = last_shape[1]
    last_summary = summarize_last_token_attention(last)
    return {
        "num_layers": layers,
        "first_layer_shape": first_shape,
        "last_layer_shape": last_shape,
        "num_heads_last_layer": num_heads,
        "last_layer_mean": float(last.detach().float().mean().item()),
        "last_token_summary": last_summary,
        "phase": "prefill",
    }


@torch.no_grad()
def probe_prefill_attention(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    *,
    attempt_label: str,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Single prefill forward with ``output_attentions=True``."""
    base: dict[str, Any] = {
        "attempt": attempt_label,
        "phase": "prefill",
        "output_attentions": True,
        "attn_implementation": attn_implementation or "default",
        "weights_obtained": False,
        "attention_summary": None,
        "error": None,
    }
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    try:
        out = model(
            prompt_ids,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
        )
    except TypeError as exc:
        base["error"] = f"TypeError: {exc}"
        return base
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    attentions = getattr(out, "attentions", None)
    if attentions is None or len(attentions) == 0:
        base["error"] = (
            "forward returned attentions=None or empty "
            "(backend may not support output_attentions=True)"
        )
        return base

    try:
        summary = _describe_attention_stack(attentions)
    except Exception as exc:
        base["error"] = f"attention parse failed: {exc}"
        return base

    base["weights_obtained"] = True
    base["attention_summary"] = summary
    return base


@torch.no_grad()
def probe_verifier_single_step_attention(
    runtime: ModelRuntime,
    prompt: str,
    *,
    attempt_label: str,
) -> dict[str, Any]:
    """Prefill via ExactKV helper, then one decode-step forward with attentions."""
    base: dict[str, Any] = {
        "attempt": attempt_label,
        "phase": "verifier_single_step",
        "output_attentions": True,
        "weights_obtained": False,
        "attention_summary": None,
        "error": None,
    }
    state = prefill_to_full_state(runtime, prompt)
    next_id = torch.tensor(
        [[state.metadata["next_token_id"]]],
        dtype=torch.long,
        device=runtime.device,
    )
    try:
        out = runtime.forward(
            next_id,
            past_key_values=state.past_key_values,
            use_cache=True,
            output_attentions=True,
        )
    except TypeError as exc:
        base["error"] = f"TypeError: {exc}"
        return base
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    attentions = getattr(out, "attentions", None)
    if attentions is None or len(attentions) == 0:
        base["error"] = "decode-step forward returned no attentions"
        return base

    try:
        summary = _describe_attention_stack(attentions)
        summary["phase"] = "verifier_single_step"
    except Exception as exc:
        base["error"] = f"attention parse failed: {exc}"
        return base

    base["weights_obtained"] = True
    base["attention_summary"] = summary
    return base


def load_probe_model(
    model_name: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, torch.device, str]:
    """Load HF causal LM for attention probes (not ModelRuntime)."""
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    load_kwargs.update(_hf_dtype_kwarg(torch_dtype))
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device != "auto":
        model = model.to(device)
        resolved = torch.device(device)
    else:
        resolved = next(model.parameters()).device
    model.eval()
    impl = attn_implementation
    if impl is None:
        impl = getattr(getattr(model, "config", None), "_attn_implementation", "default")
    return model, tokenizer, resolved, str(impl)


def collect_environment_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["cuda_version"] = torch.version.cuda
        meta["gpu_device_name"] = torch.cuda.get_device_name(0)
    return meta


def evaluate_feasibility(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive go/no-go from attempt records."""
    any_weights = any(a.get("weights_obtained") for a in attempts)
    qwen_prefill_ok = any(
        a.get("weights_obtained")
        and a.get("phase") == "prefill"
        and "qwen" in a.get("model_name", "").lower()
        for a in attempts
    )
    qwen_decode_ok = any(
        a.get("weights_obtained")
        and a.get("phase") == "verifier_single_step"
        for a in attempts
    )
    if qwen_prefill_ok:
        recommendation = "restricted_go_prefill_only"
        detail = (
            "True attention weights obtainable on Qwen prefill (eager or fallback path). "
            "Use tiny prefill-only snapshots for divergence forensics; do not use for "
            "acceptance decisions."
        )
    elif any_weights:
        recommendation = "restricted_go_non_qwen_plumbing_only"
        detail = (
            "Attention plumbing works on fallback model only; Qwen2.5 remains blocked "
            "for ExactKV primary model forensics until eager path or API changes."
        )
    else:
        recommendation = "no_go"
        detail = (
            "No attempt returned true attention weights; defer per-head forensics "
            "until backend/model path changes."
        )
    return {
        "any_weights_obtained": any_weights,
        "qwen_prefill_weights": qwen_prefill_ok,
        "qwen_decode_step_weights": qwen_decode_ok,
        "recommendation": recommendation,
        "detail": detail,
    }
