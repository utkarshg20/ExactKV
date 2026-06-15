"""Offline HF single-layer Q/K/V extraction and attention-drift probe (Phase 16B).

Derives Q/K/V from a real Hugging Face transformer layer and compares full,
materialized-compressed, and streaming-compressed attention. **Not** wired into
ExactKV default generation or model inference.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from exactkv.attention.streaming_quant_attention import (
    FORBIDDEN_ATTENTION_CLAIMS,
    attention_full,
    attention_materialized_compressed,
    attention_streaming_compressed,
    estimate_attention_memory_bytes,
    quantize_kv_int8_reference,
    run_attention_feasibility_cell,
)

EXPERIMENT_067_ID = "exp067_hf_single_layer_attention_drift"
DEFAULT_EXP067_REPORT = Path("reports/experiment_067_hf_single_layer_attention_drift.json")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"

EXP067_CLAIM_NOTE = (
    "Offline HF single-layer attention-drift probe (Phase 16B). Derives Q/K/V from "
    "a real transformer layer and measures int8 reference KV quantization drift. "
    "Not model generation integration, vLLM, CUDA/Triton kernels, or ExactKV "
    "default runtime. Theoretical memory accounting only; no measured active GPU "
    "memory, speed, throughput, latency, or serving claim."
)

DEFAULT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "ExactKV preserves greedy outputs under verification.",
    "Streaming attention processes KV in chunks.",
    "A small language model runs on CPU.",
    "Quantized key-value caches reduce stored bytes.",
    "Rotary embeddings encode position information.",
)

EXTRACTION_MODES = ("exact_qwen2_like", "projection_only", "blocked")
ROPE_STATUSES = ("applied", "skipped", "unsupported", "failed")
GQA_STATUSES = ("not_needed", "repeated", "unsupported")


@dataclass
class ExtractedQKV:
    """Q/K/V tensors in attention layout [B, H, T, D]."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    layer_idx: int
    extraction_mode: str
    rope_status: str
    grouped_query_status: str
    o_proj: nn.Linear | None
    blockers: list[str] = field(default_factory=list)

    @property
    def shapes(self) -> dict[str, list[int]]:
        return {
            "q_shape": list(self.q.shape),
            "k_shape": list(self.k.shape),
            "v_shape": list(self.v.shape),
        }


@dataclass
class DriftMetrics:
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    relative_l2_error: float
    top_dim_max_abs: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_prompts(max_prompts: int = 4) -> list[tuple[str, str]]:
    """Return (prompt_id, text) pairs."""
    out: list[tuple[str, str]] = []
    for i, text in enumerate(DEFAULT_PROMPTS[:max_prompts]):
        out.append((f"prompt_{i}", text))
    return out


def compute_drift_metrics(reference: torch.Tensor, other: torch.Tensor) -> DriftMetrics:
    """Compare two attention-context tensors."""
    if reference.shape != other.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {other.shape}")
    diff = (reference - other).detach()
    ref = reference.detach()
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    cos = float(
        F.cosine_similarity(ref.reshape(-1), other.reshape(-1), dim=0, eps=1e-8).item()
    )
    denom = ref.norm().clamp(min=1e-12)
    rel_l2 = float((diff.norm() / denom).item())
    # Per-head-dim max abs drift summary
    if diff.ndim >= 1:
        per_dim = diff.abs().amax(dim=tuple(range(diff.ndim - 1)))
        top_dim = float(per_dim.max().item())
    else:
        top_dim = max_abs
    return DriftMetrics(
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        cosine_similarity=cos,
        relative_l2_error=rel_l2,
        top_dim_max_abs=top_dim,
    )


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, n_kv, t, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return hidden_states.reshape(b, n_kv * n_rep, t, d)


def _reshape_projections(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    b, seq, _ = q.shape
    q = q.view(b, seq, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(b, seq, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    v = v.view(b, seq, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    gqa_status = "not_needed"
    if num_kv_heads != num_heads:
        if num_heads % num_kv_heads != 0:
            return q, k, v, "unsupported"
        n_rep = num_heads // num_kv_heads
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
        gqa_status = "repeated"
    return q, k, v, gqa_status


def _try_apply_qwen2_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    rotary_emb: Any,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except ImportError:
        return q, k, "unsupported"

    try:
        cos, sin = rotary_emb(q, position_ids)
        q2, k2 = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        return q2, k2, "applied"
    except Exception:
        return q, k, "failed"


def extract_qkv_from_qwen2_layer(
    hidden_states: torch.Tensor,
    layer: Any,
    *,
    layer_idx: int,
    rotary_emb: Any | None = None,
    position_ids: torch.Tensor | None = None,
    allow_projection_only: bool = True,
) -> ExtractedQKV:
    """Best-effort Q/K/V extraction from a Qwen2/Qwen2.5-like decoder layer."""
    blockers: list[str] = []
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return ExtractedQKV(
            q=torch.empty(0),
            k=torch.empty(0),
            v=torch.empty(0),
            layer_idx=layer_idx,
            extraction_mode="blocked",
            rope_status="unsupported",
            grouped_query_status="unsupported",
            o_proj=None,
            blockers=["layer has no self_attn"],
        )

    for name in ("q_proj", "k_proj", "v_proj"):
        if not hasattr(attn, name):
            return ExtractedQKV(
                q=torch.empty(0),
                k=torch.empty(0),
                v=torch.empty(0),
                layer_idx=layer_idx,
                extraction_mode="blocked",
                rope_status="unsupported",
                grouped_query_status="unsupported",
                o_proj=None,
                blockers=[f"self_attn missing {name}"],
            )

    num_heads = int(getattr(attn, "num_heads", 0) or 0)
    if num_heads <= 0:
        config = getattr(attn, "config", None)
        if config is not None:
            num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(attn, "num_key_value_heads", 0) or 0)
    if num_kv_heads <= 0:
        config = getattr(attn, "config", None)
        if config is not None:
            num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads) or num_heads)
    if num_kv_heads <= 0:
        num_kv_heads = num_heads
    head_dim = int(getattr(attn, "head_dim", 0))
    if head_dim <= 0:
        hidden_size = hidden_states.shape[-1]
        head_dim = hidden_size // max(num_heads, 1)

    if num_heads <= 0:
        blockers.append("num_heads unavailable")
        return ExtractedQKV(
            q=torch.empty(0),
            k=torch.empty(0),
            v=torch.empty(0),
            layer_idx=layer_idx,
            extraction_mode="blocked",
            rope_status="unsupported",
            grouped_query_status="unsupported",
            o_proj=getattr(attn, "o_proj", None),
            blockers=blockers,
        )

    q = attn.q_proj(hidden_states)
    k = attn.k_proj(hidden_states)
    v = attn.v_proj(hidden_states)
    q, k, v, gqa_status = _reshape_projections(
        q, k, v, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    if gqa_status == "unsupported":
        return ExtractedQKV(
            q=torch.empty(0),
            k=torch.empty(0),
            v=torch.empty(0),
            layer_idx=layer_idx,
            extraction_mode="blocked",
            rope_status="unsupported",
            grouped_query_status="unsupported",
            o_proj=getattr(attn, "o_proj", None),
            blockers=["grouped-query head count mismatch"],
        )

    rope_status = "skipped"
    extraction_mode = "projection_only"
    if rotary_emb is not None and position_ids is not None:
        q, k, rope_status = _try_apply_qwen2_rope(
            q, k, rotary_emb=rotary_emb, position_ids=position_ids
        )
        if rope_status == "applied":
            extraction_mode = "exact_qwen2_like"
        elif not allow_projection_only:
            blockers.append(f"rope failed: {rope_status}")
            return ExtractedQKV(
                q=torch.empty(0),
                k=torch.empty(0),
                v=torch.empty(0),
                layer_idx=layer_idx,
                extraction_mode="blocked",
                rope_status=rope_status,
                grouped_query_status=gqa_status,
                o_proj=getattr(attn, "o_proj", None),
                blockers=blockers,
            )
    elif rotary_emb is None:
        rope_status = "unsupported"

    return ExtractedQKV(
        q=q,
        k=k,
        v=v,
        layer_idx=layer_idx,
        extraction_mode=extraction_mode,
        rope_status=rope_status,
        grouped_query_status=gqa_status,
        o_proj=getattr(attn, "o_proj", None),
        blockers=blockers,
    )


def _apply_o_proj(attn_out: torch.Tensor, o_proj: nn.Linear) -> torch.Tensor:
    """Map attention context [B,H,T,D] through output projection."""
    b, h, t, d = attn_out.shape
    merged = attn_out.transpose(1, 2).reshape(b, t, h * d)
    return o_proj(merged)


def run_hf_attention_drift_cell(
    extracted: ExtractedQKV,
    *,
    chunk_size: int,
    causal: bool = True,
) -> dict[str, Any]:
    """Run attention drift comparisons for one extracted Q/K/V cell."""
    if extracted.extraction_mode == "blocked" or extracted.q.numel() == 0:
        return {
            "extraction_status": "blocked",
            "extraction_mode": "blocked",
            "passed": False,
            "blockers": list(extracted.blockers) or ["extraction blocked"],
        }

    q, k, v = extracted.q, extracted.k, extracted.v
    result = run_attention_feasibility_cell(
        q=q, k=k, v=v, chunk_size=chunk_size, causal=causal
    )

    sm = compute_drift_metrics(result.materialized_compressed_output, result.streaming_compressed_output)
    fm = compute_drift_metrics(result.full_output, result.materialized_compressed_output)
    fs = compute_drift_metrics(result.full_output, result.streaming_compressed_output)

    cell: dict[str, Any] = {
        "extraction_status": "success",
        "extraction_mode": extracted.extraction_mode,
        "rope_status": extracted.rope_status,
        "grouped_query_status": extracted.grouped_query_status,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "chunk_size": chunk_size,
        "streaming_vs_materialized": sm.to_dict(),
        "full_vs_materialized": fm.to_dict(),
        "full_vs_streaming": fs.to_dict(),
        "memory_accounting": result.memory_accounting.to_dict(),
        "passed": result.passed,
        "tolerance": result.tolerance,
        "blockers": list(extracted.blockers),
    }

    o_proj = extracted.o_proj
    if o_proj is not None:
        try:
            full_ctx = _apply_o_proj(result.full_output, o_proj)
            mat_ctx = _apply_o_proj(result.materialized_compressed_output, o_proj)
            stream_ctx = _apply_o_proj(result.streaming_compressed_output, o_proj)
            cell["output_projection"] = {
                "streaming_vs_materialized": compute_drift_metrics(mat_ctx, stream_ctx).to_dict(),
                "full_vs_materialized": compute_drift_metrics(full_ctx, mat_ctx).to_dict(),
                "full_vs_streaming": compute_drift_metrics(full_ctx, stream_ctx).to_dict(),
                "note": (
                    "o_proj applied to attention context only; not full layer output "
                    "(no residual/MLP/layernorm)"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            cell["output_projection"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "note": "o_proj drift not available",
            }

    return cell


def resolve_layer_indices(num_layers: int, layers: Sequence[int] | None) -> list[int]:
    if num_layers <= 0:
        return []
    if not layers:
        mid = num_layers // 2
        return sorted({0, mid, num_layers - 1})
    return sorted({i for i in layers if 0 <= i < num_layers})


def extract_qkv_cells_from_model(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[tuple[str, str]],
    *,
    layer_indices: Sequence[int],
    device: torch.device | str,
    allow_projection_only: bool = True,
) -> list[tuple[str, str, int, ExtractedQKV]]:
    """Run forward passes and extract Q/K/V per prompt/layer."""
    model.eval()
    rotary_emb = getattr(getattr(model, "model", model), "rotary_emb", None)
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return []

    outputs: list[tuple[str, str, int, ExtractedQKV]] = []
    for prompt_id, text in prompts:
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        with torch.no_grad():
            fwd = model(input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = fwd.hidden_states
        if hidden_states is None:
            continue

        for layer_idx in layer_indices:
            if layer_idx >= len(hidden_states):
                continue
            # Input hidden state to decoder layer `layer_idx`
            hs = hidden_states[layer_idx]
            extracted = extract_qkv_from_qwen2_layer(
                hs,
                layers[layer_idx],
                layer_idx=layer_idx,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
                allow_projection_only=allow_projection_only,
            )
            preview = text[:80]
            outputs.append((prompt_id, preview, layer_idx, extracted))
    return outputs


def run_exp067_probe(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    layer_indices: Sequence[int] | None = None,
    chunk_sizes: Sequence[int] = (16, 32, 64),
    max_prompts: int = 4,
    local_files_only: bool = False,
    allow_projection_only: bool = True,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    prompt_provider: Callable[[int], list[tuple[str, str]]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 067 and return report dict."""
    torch_dtype = getattr(torch, dtype, torch.float32)
    blockers: list[str] = []
    cells: list[dict[str, Any]] = []

    if prompt_provider is None:
        prompts = default_prompts(max_prompts)
    else:
        prompts = prompt_provider(max_prompts)

    try:
        if model_loader is not None:
            model, tokenizer = model_loader(
                model_id=model_id,
                device=device,
                dtype=torch_dtype,
                local_files_only=local_files_only,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
            model.to(device)
        load_ok = True
    except Exception as exc:  # noqa: BLE001
        load_ok = False
        blockers.append(f"model load failed: {type(exc).__name__}: {exc}")
        model = None
        tokenizer = None

    if load_ok and model is not None and tokenizer is not None:
        num_layers = len(getattr(getattr(model, "model", model), "layers", []))
        layers = resolve_layer_indices(num_layers, list(layer_indices) if layer_indices else None)
        extractions = extract_qkv_cells_from_model(
            model,
            tokenizer,
            prompts,
            layer_indices=layers,
            device=device,
            allow_projection_only=allow_projection_only,
        )
        for prompt_id, preview, layer_idx, extracted in extractions:
            for chunk_size in chunk_sizes:
                cell = run_hf_attention_drift_cell(extracted, chunk_size=chunk_size, causal=True)
                cell.update(
                    {
                        "model_id": model_id,
                        "prompt_id": prompt_id,
                        "prompt_preview": preview,
                        "layer_idx": layer_idx,
                    }
                )
                if cell.get("extraction_status") == "blocked":
                    cell["extraction_mode"] = "blocked"
                cells.append(cell)

    successful = [c for c in cells if c.get("extraction_status") == "success"]
    blocked = [c for c in cells if c.get("extraction_status") != "success"]
    stream_pass = sum(1 for c in successful if c.get("passed"))
    max_sm = max(
        (c["streaming_vs_materialized"]["max_abs_error"] for c in successful),
        default=0.0,
    )
    full_stream_max = max(
        (c["full_vs_streaming"]["max_abs_error"] for c in successful),
        default=0.0,
    )
    full_stream_mean = (
        sum(c["full_vs_streaming"]["mean_abs_error"] for c in successful) / len(successful)
        if successful
        else 0.0
    )

    reductions = [
        c["memory_accounting"]["theoretical_streaming_working_reduction_vs_materialized"]
        for c in successful
        if c.get("memory_accounting")
    ]
    mem_summary = {
        "best_theoretical_streaming_working_reduction": max(reductions) if reductions else 0.0,
        "worst_theoretical_streaming_working_reduction": min(reductions) if reductions else 0.0,
    }

    op_cells = [c["output_projection"] for c in successful if "output_projection" in c and "error" not in c["output_projection"]]
    op_summary: dict[str, Any] | None = None
    if op_cells:
        op_summary = {
            "cells_with_output_projection": len(op_cells),
            "max_full_vs_streaming_after_o_proj": max(
                o["full_vs_streaming"]["max_abs_error"] for o in op_cells
            ),
            "mean_full_vs_streaming_after_o_proj": sum(
                o["full_vs_streaming"]["mean_abs_error"] for o in op_cells
            )
            / len(op_cells),
        }

    if not load_ok:
        status = "blocked"
    elif successful and stream_pass == len(successful):
        status = "pass"
    elif successful:
        status = "failed"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_067_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "model_load_succeeded": load_ok,
        "total_cells": len(cells),
        "successful_cells": len(successful),
        "blocked_cells": len(blocked),
        "streaming_vs_materialized_pass_cells": stream_pass,
        "max_streaming_vs_materialized_error": max_sm,
        "full_vs_streaming_drift_summary": {
            "max_abs_error": full_stream_max,
            "mean_abs_error": full_stream_mean,
            "cell_count": len(successful),
        },
        "output_projection_drift_summary": op_summary,
        "memory_accounting_summary": mem_summary,
        "extraction_blockers": blockers,
        "cells": cells,
        "claim_note": EXP067_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": [
            "Single-layer offline probe; not full model forward pass integration.",
            "projection_only mode is not exact model-layer attention.",
            "Theoretical memory accounting only.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "prompt_count": len(prompts),
        "chunk_sizes": list(chunk_sizes),
    }


def validate_exp067_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "total_cells",
        "successful_cells",
        "blocked_cells",
        "streaming_vs_materialized_pass_cells",
        "max_streaming_vs_materialized_error",
        "full_vs_streaming_drift_summary",
        "memory_accounting_summary",
        "no_performance_claims_note",
        "limitations",
        "claim_note",
        "forbidden_claims",
        "cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_067_ID:
        errors.append("experiment_id mismatch")

    for term in ("throughput_improved", "latency_improved", "speedup_claim"):
        if term in str(report).lower():
            errors.append(f"forbidden phrase: {term}")

    cells = report.get("cells")
    if not isinstance(cells, list):
        errors.append("cells must be a list")
        return errors

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {idx} not dict")
            continue
        for ck in (
            "model_id",
            "prompt_id",
            "prompt_preview",
            "layer_idx",
            "extraction_status",
            "extraction_mode",
            "rope_status",
            "grouped_query_status",
            "chunk_size",
            "passed",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"cell {idx} missing {ck}")
        if cell.get("extraction_status") == "success":
            for mk in (
                "streaming_vs_materialized",
                "full_vs_materialized",
                "full_vs_streaming",
                "memory_accounting",
            ):
                if mk not in cell:
                    errors.append(f"cell {idx} missing {mk}")
            mem = cell.get("memory_accounting")
            if isinstance(mem, dict):
                for mk in (
                    "full_kv_bytes",
                    "materialized_working_kv_bytes",
                    "streaming_peak_chunk_working_kv_bytes",
                ):
                    if mk not in mem:
                        errors.append(f"cell {idx} memory_accounting missing {mk}")

    return errors
