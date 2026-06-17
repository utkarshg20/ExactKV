"""Offline Qwen2/Qwen2.5 multi-layer drift accumulation probe (Phase 16D).

Replays consecutive decoder blocks with full, materialized-compressed, and
streaming-compressed attention paths. **Not** wired into ExactKV generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
import torch.nn as nn

from exactkv.attention.hf_single_layer_probe import (
    DEFAULT_MODEL_ID,
    FORBIDDEN_ATTENTION_CLAIMS,
    compute_drift_metrics,
    extract_qkv_from_qwen2_layer,
    generate_long_prompt_text,
    long_context_prompts,
    resolve_model_rotary_emb,
)
from exactkv.attention.streaming_quant_attention import (
    DEFAULT_STREAMING_TOLERANCE_FP32,
    MemoryAccounting,
    attention_full,
    attention_materialized_compressed,
    attention_streaming_compressed,
    compute_candidate_tolerances,
    estimate_attention_memory_bytes,
    layer_depth_aware_streaming_tolerance,
    quantize_kv_int8_reference,
    reference_high_precision_tolerance,
    strict_streaming_tolerance,
)

EXPERIMENT_069_ID = "exp069_multilayer_attention_drift_accumulation"
DEFAULT_EXP069_REPORT = Path("reports/experiment_069_multilayer_attention_drift_accumulation.json")
DEFAULT_PREFIX_LAYER_COUNTS: tuple[int, ...] = (1, 2, 4)
DEFAULT_TARGET_TOKEN_LENGTHS_069: tuple[int, ...] = (64, 128)

EXPERIMENT_070_ID = "exp070_streaming_multilayer_numerics_audit"
DEFAULT_EXP070_REPORT = Path("reports/experiment_070_streaming_multilayer_numerics_audit.json")
DEFAULT_TARGET_TOKEN_LENGTHS_070: tuple[int, ...] = (64, 128, 256)
DEFAULT_CHUNK_SIZES_070: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_ACCUMULATOR_MODES: tuple[str, ...] = ("default", "float32", "float64")

PHASE16D_REGRESSION_CELL: dict[str, Any] = {
    "prompt_id": "long_128",
    "target_token_length": 128,
    "prefix_layer_count": 4,
    "chunk_size": 32,
    "accumulator_mode": "default",
    "model_id": DEFAULT_MODEL_ID,
    "dtype": "float32",
    "device": "cpu",
    "phase16d_observed_error": 5.79e-4,
    "phase16d_tolerance": DEFAULT_STREAMING_TOLERANCE_FP32,
}

EXP070_CLAIM_NOTE = (
    "Numerical audit of offline streaming compressed attention under multi-layer "
    "accumulation (Phase 16E). Not model generation integration, vLLM, CUDA/Triton "
    "kernels, or ExactKV default runtime. Tolerance recommendations are diagnostic "
    "only. Theoretical memory accounting only; no measured active GPU memory, speed, "
    "throughput, latency, or serving claim."
)

EXP069_CLAIM_NOTE = (
    "Offline multi-layer drift accumulation probe (Phase 16D). Replays consecutive "
    "Qwen2/Qwen2.5 decoder blocks with compressed attention paths. Not model "
    "generation integration, vLLM, CUDA/Triton kernels, or ExactKV default runtime. "
    "Full-block parity is measured before interpreting drift. Theoretical memory "
    "accounting only; no measured active GPU memory, speed, throughput, latency, "
    "or serving claim."
)

AttentionPath = Literal["full", "materialized_compressed", "streaming_compressed"]

DEFAULT_PARITY_TOLERANCE_FP32 = 1e-2


@dataclass
class LayerMemoryRecord:
    layer_idx: int
    full_kv_bytes: int
    stored_quantized_kv_bytes: int
    materialized_working_kv_bytes: int
    streaming_peak_chunk_working_kv_bytes: int
    metadata_bytes: int
    chunk_size: int
    num_chunks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _elem_size(dtype: torch.dtype) -> int:
    return 4 if dtype == torch.float32 else 2


def _layer_memory_from_qkv(
    q: torch.Tensor,
    *,
    chunk_size: int,
    layer_idx: int,
) -> LayerMemoryRecord:
    mem = estimate_attention_memory_bytes(
        batch=q.shape[0],
        heads=q.shape[1],
        seq_len=q.shape[2],
        head_dim=q.shape[3],
        element_size_fp=_elem_size(q.dtype),
        chunk_size=chunk_size,
    )
    return LayerMemoryRecord(
        layer_idx=layer_idx,
        full_kv_bytes=mem.full_kv_bytes,
        stored_quantized_kv_bytes=mem.stored_quantized_kv_bytes,
        materialized_working_kv_bytes=mem.materialized_working_kv_bytes,
        streaming_peak_chunk_working_kv_bytes=mem.streaming_peak_chunk_working_kv_bytes,
        metadata_bytes=mem.metadata_bytes,
        chunk_size=mem.chunk_size,
        num_chunks=mem.num_chunks,
    )


def aggregate_layer_memory(records: Sequence[LayerMemoryRecord]) -> dict[str, Any]:
    if not records:
        return {
            "aggregate_full_kv_bytes": 0,
            "aggregate_stored_quantized_kv_bytes": 0,
            "aggregate_materialized_working_kv_bytes": 0,
            "aggregate_streaming_peak_working_kv_bytes_conservative": 0,
            "best_theoretical_streaming_reduction": 0.0,
            "layer_count": 0,
        }
    agg_full = sum(r.full_kv_bytes for r in records)
    agg_stored = sum(r.stored_quantized_kv_bytes for r in records)
    agg_mat = sum(r.materialized_working_kv_bytes for r in records)
    agg_stream_peak = max(r.streaming_peak_chunk_working_kv_bytes for r in records)
    if agg_mat > 0:
        reduction = 1.0 - (agg_stream_peak / agg_mat)
    else:
        reduction = 0.0
    return {
        "aggregate_full_kv_bytes": agg_full,
        "aggregate_stored_quantized_kv_bytes": agg_stored,
        "aggregate_materialized_working_kv_bytes": agg_mat,
        "aggregate_streaming_peak_working_kv_bytes_conservative": agg_stream_peak,
        "best_theoretical_streaming_reduction": reduction,
        "layer_count": len(records),
    }


def _attention_context(
    extracted_qkv: Any,
    *,
    path: AttentionPath,
    chunk_size: int,
    accumulator_dtype: str | None = None,
    collect_streaming_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any] | None]:
    q, k, v = extracted_qkv.q, extracted_qkv.k, extracted_qkv.v
    if path == "full":
        return attention_full(q, k, v, causal=True)
    qkv = quantize_kv_int8_reference(k, v)
    if path == "materialized_compressed":
        return attention_materialized_compressed(q, qkv, causal=True)
    result = attention_streaming_compressed(
        q,
        qkv,
        chunk_size,
        causal=True,
        accumulator_dtype=accumulator_dtype,
        return_diagnostics=collect_streaming_diagnostics,
    )
    if collect_streaming_diagnostics:
        out, diag = result
        return out, diag
    return result


def run_qwen_decoder_block(
    hidden: torch.Tensor,
    layer: Any,
    *,
    layer_idx: int,
    attention_path: AttentionPath,
    chunk_size: int,
    rotary_emb: Any | None,
    position_ids: torch.Tensor,
    streaming_accumulator_dtype: str | None = None,
    collect_streaming_diagnostics: bool = False,
) -> tuple[torch.Tensor, LayerMemoryRecord | None, dict[str, Any] | None]:
    """Replay one Qwen2/Qwen2.5 decoder block with chosen attention path."""
    residual = hidden
    normed = layer.input_layernorm(hidden)
    extracted = extract_qkv_from_qwen2_layer(
        normed,
        layer,
        layer_idx=layer_idx,
        rotary_emb=rotary_emb,
        position_ids=position_ids,
        allow_projection_only=True,
    )
    if extracted.extraction_mode == "blocked" or extracted.q.numel() == 0:
        raise RuntimeError(
            f"layer {layer_idx} QKV extraction blocked: {extracted.blockers}"
        )

    layer_diag: dict[str, Any] | None = None
    attn_result = _attention_context(
        extracted,
        path=attention_path,
        chunk_size=chunk_size,
        accumulator_dtype=streaming_accumulator_dtype,
        collect_streaming_diagnostics=collect_streaming_diagnostics,
    )
    if collect_streaming_diagnostics and attention_path == "streaming_compressed":
        attn_ctx, layer_diag = attn_result
    else:
        attn_ctx = attn_result  # type: ignore[assignment]

    o_proj = extracted.o_proj
    if o_proj is None:
        raise RuntimeError(f"layer {layer_idx} missing o_proj")

    b, h, t, d = attn_ctx.shape
    merged = attn_ctx.transpose(1, 2).reshape(b, t, h * d)
    attn_out = o_proj(merged)
    hidden = residual + attn_out

    residual2 = hidden
    normed2 = layer.post_attention_layernorm(hidden)
    mlp_out = layer.mlp(normed2)
    hidden = residual2 + mlp_out

    mem = _layer_memory_from_qkv(extracted.q, chunk_size=chunk_size, layer_idx=layer_idx)
    return hidden, mem, layer_diag


def run_qwen_decoder_block_traced(
    hidden: torch.Tensor,
    layer: Any,
    *,
    layer_idx: int,
    attention_path: AttentionPath,
    chunk_size: int,
    rotary_emb: Any | None,
    position_ids: torch.Tensor,
    streaming_accumulator_dtype: str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Replay one decoder block and return post-MLP hidden plus per-stage checkpoints."""
    residual = hidden
    layer_input = hidden
    normed = layer.input_layernorm(hidden)
    extracted = extract_qkv_from_qwen2_layer(
        normed,
        layer,
        layer_idx=layer_idx,
        rotary_emb=rotary_emb,
        position_ids=position_ids,
        allow_projection_only=True,
    )
    if extracted.extraction_mode == "blocked" or extracted.q.numel() == 0:
        raise RuntimeError(
            f"layer {layer_idx} QKV extraction blocked: {extracted.blockers}"
        )

    attn_ctx = _attention_context(
        extracted,
        path=attention_path,
        chunk_size=chunk_size,
        accumulator_dtype=streaming_accumulator_dtype,
    )
    o_proj = extracted.o_proj
    if o_proj is None:
        raise RuntimeError(f"layer {layer_idx} missing o_proj")

    b, h, t, d = attn_ctx.shape
    merged = attn_ctx.transpose(1, 2).reshape(b, t, h * d)
    attn_out = o_proj(merged)
    post_attention_hidden = residual + attn_out

    residual2 = post_attention_hidden
    normed2 = layer.post_attention_layernorm(post_attention_hidden)
    mlp_out = layer.mlp(normed2)
    post_mlp_hidden = residual2 + mlp_out

    checkpoints = {
        "layer_input": layer_input.detach(),
        "attn_context": merged.detach(),
        "attn_output": attn_out.detach(),
        "post_attention_hidden": post_attention_hidden.detach(),
        "post_mlp_hidden": post_mlp_hidden.detach(),
    }
    return post_mlp_hidden, checkpoints


def replay_prefix_layers(
    hidden: torch.Tensor,
    layers: Sequence[Any],
    *,
    prefix_layer_count: int,
    attention_path: AttentionPath,
    chunk_size: int,
    rotary_emb: Any | None,
    position_ids: torch.Tensor,
    streaming_accumulator_dtype: str | None = None,
    collect_streaming_diagnostics: bool = False,
) -> tuple[torch.Tensor, list[LayerMemoryRecord], list[dict[str, Any]]]:
    """Run the first ``prefix_layer_count`` decoder layers."""
    mem_records: list[LayerMemoryRecord] = []
    diagnostics: list[dict[str, Any]] = []
    for idx in range(prefix_layer_count):
        hidden, mem, layer_diag = run_qwen_decoder_block(
            hidden,
            layers[idx],
            layer_idx=idx,
            attention_path=attention_path,
            chunk_size=chunk_size,
            rotary_emb=rotary_emb,
            position_ids=position_ids,
            streaming_accumulator_dtype=streaming_accumulator_dtype,
            collect_streaming_diagnostics=collect_streaming_diagnostics,
        )
        mem_records.append(mem)
        if layer_diag is not None:
            layer_diag = {**layer_diag, "layer_idx": idx}
            diagnostics.append(layer_diag)
    return hidden, mem_records, diagnostics


def check_full_block_parity(
    replayed_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    *,
    tolerance: float = DEFAULT_PARITY_TOLERANCE_FP32,
) -> dict[str, Any]:
    """Compare offline full-path replay hidden state to HF reference."""
    if replayed_hidden.shape != reference_hidden.shape:
        return {
            "full_block_parity_status": "failed",
            "blockers": [
                f"shape mismatch replay {tuple(replayed_hidden.shape)} "
                f"vs hf {tuple(reference_hidden.shape)}"
            ],
        }
    metrics = compute_drift_metrics(reference_hidden, replayed_hidden)
    status = "passed" if metrics.max_abs_error <= tolerance else "failed"
    return {
        "full_block_parity_status": status,
        "full_block_parity_metrics": metrics.to_dict(),
        "parity_tolerance": tolerance,
    }


def run_multilayer_drift_cell(
    *,
    model: Any,
    input_ids: torch.Tensor,
    hf_hidden_states: tuple[torch.Tensor, ...],
    prompt_id: str,
    prompt_preview: str,
    target_token_length: int,
    actual_token_length: int,
    prefix_layer_count: int,
    chunk_size: int,
    allow_parity_fail: bool = False,
    parity_tolerance: float = DEFAULT_PARITY_TOLERANCE_FP32,
    streaming_tolerance: float = DEFAULT_STREAMING_TOLERANCE_FP32,
) -> dict[str, Any]:
    """Run one multi-layer drift accumulation cell."""
    blockers: list[str] = []
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return _blocked_cell(
            prompt_id=prompt_id,
            prompt_preview=prompt_preview,
            target_token_length=target_token_length,
            actual_token_length=actual_token_length,
            prefix_layer_count=prefix_layer_count,
            chunk_size=chunk_size,
            blockers=["model has no decoder layers"],
        )

    if prefix_layer_count > len(layers):
        return _blocked_cell(
            prompt_id=prompt_id,
            prompt_preview=prompt_preview,
            target_token_length=target_token_length,
            actual_token_length=actual_token_length,
            prefix_layer_count=prefix_layer_count,
            chunk_size=chunk_size,
            blockers=[f"prefix_layer_count {prefix_layer_count} exceeds model depth"],
        )

    rotary_emb = resolve_model_rotary_emb(model)
    seq_len = input_ids.shape[-1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    embed = getattr(getattr(model, "model", model), "embed_tokens", None)
    if embed is None:
        return _blocked_cell(
            prompt_id=prompt_id,
            prompt_preview=prompt_preview,
            target_token_length=target_token_length,
            actual_token_length=actual_token_length,
            prefix_layer_count=prefix_layer_count,
            chunk_size=chunk_size,
            blockers=["model missing embed_tokens"],
        )

    with torch.no_grad():
        hidden0 = embed(input_ids)

        try:
            full_hidden, full_mem, _ = replay_prefix_layers(
                hidden0,
                layers,
                prefix_layer_count=prefix_layer_count,
                attention_path="full",
                chunk_size=chunk_size,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
            )
            mat_hidden, mat_mem, _ = replay_prefix_layers(
                hidden0,
                layers,
                prefix_layer_count=prefix_layer_count,
                attention_path="materialized_compressed",
                chunk_size=chunk_size,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
            )
            stream_hidden, stream_mem, _ = replay_prefix_layers(
                hidden0,
                layers,
                prefix_layer_count=prefix_layer_count,
                attention_path="streaming_compressed",
                chunk_size=chunk_size,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
            )
        except Exception as exc:  # noqa: BLE001
            return _blocked_cell(
                prompt_id=prompt_id,
                prompt_preview=prompt_preview,
                target_token_length=target_token_length,
                actual_token_length=actual_token_length,
                prefix_layer_count=prefix_layer_count,
                chunk_size=chunk_size,
                blockers=[f"replay failed: {type(exc).__name__}: {exc}"],
            )

    ref_idx = prefix_layer_count
    if ref_idx >= len(hf_hidden_states):
        blockers.append(f"HF hidden_states missing index {ref_idx}")
        parity_info: dict[str, Any] = {"full_block_parity_status": "blocked", "blockers": blockers}
    else:
        parity_info = check_full_block_parity(
            full_hidden,
            hf_hidden_states[ref_idx],
            tolerance=parity_tolerance,
        )

    sm = compute_drift_metrics(mat_hidden, stream_hidden)
    fs = compute_drift_metrics(full_hidden, stream_hidden)
    fm = compute_drift_metrics(full_hidden, mat_hidden)

    streaming_pass = sm.max_abs_error <= streaming_tolerance
    parity_status = parity_info.get("full_block_parity_status", "blocked")
    parity_ok = parity_status == "passed"
    if parity_status == "failed" and not allow_parity_fail:
        blockers.append(
            f"full_block_parity failed max_abs={parity_info.get('full_block_parity_metrics', {}).get('max_abs_error')}"
        )

    passed = streaming_pass and (parity_ok or allow_parity_fail)

    return {
        "prompt_id": prompt_id,
        "prompt_preview": prompt_preview,
        "target_token_length": target_token_length,
        "actual_token_length": actual_token_length,
        "prefix_layer_count": prefix_layer_count,
        "chunk_size": chunk_size,
        "full_block_parity_metrics": parity_info.get("full_block_parity_metrics"),
        "full_block_parity_status": parity_status,
        "parity_tolerance": parity_tolerance,
        "streaming_vs_materialized_hidden_metrics": sm.to_dict(),
        "full_vs_streaming_hidden_metrics": fs.to_dict(),
        "full_vs_materialized_hidden_metrics": fm.to_dict(),
        "per_layer_memory_accounting": {
            "full_path": [m.to_dict() for m in full_mem],
            "materialized_path": [m.to_dict() for m in mat_mem],
            "streaming_path": [m.to_dict() for m in stream_mem],
        },
        "aggregate_memory_accounting": {
            "full_path": aggregate_layer_memory(full_mem),
            "materialized_path": aggregate_layer_memory(mat_mem),
            "streaming_path": aggregate_layer_memory(stream_mem),
        },
        "streaming_tolerance": streaming_tolerance,
        "streaming_passed": streaming_pass,
        "passed": passed,
        "blockers": blockers,
    }


def _blocked_cell(
    *,
    prompt_id: str,
    prompt_preview: str,
    target_token_length: int,
    actual_token_length: int,
    prefix_layer_count: int,
    chunk_size: int,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "prompt_preview": prompt_preview,
        "target_token_length": target_token_length,
        "actual_token_length": actual_token_length,
        "prefix_layer_count": prefix_layer_count,
        "chunk_size": chunk_size,
        "full_block_parity_status": "blocked",
        "full_block_parity_metrics": None,
        "streaming_vs_materialized_hidden_metrics": None,
        "full_vs_streaming_hidden_metrics": None,
        "full_vs_materialized_hidden_metrics": None,
        "per_layer_memory_accounting": None,
        "aggregate_memory_accounting": None,
        "streaming_passed": False,
        "passed": False,
        "blockers": blockers,
    }


def run_exp069_probe(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    target_token_lengths: Sequence[int] = DEFAULT_TARGET_TOKEN_LENGTHS_069,
    prefix_layer_counts: Sequence[int] = DEFAULT_PREFIX_LAYER_COUNTS,
    chunk_sizes: Sequence[int] = (16, 32, 64),
    max_prompts: int = 2,
    local_files_only: bool = False,
    allow_parity_fail: bool = False,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    prompt_provider: Callable[[Any, Sequence[int], int], list[tuple[str, str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 069 multi-layer drift accumulation probe."""
    torch_dtype = getattr(torch, dtype, torch.float32)
    blockers: list[str] = []
    cells: list[dict[str, Any]] = []
    prompts: list[tuple[str, str, int, int]] = []

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
        model.eval()
        if prompt_provider is not None:
            prompts = prompt_provider(tokenizer, target_token_lengths, max_prompts)
        else:
            prompts = long_context_prompts(
                tokenizer, target_token_lengths, max_prompts=max_prompts
            )

        for prompt_id, text, target_len, actual_len in prompts:
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            with torch.no_grad():
                fwd = model(input_ids, output_hidden_states=True, use_cache=False)
            hf_hs = fwd.hidden_states
            if hf_hs is None:
                continue
            preview = text[:80]

            for n_layers in prefix_layer_counts:
                for chunk_size in chunk_sizes:
                    cell = run_multilayer_drift_cell(
                        model=model,
                        input_ids=input_ids,
                        hf_hidden_states=hf_hs,
                        prompt_id=prompt_id,
                        prompt_preview=preview,
                        target_token_length=target_len,
                        actual_token_length=actual_len,
                        prefix_layer_count=n_layers,
                        chunk_size=chunk_size,
                        allow_parity_fail=allow_parity_fail,
                    )
                    cell["model_id"] = model_id
                    cells.append(cell)

    successful = [c for c in cells if c.get("streaming_vs_materialized_hidden_metrics") is not None]
    blocked = [c for c in cells if c.get("streaming_vs_materialized_hidden_metrics") is None]
    stream_pass = sum(1 for c in successful if c.get("streaming_passed"))
    parity_pass = sum(
        1 for c in successful if c.get("full_block_parity_status") == "passed"
    )
    max_sm = max(
        (
            c["streaming_vs_materialized_hidden_metrics"]["max_abs_error"]
            for c in successful
        ),
        default=0.0,
    )

    fs_max = max(
        (c["full_vs_streaming_hidden_metrics"]["max_abs_error"] for c in successful),
        default=0.0,
    )
    fs_mean = (
        sum(c["full_vs_streaming_hidden_metrics"]["mean_abs_error"] for c in successful)
        / len(successful)
        if successful
        else 0.0
    )
    fm_max = max(
        (c["full_vs_materialized_hidden_metrics"]["max_abs_error"] for c in successful),
        default=0.0,
    )
    fm_mean = (
        sum(c["full_vs_materialized_hidden_metrics"]["mean_abs_error"] for c in successful)
        / len(successful)
        if successful
        else 0.0
    )

    reductions: list[float] = []
    num_chunks_list: list[int] = []
    actual_lengths: list[int] = []
    max_prefix = 0
    for c in successful:
        agg = c.get("aggregate_memory_accounting", {})
        stream_agg = agg.get("streaming_path", {}) if isinstance(agg, dict) else {}
        if stream_agg:
            reductions.append(float(stream_agg.get("best_theoretical_streaming_reduction", 0.0)))
        per_layer = c.get("per_layer_memory_accounting", {})
        if isinstance(per_layer, dict):
            stream_layers = per_layer.get("streaming_path", [])
            if stream_layers:
                num_chunks_list.append(max(layer.get("num_chunks", 0) for layer in stream_layers))
        actual_lengths.append(int(c.get("actual_token_length", 0)))
        max_prefix = max(max_prefix, int(c.get("prefix_layer_count", 0)))

    mem_summary = {
        "best_theoretical_streaming_reduction": max(reductions) if reductions else 0.0,
        "worst_theoretical_streaming_reduction": min(reductions) if reductions else 0.0,
        "cells_with_reduction_gt_zero": sum(1 for r in reductions if r > 0),
    }

    if not load_ok:
        status = "blocked"
    elif successful and all(c.get("passed") for c in successful):
        status = "pass"
    elif successful:
        status = "failed"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_069_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "model_load_succeeded": load_ok,
        "target_token_lengths": list(target_token_lengths)[:max_prompts],
        "prefix_layer_counts": list(prefix_layer_counts),
        "chunk_sizes": list(chunk_sizes),
        "total_cells": len(cells),
        "successful_cells": len(successful),
        "blocked_cells": len(blocked),
        "full_block_parity_pass_cells": parity_pass,
        "streaming_vs_materialized_pass_cells": stream_pass,
        "max_streaming_vs_materialized_error": max_sm,
        "full_vs_streaming_drift_summary": {
            "max_abs_error": fs_max,
            "mean_abs_error": fs_mean,
            "cell_count": len(successful),
        },
        "full_vs_materialized_drift_summary": {
            "max_abs_error": fm_max,
            "mean_abs_error": fm_mean,
            "cell_count": len(successful),
        },
        "memory_accounting_summary": mem_summary,
        "longest_context_tested": max(actual_lengths) if actual_lengths else 0,
        "max_prefix_layers_tested": max_prefix,
        "max_num_chunks": max(num_chunks_list) if num_chunks_list else 0,
        "allow_parity_fail": allow_parity_fail,
        "extraction_blockers": blockers,
        "cells": cells,
        "claim_note": EXP069_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": [
            "Offline multi-layer replay; not full model generation integration.",
            "Full-block parity required before interpreting drift unless --allow-parity-fail.",
            "Compressed attention substituted per layer; not production integration.",
            "Theoretical memory accounting only.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "prompt_count": len(prompts),
    }


def validate_exp069_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "target_token_lengths",
        "prefix_layer_counts",
        "chunk_sizes",
        "total_cells",
        "successful_cells",
        "blocked_cells",
        "full_block_parity_pass_cells",
        "streaming_vs_materialized_pass_cells",
        "max_streaming_vs_materialized_error",
        "full_vs_streaming_drift_summary",
        "full_vs_materialized_drift_summary",
        "memory_accounting_summary",
        "longest_context_tested",
        "max_prefix_layers_tested",
        "max_num_chunks",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "forbidden_claims",
        "cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_069_ID:
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
            "prompt_id",
            "target_token_length",
            "actual_token_length",
            "prefix_layer_count",
            "chunk_size",
            "full_block_parity_status",
            "passed",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"cell {idx} missing {ck}")
        if cell.get("streaming_vs_materialized_hidden_metrics") is not None:
            agg = cell.get("aggregate_memory_accounting")
            if not isinstance(agg, dict):
                errors.append(f"cell {idx} missing aggregate_memory_accounting")

    return errors


def _is_phase16d_regression_cell(cell: dict[str, Any]) -> bool:
    return (
        cell.get("prompt_id") == PHASE16D_REGRESSION_CELL["prompt_id"]
        and cell.get("target_token_length") == PHASE16D_REGRESSION_CELL["target_token_length"]
        and cell.get("prefix_layer_count") == PHASE16D_REGRESSION_CELL["prefix_layer_count"]
        and cell.get("chunk_size") == PHASE16D_REGRESSION_CELL["chunk_size"]
        and cell.get("accumulator_mode") == PHASE16D_REGRESSION_CELL["accumulator_mode"]
    )


def recommend_tolerance_policy(
    cells: Sequence[dict[str, Any]],
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Recommend a tolerance policy from measured audit cells (diagnostic only)."""
    successful = [
        c for c in cells if c.get("streaming_vs_materialized_hidden_metrics") is not None
    ]
    if not successful:
        return {
            "policy": "insufficient_data",
            "rationale": "No successful audit cells to analyze.",
            "strict_tolerance": strict_streaming_tolerance(dtype),
        }

    strict_fails = [c for c in successful if not c.get("strict_tolerance_pass")]
    default_fails = [
        c for c in successful
        if c.get("accumulator_mode") == "default" and not c.get("strict_tolerance_pass")
    ]
    fp32_pass_strict = all(
        c.get("strict_tolerance_pass")
        for c in successful
        if c.get("accumulator_mode") == "float32"
    )
    fp64_pass_strict = all(
        c.get("strict_tolerance_pass")
        for c in successful
        if c.get("accumulator_mode") == "float64"
    )

    if not strict_fails:
        return {
            "policy": "keep_strict_tolerance",
            "rationale": (
                "All audited cells pass the Phase 16D strict tolerance under tested "
                "accumulator modes."
            ),
            "strict_tolerance": strict_streaming_tolerance(dtype),
            "recommended_tolerance_formula": "strict_16d",
        }

    if default_fails and fp32_pass_strict and fp64_pass_strict:
        return {
            "policy": "keep_strict_tolerance_use_float32_accumulator",
            "rationale": (
                "Default accumulator fails strict tolerance at multi-layer boundaries, "
                "but explicit float32/float64 accumulators pass. This indicates expected "
                "floating-point accumulation error, not an online-softmax algorithm bug."
            ),
            "strict_tolerance": strict_streaming_tolerance(dtype),
            "recommended_tolerance_formula": "strict_16d",
            "accumulator_recommendation": "float32",
        }

    depth_aware_would_pass = all(
        c["streaming_vs_materialized_hidden_metrics"]["max_abs_error"]
        <= layer_depth_aware_streaming_tolerance(dtype, int(c["prefix_layer_count"]))
        for c in strict_fails
    )
    if depth_aware_would_pass:
        return {
            "policy": "documented_layer_depth_aware_tolerance",
            "rationale": (
                "Strict tolerance failures correlate with prefix depth; a sqrt(depth)-scaled "
                "tolerance explains measured errors without loosening single-layer gates."
            ),
            "strict_tolerance": strict_streaming_tolerance(dtype),
            "recommended_tolerance_formula": "layer_depth_aware",
        }

    return {
        "policy": "keep_strict_tolerance_report_boundary_fail",
        "rationale": (
            "One or more cells fail strict tolerance without a clear accumulator-precision "
            "remedy; keep strict tolerance and report boundary failures explicitly."
        ),
        "strict_tolerance": strict_streaming_tolerance(dtype),
        "recommended_tolerance_formula": "strict_16d",
    }


def _recommended_tolerance_for_cell(
    cell: dict[str, Any],
    *,
    dtype: torch.dtype,
    policy_info: dict[str, Any],
    fp64_reference_error: float | None,
) -> float:
    formula = policy_info.get("recommended_tolerance_formula", "strict_16d")
    if formula == "layer_depth_aware":
        return layer_depth_aware_streaming_tolerance(
            dtype, int(cell["prefix_layer_count"])
        )
    if formula == "reference_high_precision" and fp64_reference_error is not None:
        return reference_high_precision_tolerance(fp64_reference_error)
    return strict_streaming_tolerance(dtype)


def run_exp070_numerics_cell(
    *,
    model: Any,
    input_ids: torch.Tensor,
    prompt_id: str,
    target_token_length: int,
    actual_token_length: int,
    prefix_layer_count: int,
    chunk_size: int,
    accumulator_mode: str,
    fp64_reference_error: float | None = None,
) -> dict[str, Any]:
    """Run one Experiment 070 numerics audit cell."""
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return _exp070_blocked_cell(
            prompt_id, target_token_length, prefix_layer_count, chunk_size,
            accumulator_mode, ["model has no decoder layers"],
        )
    if prefix_layer_count > len(layers):
        return _exp070_blocked_cell(
            prompt_id, target_token_length, prefix_layer_count, chunk_size,
            accumulator_mode,
            [f"prefix_layer_count {prefix_layer_count} exceeds model depth"],
        )

    rotary_emb = resolve_model_rotary_emb(model)
    seq_len = input_ids.shape[-1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    embed = getattr(getattr(model, "model", model), "embed_tokens", None)
    if embed is None:
        return _exp070_blocked_cell(
            prompt_id, target_token_length, prefix_layer_count, chunk_size,
            accumulator_mode, ["model missing embed_tokens"],
        )

    acc_arg = None if accumulator_mode == "default" else accumulator_mode
    with torch.no_grad():
        hidden0 = embed(input_ids)
        try:
            mat_hidden, _, _ = replay_prefix_layers(
                hidden0,
                layers,
                prefix_layer_count=prefix_layer_count,
                attention_path="materialized_compressed",
                chunk_size=chunk_size,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
            )
            stream_hidden, _, layer_diags = replay_prefix_layers(
                hidden0,
                layers,
                prefix_layer_count=prefix_layer_count,
                attention_path="streaming_compressed",
                chunk_size=chunk_size,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
                streaming_accumulator_dtype=acc_arg,
                collect_streaming_diagnostics=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _exp070_blocked_cell(
                prompt_id, target_token_length, prefix_layer_count, chunk_size,
                accumulator_mode, [f"replay failed: {type(exc).__name__}: {exc}"],
            )

    metrics = compute_drift_metrics(mat_hidden, stream_hidden)
    dtype = hidden0.dtype
    candidate_tols = compute_candidate_tolerances(
        dtype=dtype,
        prefix_layer_count=prefix_layer_count,
        fp64_max_abs_error=fp64_reference_error,
    )
    strict_tol = candidate_tols["strict_16d"]
    strict_pass = metrics.max_abs_error <= strict_tol

    diagnostics: dict[str, Any] = {
        "layer_diagnostics": layer_diags,
        "any_layer_nan": any(d.get("has_nan") for d in layer_diags),
        "any_layer_inf": any(d.get("has_inf") for d in layer_diags),
        "candidate_tolerances": candidate_tols,
    }
    if fp64_reference_error is not None:
        diagnostics["fp64_reference_max_abs_error"] = fp64_reference_error

    return {
        "prompt_id": prompt_id,
        "target_token_length": target_token_length,
        "actual_token_length": actual_token_length,
        "prefix_layer_count": prefix_layer_count,
        "chunk_size": chunk_size,
        "accumulator_mode": accumulator_mode,
        "streaming_vs_materialized_hidden_metrics": metrics.to_dict(),
        "strict_tolerance": strict_tol,
        "strict_tolerance_pass": strict_pass,
        "recommended_tolerance_pass": None,
        "diagnostics": diagnostics,
        "blockers": [],
        "phase16d_regression_target": _is_phase16d_regression_cell(
            {
                "prompt_id": prompt_id,
                "target_token_length": target_token_length,
                "prefix_layer_count": prefix_layer_count,
                "chunk_size": chunk_size,
                "accumulator_mode": accumulator_mode,
            }
        ),
        "fp64_reference_max_abs_error": fp64_reference_error,
    }


def _exp070_blocked_cell(
    prompt_id: str,
    target_token_length: int,
    prefix_layer_count: int,
    chunk_size: int,
    accumulator_mode: str,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "target_token_length": target_token_length,
        "actual_token_length": 0,
        "prefix_layer_count": prefix_layer_count,
        "chunk_size": chunk_size,
        "accumulator_mode": accumulator_mode,
        "streaming_vs_materialized_hidden_metrics": None,
        "strict_tolerance_pass": False,
        "recommended_tolerance_pass": False,
        "diagnostics": None,
        "blockers": blockers,
        "phase16d_regression_target": _is_phase16d_regression_cell(
            {
                "prompt_id": prompt_id,
                "target_token_length": target_token_length,
                "prefix_layer_count": prefix_layer_count,
                "chunk_size": chunk_size,
                "accumulator_mode": accumulator_mode,
            }
        ),
    }


def run_exp070_probe(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    target_token_lengths: Sequence[int] = DEFAULT_TARGET_TOKEN_LENGTHS_070,
    prefix_layer_counts: Sequence[int] = DEFAULT_PREFIX_LAYER_COUNTS,
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES_070,
    accumulator_modes: Sequence[str] = DEFAULT_ACCUMULATOR_MODES,
    max_prompts: int = 2,
    local_files_only: bool = False,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    prompt_provider: Callable[[Any, Sequence[int], int], list[tuple[str, str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 070 streaming multi-layer numerics audit."""
    torch_dtype = getattr(torch, dtype, torch.float32)
    blockers: list[str] = []
    cells: list[dict[str, Any]] = []
    prompts: list[tuple[str, str, int, int]] = []
    fp64_refs: dict[tuple[str, int, int, int], float] = {}

    try:
        if model_loader is not None:
            model, tokenizer = model_loader(
                model_id=model_id, device=device, dtype=torch_dtype,
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

    algorithm_change_made = False

    if load_ok and model is not None and tokenizer is not None:
        model.eval()
        if prompt_provider is not None:
            prompts = prompt_provider(tokenizer, target_token_lengths, max_prompts)
        else:
            prompts = long_context_prompts(
                tokenizer, target_token_lengths, max_prompts=max_prompts
            )

        layers = getattr(getattr(model, "model", model), "layers", None)
        rotary_emb = resolve_model_rotary_emb(model)
        embed = getattr(getattr(model, "model", model), "embed_tokens", None)

        if layers is not None and embed is not None:
            for prompt_id, text, target_len, _actual in prompts:
                input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
                position_ids = torch.arange(input_ids.shape[-1], device=device).unsqueeze(0)
                with torch.no_grad():
                    hidden0 = embed(input_ids)
                    for prefix_layer_count in prefix_layer_counts:
                        for chunk_size in chunk_sizes:
                            try:
                                mat_hidden, _, _ = replay_prefix_layers(
                                    hidden0,
                                    layers,
                                    prefix_layer_count=prefix_layer_count,
                                    attention_path="materialized_compressed",
                                    chunk_size=chunk_size,
                                    rotary_emb=rotary_emb,
                                    position_ids=position_ids,
                                )
                                fp64_hidden, _, _ = replay_prefix_layers(
                                    hidden0,
                                    layers,
                                    prefix_layer_count=prefix_layer_count,
                                    attention_path="streaming_compressed",
                                    chunk_size=chunk_size,
                                    rotary_emb=rotary_emb,
                                    position_ids=position_ids,
                                    streaming_accumulator_dtype="float64",
                                )
                                err = float((mat_hidden - fp64_hidden).abs().max().item())
                                fp64_refs[
                                    (prompt_id, target_len, prefix_layer_count, chunk_size)
                                ] = err
                            except Exception:  # noqa: BLE001
                                continue

        for prompt_id, text, target_len, actual_len in prompts:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            for prefix_layer_count in prefix_layer_counts:
                for chunk_size in chunk_sizes:
                    ref_key = (prompt_id, target_len, prefix_layer_count, chunk_size)
                    fp64_ref = fp64_refs.get(ref_key)
                    for acc_mode in accumulator_modes:
                        cell = run_exp070_numerics_cell(
                            model=model,
                            input_ids=input_ids,
                            prompt_id=prompt_id,
                            target_token_length=target_len,
                            actual_token_length=actual_len,
                            prefix_layer_count=prefix_layer_count,
                            chunk_size=chunk_size,
                            accumulator_mode=acc_mode,
                            fp64_reference_error=fp64_ref,
                        )
                        cell["model_id"] = model_id
                        cells.append(cell)

    successful = [c for c in cells if c.get("streaming_vs_materialized_hidden_metrics")]
    blocked = [c for c in cells if c.get("streaming_vs_materialized_hidden_metrics") is None]

    policy_info = recommend_tolerance_policy(successful, dtype=torch_dtype)
    for cell in successful:
        rec_tol = _recommended_tolerance_for_cell(
            cell,
            dtype=torch_dtype,
            policy_info=policy_info,
            fp64_reference_error=cell.get("fp64_reference_max_abs_error"),
        )
        max_err = cell["streaming_vs_materialized_hidden_metrics"]["max_abs_error"]
        cell["recommended_tolerance"] = rec_tol
        cell["recommended_tolerance_pass"] = max_err <= rec_tol

    strict_pass = sum(1 for c in successful if c.get("strict_tolerance_pass"))
    strict_fail = len(successful) - strict_pass
    rec_pass = sum(1 for c in successful if c.get("recommended_tolerance_pass"))
    rec_fail = len(successful) - rec_pass

    max_by_mode: dict[str, float] = {}
    max_by_depth: dict[str, float] = {}
    max_by_chunk: dict[str, float] = {}
    for c in successful:
        mode = str(c["accumulator_mode"])
        depth = str(c["prefix_layer_count"])
        chunk = str(c["chunk_size"])
        err = c["streaming_vs_materialized_hidden_metrics"]["max_abs_error"]
        max_by_mode[mode] = max(max_by_mode.get(mode, 0.0), err)
        max_by_depth[depth] = max(max_by_depth.get(depth, 0.0), err)
        max_by_chunk[chunk] = max(max_by_chunk.get(chunk, 0.0), err)

    regression_cells = [c for c in successful if c.get("phase16d_regression_target")]
    phase16d_reproduced = False
    phase16d_status = "not_run"
    if regression_cells:
        reg = regression_cells[0]
        err = reg["streaming_vs_materialized_hidden_metrics"]["max_abs_error"]
        old_err = PHASE16D_REGRESSION_CELL["phase16d_observed_error"]
        tol = PHASE16D_REGRESSION_CELL["phase16d_tolerance"]
        phase16d_reproduced = err > tol or abs(err - old_err) < old_err * 0.5
        if reg.get("strict_tolerance_pass"):
            phase16d_status = "passed_after_audit"
        elif err > tol:
            phase16d_status = "reproduced_failure"
        else:
            phase16d_status = "reproduced_within_tolerance"

    if not load_ok:
        status = "blocked"
    elif strict_fail == 0:
        status = "pass"
    elif rec_fail == 0:
        status = "pass_with_recommended_tolerance"
    else:
        status = "failed"

    return {
        "experiment_id": EXPERIMENT_070_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "model_load_succeeded": load_ok,
        "target_token_lengths": list(target_token_lengths),
        "prefix_layer_counts": list(prefix_layer_counts),
        "chunk_sizes": list(chunk_sizes),
        "accumulator_modes": list(accumulator_modes),
        "total_cells": len(cells),
        "successful_cells": len(successful),
        "blocked_cells": len(blocked),
        "failed_cells_under_strict_tolerance": strict_fail,
        "failed_cells_under_recommended_tolerance": rec_fail,
        "phase16d_regression_target": PHASE16D_REGRESSION_CELL,
        "phase16d_failure_reproduced": phase16d_reproduced,
        "phase16d_failure_status_after_audit": phase16d_status,
        "max_error_by_accumulator_mode": max_by_mode,
        "max_error_by_prefix_depth": max_by_depth,
        "max_error_by_chunk_size": max_by_chunk,
        "tolerance_policy_recommendation": policy_info,
        "algorithm_change_made": algorithm_change_made,
        "extraction_blockers": blockers,
        "cells": cells,
        "claim_note": EXP070_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": [
            "Numerical audit of offline streaming attention; not generation integration.",
            "Tolerance recommendations are diagnostic only.",
            "float64 accumulator mode is for CPU audit reference, not production runtime.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "prompt_count": len(prompts),
    }


def validate_exp070_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "target_token_lengths",
        "prefix_layer_counts",
        "chunk_sizes",
        "accumulator_modes",
        "total_cells",
        "successful_cells",
        "failed_cells_under_strict_tolerance",
        "failed_cells_under_recommended_tolerance",
        "phase16d_failure_reproduced",
        "phase16d_failure_status_after_audit",
        "max_error_by_accumulator_mode",
        "max_error_by_prefix_depth",
        "max_error_by_chunk_size",
        "tolerance_policy_recommendation",
        "algorithm_change_made",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "forbidden_claims",
        "cells",
        "phase16d_regression_target",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_070_ID:
        errors.append("experiment_id mismatch")

    cells = report.get("cells")
    if not isinstance(cells, list):
        errors.append("cells must be a list")
        return errors

    has_regression_marker = any(
        isinstance(c, dict) and c.get("phase16d_regression_target") for c in cells
    )
    if not has_regression_marker:
        errors.append("missing phase16d_regression_target cell marker")

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {idx} not dict")
            continue
        for ck in (
            "prompt_id",
            "target_token_length",
            "prefix_layer_count",
            "chunk_size",
            "accumulator_mode",
            "strict_tolerance_pass",
            "recommended_tolerance_pass",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"cell {idx} missing {ck}")

    return errors
