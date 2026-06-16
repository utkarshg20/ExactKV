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
    estimate_attention_memory_bytes,
    quantize_kv_int8_reference,
)

EXPERIMENT_069_ID = "exp069_multilayer_attention_drift_accumulation"
DEFAULT_EXP069_REPORT = Path("reports/experiment_069_multilayer_attention_drift_accumulation.json")
DEFAULT_PREFIX_LAYER_COUNTS: tuple[int, ...] = (1, 2, 4)
DEFAULT_TARGET_TOKEN_LENGTHS_069: tuple[int, ...] = (64, 128)

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
) -> torch.Tensor:
    q, k, v = extracted_qkv.q, extracted_qkv.k, extracted_qkv.v
    if path == "full":
        return attention_full(q, k, v, causal=True)
    qkv = quantize_kv_int8_reference(k, v)
    if path == "materialized_compressed":
        return attention_materialized_compressed(q, qkv, causal=True)
    return attention_streaming_compressed(q, qkv, chunk_size, causal=True)


def run_qwen_decoder_block(
    hidden: torch.Tensor,
    layer: Any,
    *,
    layer_idx: int,
    attention_path: AttentionPath,
    chunk_size: int,
    rotary_emb: Any | None,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, LayerMemoryRecord | None]:
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

    attn_ctx = _attention_context(extracted, path=attention_path, chunk_size=chunk_size)
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
    return hidden, mem


def replay_prefix_layers(
    hidden: torch.Tensor,
    layers: Sequence[Any],
    *,
    prefix_layer_count: int,
    attention_path: AttentionPath,
    chunk_size: int,
    rotary_emb: Any | None,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[LayerMemoryRecord]]:
    """Run the first ``prefix_layer_count`` decoder layers."""
    mem_records: list[LayerMemoryRecord] = []
    for idx in range(prefix_layer_count):
        hidden, mem = run_qwen_decoder_block(
            hidden,
            layers[idx],
            layer_idx=idx,
            attention_path=attention_path,
            chunk_size=chunk_size,
            rotary_emb=rotary_emb,
            position_ids=position_ids,
        )
        mem_records.append(mem)
    return hidden, mem_records


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
            full_hidden, full_mem = replay_prefix_layers(
                hidden0,
                layers,
                prefix_layer_count=prefix_layer_count,
                attention_path="full",
                chunk_size=chunk_size,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
            )
            mat_hidden, mat_mem = replay_prefix_layers(
                hidden0,
                layers,
                prefix_layer_count=prefix_layer_count,
                attention_path="materialized_compressed",
                chunk_size=chunk_size,
                rotary_emb=rotary_emb,
                position_ids=position_ids,
            )
            stream_hidden, stream_mem = replay_prefix_layers(
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
