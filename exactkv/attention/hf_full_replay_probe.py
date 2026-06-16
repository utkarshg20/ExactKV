"""Offline full-prefix logit drift smoke (Phase 16F).

Replays the entire Qwen2/Qwen2.5 decoder stack with full, materialized-compressed,
and streaming-compressed attention, then compares final hidden states and
next-token logits. **Not** wired into ExactKV generation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from exactkv.attention.hf_multilayer_probe import (
    AttentionPath,
    LayerMemoryRecord,
    aggregate_layer_memory,
    replay_prefix_layers,
)
from exactkv.attention.hf_single_layer_probe import (
    DEFAULT_MODEL_ID,
    FORBIDDEN_ATTENTION_CLAIMS,
    compute_drift_metrics,
    long_context_prompts,
    resolve_model_rotary_emb,
)
from exactkv.attention.streaming_quant_attention import (
    layer_depth_aware_streaming_tolerance,
)

EXPERIMENT_071_ID = "exp071_full_prefix_logit_drift_smoke"
DEFAULT_EXP071_REPORT = Path("reports/experiment_071_full_prefix_logit_drift_smoke.json")
DEFAULT_TARGET_TOKEN_LENGTHS_071: tuple[int, ...] = (32, 64, 128)
DEFAULT_CHUNK_SIZES_071: tuple[int, ...] = (16, 32, 64)
DEFAULT_ACCUMULATOR_MODE_071 = "float32"

EXP071_CLAIM_NOTE = (
    "Offline full-prefix logit drift smoke (Phase 16F). Replays the full Qwen2/Qwen2.5 "
    "decoder stack with compressed attention paths and compares next-token logits for "
    "fixed prompts. Not model generation integration, vLLM, CUDA/Triton kernels, or "
    "ExactKV default runtime. Full-model parity is measured before interpreting drift. "
    "Theoretical memory accounting only; no measured active GPU memory, speed, "
    "throughput, latency, or serving claim."
)

DEFAULT_FULL_MODEL_PARITY_TOLERANCE_FP32 = 1e-2
DEFAULT_FULL_MODEL_LOGIT_PARITY_TOLERANCE_FP32 = 1e-1


@dataclass
class LogitDriftMetrics:
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    relative_l2_error: float
    top1_agreement: bool
    top5_overlap: int
    top10_overlap: int
    reference_top1_token_id: int
    other_top1_token_id: int
    reference_top1_probability: float | None
    other_top1_probability: float | None
    reference_logit_margin_top12: float
    other_logit_margin_top12: float
    top1_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_final_norm(model: Any) -> Any | None:
    inner = getattr(model, "model", model)
    return getattr(inner, "norm", None)


def _resolve_lm_head(model: Any) -> Any | None:
    return getattr(model, "lm_head", None)


def _num_decoder_layers(model: Any) -> int:
    layers = getattr(getattr(model, "model", model), "layers", None)
    return len(layers) if layers is not None else 0


def _last_position_logits(hidden: torch.Tensor, norm: Any, lm_head: Any) -> torch.Tensor:
    """Apply final norm and lm_head; return next-token logits at last position."""
    normed = norm(hidden)
    logits = lm_head(normed)
    return logits[:, -1, :]


def _safe_top1_probability(logits: torch.Tensor) -> float | None:
    if not torch.isfinite(logits).all():
        return None
    probs = F.softmax(logits.float().squeeze(), dim=-1)
    if not torch.isfinite(probs).all():
        return None
    return float(probs.max().item())


def _logit_margin_top12(logits: torch.Tensor) -> float:
    vec = logits.float().squeeze()
    top2 = torch.topk(vec, min(2, vec.numel()))
    if top2.values.numel() < 2:
        return float(top2.values[0].item()) if top2.values.numel() else 0.0
    return float((top2.values[0] - top2.values[1]).item())


def _topk_token_ids(logits: torch.Tensor, k: int) -> list[int]:
    vec = logits.squeeze()
    k = min(k, vec.numel())
    return torch.topk(vec, k).indices.tolist()


def compute_logit_drift_metrics(
    reference_logits: torch.Tensor,
    other_logits: torch.Tensor,
) -> LogitDriftMetrics:
    """Compare next-token logits (last position, shape [B, vocab])."""
    if reference_logits.shape != other_logits.shape:
        raise ValueError(
            f"logit shape mismatch: {reference_logits.shape} vs {other_logits.shape}"
        )
    ref = reference_logits.detach()
    oth = other_logits.detach()
    drift = compute_drift_metrics(ref, oth)
    ref_top1 = _topk_token_ids(ref, 1)[0]
    oth_top1 = _topk_token_ids(oth, 1)[0]
    ref_top5 = _topk_token_ids(ref, 5)
    oth_top5 = _topk_token_ids(oth, 5)
    ref_top10 = _topk_token_ids(ref, 10)
    oth_top10 = _topk_token_ids(oth, 10)
    return LogitDriftMetrics(
        max_abs_error=drift.max_abs_error,
        mean_abs_error=drift.mean_abs_error,
        cosine_similarity=drift.cosine_similarity,
        relative_l2_error=drift.relative_l2_error,
        top1_agreement=ref_top1 == oth_top1,
        top5_overlap=len(set(ref_top5) & set(oth_top5)),
        top10_overlap=len(set(ref_top10) & set(oth_top10)),
        reference_top1_token_id=ref_top1,
        other_top1_token_id=oth_top1,
        reference_top1_probability=_safe_top1_probability(ref),
        other_top1_probability=_safe_top1_probability(oth),
        reference_logit_margin_top12=_logit_margin_top12(ref),
        other_logit_margin_top12=_logit_margin_top12(oth),
        top1_changed=ref_top1 != oth_top1,
    )


def replay_full_decoder_stack(
    model: Any,
    input_ids: torch.Tensor,
    *,
    attention_path: AttentionPath,
    chunk_size: int,
    streaming_accumulator_dtype: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[LayerMemoryRecord]]:
    """Replay all decoder layers, apply norm + lm_head; return hidden, last logits, memory."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    norm = _resolve_final_norm(model)
    lm_head = _resolve_lm_head(model)
    embed = getattr(inner, "embed_tokens", None)
    if layers is None or norm is None or lm_head is None or embed is None:
        raise RuntimeError("model missing layers, norm, lm_head, or embed_tokens")

    rotary_emb = resolve_model_rotary_emb(model)
    seq_len = input_ids.shape[-1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    with torch.no_grad():
        hidden0 = embed(input_ids)
        hidden, mem_records, _ = replay_prefix_layers(
            hidden0,
            layers,
            prefix_layer_count=len(layers),
            attention_path=attention_path,
            chunk_size=chunk_size,
            rotary_emb=rotary_emb,
            position_ids=position_ids,
            streaming_accumulator_dtype=streaming_accumulator_dtype,
        )
        logits = _last_position_logits(hidden, norm, lm_head)
    return hidden, logits, mem_records


def check_full_model_parity(
    manual_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    manual_hidden: torch.Tensor,
    hf_hidden: torch.Tensor,
    *,
    hidden_tolerance: float = DEFAULT_FULL_MODEL_PARITY_TOLERANCE_FP32,
    logit_tolerance: float = DEFAULT_FULL_MODEL_LOGIT_PARITY_TOLERANCE_FP32,
) -> dict[str, Any]:
    """Compare manual full-path replay against HF reference."""
    logit_metrics = compute_logit_drift_metrics(hf_logits, manual_logits)
    hidden_metrics = compute_drift_metrics(hf_hidden, manual_hidden)
    logit_ok = logit_metrics.max_abs_error <= logit_tolerance
    hidden_ok = hidden_metrics.max_abs_error <= hidden_tolerance
    status = "passed" if (logit_ok and hidden_ok) else "failed"
    failures: list[str] = []
    if not logit_ok:
        failures.append(f"logit_max_abs={logit_metrics.max_abs_error}")
    if not hidden_ok:
        failures.append(f"hidden_max_abs={hidden_metrics.max_abs_error}")
    return {
        "full_model_parity_status": status,
        "full_model_parity_metrics": {
            "logits": logit_metrics.to_dict(),
            "hidden_last_position_post_norm": hidden_metrics.to_dict(),
            "top1_agreement": logit_metrics.top1_agreement,
            "top5_overlap": logit_metrics.top5_overlap,
        },
        "hidden_tolerance": hidden_tolerance,
        "logit_tolerance": logit_tolerance,
        "parity_failures": failures,
    }


def aggregate_full_stack_memory(
    mem_records: Sequence[LayerMemoryRecord],
    *,
    context_length: int,
) -> dict[str, Any]:
    """Theoretical aggregate memory across all replayed layers."""
    agg = aggregate_layer_memory(mem_records)
    metadata = sum(r.metadata_bytes for r in mem_records)
    chunk_size = mem_records[0].chunk_size if mem_records else 0
    num_layers = len(mem_records)
    return {
        "aggregate_full_kv_bytes": agg["aggregate_full_kv_bytes"],
        "aggregate_stored_quantized_kv_bytes": agg["aggregate_stored_quantized_kv_bytes"],
        "aggregate_materialized_working_kv_bytes": agg["aggregate_materialized_working_kv_bytes"],
        "aggregate_streaming_peak_working_kv_bytes_conservative": agg[
            "aggregate_streaming_peak_working_kv_bytes_conservative"
        ],
        "metadata_bytes": metadata,
        "chunk_size": chunk_size,
        "num_layers": num_layers,
        "context_length": context_length,
        "best_theoretical_streaming_reduction": agg["best_theoretical_streaming_reduction"],
    }


def _depth_aware_tolerance(dtype: torch.dtype, num_layers: int) -> float:
    return layer_depth_aware_streaming_tolerance(dtype, num_layers)


def run_exp071_logit_cell(
    *,
    model: Any,
    input_ids: torch.Tensor,
    hf_logits: torch.Tensor,
    hf_hidden: torch.Tensor,
    prompt_id: str,
    target_token_length: int,
    actual_token_length: int,
    chunk_size: int,
    accumulator_mode: str = DEFAULT_ACCUMULATOR_MODE_071,
    allow_parity_fail: bool = False,
) -> dict[str, Any]:
    """Run one full-prefix logit drift smoke cell."""
    blockers: list[str] = []
    num_layers = _num_decoder_layers(model)
    if num_layers == 0:
        return _blocked_cell_071(
            prompt_id, target_token_length, actual_token_length, chunk_size, blockers=[
                "model has no decoder layers"
            ],
        )

    acc_arg = None if accumulator_mode == "default" else accumulator_mode
    dtype = hf_logits.dtype

    try:
        full_hidden, full_logits, full_mem = replay_full_decoder_stack(
            model, input_ids, attention_path="full", chunk_size=chunk_size,
        )
        mat_hidden, mat_logits, mat_mem = replay_full_decoder_stack(
            model, input_ids, attention_path="materialized_compressed", chunk_size=chunk_size,
        )
        stream_hidden, stream_logits, stream_mem = replay_full_decoder_stack(
            model,
            input_ids,
            attention_path="streaming_compressed",
            chunk_size=chunk_size,
            streaming_accumulator_dtype=acc_arg,
        )
    except Exception as exc:  # noqa: BLE001
        return _blocked_cell_071(
            prompt_id, target_token_length, actual_token_length, chunk_size,
            blockers=[f"replay failed: {type(exc).__name__}: {exc}"],
        )

    norm = _resolve_final_norm(model)
    if norm is not None:
        manual_hidden_parity = norm(full_hidden[:, -1:, :])
    else:
        manual_hidden_parity = full_hidden[:, -1:, :]
    hf_hidden_parity = hf_hidden[:, -1:, :]

    parity = check_full_model_parity(
        full_logits, hf_logits, manual_hidden_parity, hf_hidden_parity,
    )
    parity_status = parity["full_model_parity_status"]
    if parity_status == "failed" and not allow_parity_fail:
        failures = parity.get("parity_failures", [])
        blockers.append(
            "full_model_parity failed: " + ", ".join(failures) if failures else "unknown"
        )

    tol = _depth_aware_tolerance(dtype, num_layers)

    sm_hidden = compute_drift_metrics(mat_hidden, stream_hidden)
    sm_logit = compute_logit_drift_metrics(mat_logits, stream_logits)
    fs_hidden = compute_drift_metrics(full_hidden, stream_hidden)
    fs_logit = compute_logit_drift_metrics(full_logits, stream_logits)
    fm_logit = compute_logit_drift_metrics(full_logits, mat_logits)

    # Primary pass: streaming logits vs materialized logits (Phase 16F criterion).
    streaming_pass = sm_logit.max_abs_error <= tol
    parity_ok = parity_status == "passed"
    passed = streaming_pass and (parity_ok or allow_parity_fail)

    mem_acct = aggregate_full_stack_memory(
        stream_mem, context_length=actual_token_length,
    )

    return {
        "prompt_id": prompt_id,
        "target_token_length": target_token_length,
        "actual_token_length": actual_token_length,
        "chunk_size": chunk_size,
        "accumulator_mode": accumulator_mode,
        "num_layers_replayed": num_layers,
        "depth_aware_tolerance": tol,
        "full_model_parity_status": parity_status,
        "full_model_parity_metrics": parity["full_model_parity_metrics"],
        "streaming_vs_materialized_hidden_metrics": sm_hidden.to_dict(),
        "streaming_vs_materialized_logit_metrics": sm_logit.to_dict(),
        "full_vs_streaming_hidden_metrics": fs_hidden.to_dict(),
        "full_vs_streaming_logit_metrics": fs_logit.to_dict(),
        "full_vs_materialized_logit_metrics": fm_logit.to_dict(),
        "full_top1_token_id": fs_logit.reference_top1_token_id,
        "streaming_top1_token_id": fs_logit.other_top1_token_id,
        "materialized_top1_token_id": sm_logit.reference_top1_token_id,
        "full_top5_token_ids": _topk_token_ids(full_logits, 5),
        "streaming_top5_token_ids": _topk_token_ids(stream_logits, 5),
        "materialized_top5_token_ids": _topk_token_ids(mat_logits, 5),
        "top1_changed_full_vs_streaming": fs_logit.top1_changed,
        "top5_overlap_full_vs_streaming": fs_logit.top5_overlap,
        "top5_overlap_streaming_vs_materialized": sm_logit.top5_overlap,
        "memory_accounting": mem_acct,
        "streaming_passed": streaming_pass,
        "passed": passed,
        "blockers": blockers,
    }


def _blocked_cell_071(
    prompt_id: str,
    target_token_length: int,
    actual_token_length: int,
    chunk_size: int,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "target_token_length": target_token_length,
        "actual_token_length": actual_token_length,
        "chunk_size": chunk_size,
        "full_model_parity_status": "blocked",
        "full_model_parity_metrics": None,
        "streaming_vs_materialized_hidden_metrics": None,
        "streaming_vs_materialized_logit_metrics": None,
        "full_vs_streaming_hidden_metrics": None,
        "full_vs_streaming_logit_metrics": None,
        "full_vs_materialized_logit_metrics": None,
        "full_top1_token_id": None,
        "streaming_top1_token_id": None,
        "materialized_top1_token_id": None,
        "full_top5_token_ids": None,
        "streaming_top5_token_ids": None,
        "materialized_top5_token_ids": None,
        "top1_changed_full_vs_streaming": None,
        "top5_overlap_full_vs_streaming": None,
        "top5_overlap_streaming_vs_materialized": None,
        "memory_accounting": None,
        "streaming_passed": False,
        "passed": False,
        "blockers": blockers,
    }


def run_exp071_probe(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    target_token_lengths: Sequence[int] = DEFAULT_TARGET_TOKEN_LENGTHS_071,
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES_071,
    max_prompts: int = 2,
    accumulator_mode: str = DEFAULT_ACCUMULATOR_MODE_071,
    local_files_only: bool = False,
    allow_parity_fail: bool = False,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    prompt_provider: Callable[[Any, Sequence[int], int], list[tuple[str, str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 071 full-prefix logit drift smoke."""
    torch_dtype = getattr(torch, dtype, torch.float32)
    blockers: list[str] = []
    cells: list[dict[str, Any]] = []
    prompts: list[tuple[str, str, int, int]] = []

    try:
        if model_loader is not None:
            model, tokenizer = model_loader(
                model_id=model_id, device=device, dtype=torch_dtype,
                local_files_only=local_files_only,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, local_files_only=local_files_only,
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

    num_layers = 0
    if load_ok and model is not None and tokenizer is not None:
        model.eval()
        num_layers = _num_decoder_layers(model)
        if prompt_provider is not None:
            prompts = prompt_provider(tokenizer, target_token_lengths, max_prompts)
        else:
            prompts = long_context_prompts(
                tokenizer, target_token_lengths, max_prompts=max_prompts,
            )

        for prompt_id, text, target_len, actual_len in prompts:
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            with torch.no_grad():
                hf_out = model(input_ids, output_hidden_states=True, use_cache=False)
            hf_logits = hf_out.logits[:, -1, :]
            hf_hs = hf_out.hidden_states
            hf_hidden = hf_hs[-1] if hf_hs is not None else torch.zeros(1)

            for chunk_size in chunk_sizes:
                cell = run_exp071_logit_cell(
                    model=model,
                    input_ids=input_ids,
                    hf_logits=hf_logits,
                    hf_hidden=hf_hidden,
                    prompt_id=prompt_id,
                    target_token_length=target_len,
                    actual_token_length=actual_len,
                    chunk_size=chunk_size,
                    accumulator_mode=accumulator_mode,
                    allow_parity_fail=allow_parity_fail,
                )
                cell["model_id"] = model_id
                cells.append(cell)

    successful = [c for c in cells if c.get("streaming_vs_materialized_logit_metrics")]
    blocked = [c for c in cells if c.get("streaming_vs_materialized_logit_metrics") is None]
    parity_pass = sum(
        1 for c in successful if c.get("full_model_parity_status") == "passed"
    )
    stream_pass = sum(1 for c in successful if c.get("streaming_passed"))
    top1_changed = sum(
        1 for c in successful if c.get("top1_changed_full_vs_streaming")
    )

    max_sm_hidden = max(
        (c["streaming_vs_materialized_hidden_metrics"]["max_abs_error"] for c in successful),
        default=0.0,
    )
    max_sm_logit = max(
        (c["streaming_vs_materialized_logit_metrics"]["max_abs_error"] for c in successful),
        default=0.0,
    )

    fs_logit_max = max(
        (c["full_vs_streaming_logit_metrics"]["max_abs_error"] for c in successful),
        default=0.0,
    )
    fs_logit_mean = (
        sum(c["full_vs_streaming_logit_metrics"]["mean_abs_error"] for c in successful)
        / len(successful)
        if successful
        else 0.0
    )

    top5_overlaps_sm: list[int] = []
    top5_overlaps_fs: list[int] = []
    top10_overlaps_sm: list[int] = []
    reductions: list[float] = []
    actual_lengths: list[int] = []

    for c in successful:
        sm = c["streaming_vs_materialized_logit_metrics"]
        fs = c["full_vs_streaming_logit_metrics"]
        top5_overlaps_sm.append(int(sm.get("top5_overlap", 0)))
        top5_overlaps_fs.append(int(fs.get("top5_overlap", 0)))
        top10_overlaps_sm.append(int(sm.get("top10_overlap", 0)))
        mem = c.get("memory_accounting", {})
        if isinstance(mem, dict):
            reductions.append(float(mem.get("best_theoretical_streaming_reduction", 0.0)))
        actual_lengths.append(int(c.get("actual_token_length", 0)))

    if not load_ok:
        status = "blocked"
    elif successful and all(c.get("passed") for c in successful):
        status = "pass"
    elif successful:
        status = "failed"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_071_ID,
        "status": status,
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "model_load_succeeded": load_ok,
        "target_token_lengths": list(target_token_lengths)[:max_prompts],
        "chunk_sizes": list(chunk_sizes),
        "accumulator_mode": accumulator_mode,
        "total_cells": len(cells),
        "successful_cells": len(successful),
        "blocked_cells": len(blocked),
        "full_model_parity_pass_cells": parity_pass,
        "streaming_vs_materialized_pass_cells": stream_pass,
        "compressed_top1_changed_cells": top1_changed,
        "max_streaming_vs_materialized_logit_error": max_sm_logit,
        "max_streaming_vs_materialized_hidden_error": max_sm_hidden,
        "full_vs_streaming_logit_drift_summary": {
            "max_abs_error": fs_logit_max,
            "mean_abs_error": fs_logit_mean,
            "cell_count": len(successful),
        },
        "top1_change_summary": {
            "cells_with_top1_change_full_vs_streaming": top1_changed,
            "cell_count": len(successful),
        },
        "topk_overlap_summary": {
            "streaming_vs_materialized_top5_mean": (
                sum(top5_overlaps_sm) / len(top5_overlaps_sm) if top5_overlaps_sm else 0.0
            ),
            "full_vs_streaming_top5_mean": (
                sum(top5_overlaps_fs) / len(top5_overlaps_fs) if top5_overlaps_fs else 0.0
            ),
            "streaming_vs_materialized_top10_mean": (
                sum(top10_overlaps_sm) / len(top10_overlaps_sm) if top10_overlaps_sm else 0.0
            ),
        },
        "memory_accounting_summary": {
            "best_theoretical_streaming_reduction": max(reductions) if reductions else 0.0,
            "worst_theoretical_streaming_reduction": min(reductions) if reductions else 0.0,
        },
        "longest_context_tested": max(actual_lengths) if actual_lengths else 0,
        "num_layers_replayed": num_layers,
        "allow_parity_fail": allow_parity_fail,
        "extraction_blockers": blockers,
        "cells": cells,
        "claim_note": EXP071_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": [
            "Offline full-prefix logit smoke; not token generation integration.",
            "Computing logits for fixed prompts is not the same as generating tokens.",
            "Full-model parity required before interpreting drift unless --allow-parity-fail.",
            "Theoretical memory accounting only.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "prompt_count": len(prompts),
    }


def validate_exp071_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_id",
        "device",
        "dtype",
        "target_token_lengths",
        "chunk_sizes",
        "total_cells",
        "successful_cells",
        "blocked_cells",
        "full_model_parity_pass_cells",
        "streaming_vs_materialized_pass_cells",
        "compressed_top1_changed_cells",
        "max_streaming_vs_materialized_logit_error",
        "max_streaming_vs_materialized_hidden_error",
        "full_vs_streaming_logit_drift_summary",
        "top1_change_summary",
        "topk_overlap_summary",
        "memory_accounting_summary",
        "longest_context_tested",
        "num_layers_replayed",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "forbidden_claims",
        "cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_071_ID:
        errors.append("experiment_id mismatch")

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
            "chunk_size",
            "full_model_parity_status",
            "passed",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"cell {idx} missing {ck}")

    return errors
