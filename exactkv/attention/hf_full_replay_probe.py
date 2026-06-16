"""Offline full-prefix logit drift smoke (Phase 16F).

Replays the entire Qwen2/Qwen2.5 decoder stack with full, materialized-compressed,
and streaming-compressed attention, then compares final hidden states and
next-token logits. **Not** wired into ExactKV generation.
"""
from __future__ import annotations

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


# --- Phase 16G: full-depth divergence trace ---

EXPERIMENT_072_ID = "exp072_full_depth_divergence_trace"
DEFAULT_EXP072_REPORT = Path("reports/experiment_072_full_depth_divergence_trace.json")
DEFAULT_TARGET_TOKEN_LENGTHS_072: tuple[int, ...] = (32, 64)
DEFAULT_CHUNK_SIZES_072: tuple[int, ...] = (16, 32, 64)
TRACE_MODES: tuple[str, ...] = ("free_running", "teacher_forced_layer_inputs")
TRACE_CHECKPOINTS: tuple[str, ...] = (
    "layer_input",
    "attn_context",
    "attn_output",
    "post_attention_hidden",
    "post_mlp_hidden",
)
PHASE16F_LOGIT_TOLERANCE_REF = 0.002449489742783178

EXP072_CLAIM_NOTE = (
    "Offline full-depth streaming/materialized divergence trace (Phase 16G). "
    "Layer-by-layer diagnostic trace for fixed prompts. Not model generation "
    "integration, vLLM, CUDA/Triton kernels, or ExactKV default runtime. "
    "Top-k agreement is supplementary only. Theoretical memory accounting only; "
    "no measured active GPU memory, speed, throughput, latency, or serving claim."
)


def _checkpoint_metrics(
    materialized: torch.Tensor,
    streaming: torch.Tensor,
) -> dict[str, float]:
    m = compute_drift_metrics(materialized, streaming)
    return {
        "max_abs_error": m.max_abs_error,
        "mean_abs_error": m.mean_abs_error,
        "relative_l2_error": m.relative_l2_error,
        "cosine_similarity": m.cosine_similarity,
    }


def _first_layer_exceeding(
    per_layer_trace: Sequence[dict[str, Any]],
    threshold: float,
    *,
    checkpoint: str = "post_mlp_hidden",
) -> int | None:
    for entry in per_layer_trace:
        metrics = entry.get("streaming_vs_materialized", {}).get(checkpoint)
        if metrics is None:
            continue
        if metrics["max_abs_error"] > threshold:
            return int(entry["layer_idx"])
    return None


def classify_divergence_root_cause(
    *,
    teacher_forced_trace: Sequence[dict[str, Any]],
    free_running_trace: Sequence[dict[str, Any]],
    final_logit_max_abs: float,
    depth_aware_tolerance: float,
    final_top1_agreement: bool,
) -> str:
    """Data-driven root cause classification for one prompt/chunk pair."""

    def _max_at(trace: Sequence[dict[str, Any]], ckpt: str) -> float:
        vals = [
            e["streaming_vs_materialized"][ckpt]["max_abs_error"]
            for e in trace
            if ckpt in e.get("streaming_vs_materialized", {})
        ]
        return max(vals) if vals else 0.0

    tf_attn = _max_at(teacher_forced_trace, "attn_context")
    tf_attn_out = _max_at(teacher_forced_trace, "attn_output")
    tf_post_attn = _max_at(teacher_forced_trace, "post_attention_hidden")
    tf_post_mlp = _max_at(teacher_forced_trace, "post_mlp_hidden")
    fr_post_mlp = _max_at(free_running_trace, "post_mlp_hidden")

    if tf_attn > 1e-3:
        return "local_attention_mismatch"
    if tf_attn <= 1e-4 and tf_post_attn > 1e-2 and tf_post_attn > tf_attn_out * 10:
        return "post_attention_amplification"
    if tf_attn <= 1e-4 and tf_post_mlp > 1e-2 and tf_post_mlp > tf_attn * 100:
        return "mlp_residual_amplification"
    if tf_post_mlp < 1e-3 and fr_post_mlp > 1e-2 and fr_post_mlp > tf_post_mlp * 10:
        return "free_running_accumulation"
    if (
        final_logit_max_abs > depth_aware_tolerance
        and final_top1_agreement
        and tf_attn < 1e-3
    ):
        return "tolerance_policy_issue"
    if tf_attn > 1e-4:
        return "local_attention_mismatch"
    if fr_post_mlp > tf_post_mlp * 5 and tf_post_mlp < 1e-3:
        return "free_running_accumulation"
    return "unknown"


def trace_mat_vs_stream_full_depth(
    model: Any,
    input_ids: torch.Tensor,
    *,
    chunk_size: int,
    trace_mode: str,
    streaming_accumulator_dtype: str | None = "float32",
) -> dict[str, Any]:
    """Run layer-by-layer materialized vs streaming divergence trace."""
    if trace_mode not in TRACE_MODES:
        raise ValueError(f"unknown trace_mode: {trace_mode}")

    from exactkv.attention.hf_multilayer_probe import run_qwen_decoder_block_traced

    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    embed = getattr(inner, "embed_tokens", None)
    norm = _resolve_final_norm(model)
    lm_head = _resolve_lm_head(model)
    if layers is None or embed is None or norm is None or lm_head is None:
        raise RuntimeError("model missing layers, embed_tokens, norm, or lm_head")

    rotary_emb = resolve_model_rotary_emb(model)
    seq_len = input_ids.shape[-1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    num_layers = len(layers)

    with torch.no_grad():
        mat_hidden = embed(input_ids)
        stream_hidden = embed(input_ids)
        per_layer: list[dict[str, Any]] = []

        for idx, layer in enumerate(layers):
            if trace_mode == "teacher_forced_layer_inputs":
                shared_input = mat_hidden
                mat_hidden, mat_cp = run_qwen_decoder_block_traced(
                    shared_input,
                    layer,
                    layer_idx=idx,
                    attention_path="materialized_compressed",
                    chunk_size=chunk_size,
                    rotary_emb=rotary_emb,
                    position_ids=position_ids,
                )
                _, stream_cp = run_qwen_decoder_block_traced(
                    shared_input,
                    layer,
                    layer_idx=idx,
                    attention_path="streaming_compressed",
                    chunk_size=chunk_size,
                    rotary_emb=rotary_emb,
                    position_ids=position_ids,
                    streaming_accumulator_dtype=streaming_accumulator_dtype,
                )
            else:
                mat_hidden, mat_cp = run_qwen_decoder_block_traced(
                    mat_hidden,
                    layer,
                    layer_idx=idx,
                    attention_path="materialized_compressed",
                    chunk_size=chunk_size,
                    rotary_emb=rotary_emb,
                    position_ids=position_ids,
                )
                stream_hidden, stream_cp = run_qwen_decoder_block_traced(
                    stream_hidden,
                    layer,
                    layer_idx=idx,
                    attention_path="streaming_compressed",
                    chunk_size=chunk_size,
                    rotary_emb=rotary_emb,
                    position_ids=position_ids,
                    streaming_accumulator_dtype=streaming_accumulator_dtype,
                )

            layer_metrics: dict[str, dict[str, float]] = {}
            for ckpt in TRACE_CHECKPOINTS:
                layer_metrics[ckpt] = _checkpoint_metrics(mat_cp[ckpt], stream_cp[ckpt])

            per_layer.append({
                "layer_idx": idx,
                "streaming_vs_materialized": layer_metrics,
            })

        mat_logits = _last_position_logits(mat_hidden, norm, lm_head)
        stream_logits = _last_position_logits(stream_hidden, norm, lm_head)

    final_hidden_metrics = _checkpoint_metrics(mat_hidden, stream_hidden)
    final_logit_metrics = compute_logit_drift_metrics(mat_logits, stream_logits)

    return {
        "trace_mode": trace_mode,
        "per_layer_trace": per_layer,
        "final_hidden_metrics": final_hidden_metrics,
        "final_logit_metrics": final_logit_metrics.to_dict(),
        "final_top1_agreement": final_logit_metrics.top1_agreement,
        "final_top5_overlap": final_logit_metrics.top5_overlap,
        "final_top10_overlap": final_logit_metrics.top10_overlap,
        "num_layers": num_layers,
    }


def run_exp072_trace_cell(
    *,
    model: Any,
    input_ids: torch.Tensor,
    prompt_id: str,
    target_token_length: int,
    actual_token_length: int,
    chunk_size: int,
    trace_mode: str,
    accumulator_mode: str = DEFAULT_ACCUMULATOR_MODE_071,
    teacher_forced_trace: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one divergence trace cell."""
    acc_arg = None if accumulator_mode == "default" else accumulator_mode
    num_layers = _num_decoder_layers(model)
    depth_tol = layer_depth_aware_streaming_tolerance(torch.float32, num_layers)

    try:
        trace = trace_mat_vs_stream_full_depth(
            model,
            input_ids,
            chunk_size=chunk_size,
            trace_mode=trace_mode,
            streaming_accumulator_dtype=acc_arg,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "prompt_id": prompt_id,
            "target_token_length": target_token_length,
            "actual_token_length": actual_token_length,
            "chunk_size": chunk_size,
            "trace_mode": trace_mode,
            "per_layer_trace": None,
            "first_layer_exceeding_1e_4": None,
            "first_layer_exceeding_1e_3": None,
            "first_layer_exceeding_1e_2": None,
            "first_layer_exceeding_1e_1": None,
            "final_hidden_metrics": None,
            "final_logit_metrics": None,
            "final_top1_agreement": None,
            "final_top5_overlap": None,
            "final_top10_overlap": None,
            "root_cause_classification": "unknown",
            "passed": False,
            "blockers": [f"trace failed: {type(exc).__name__}: {exc}"],
        }

    per_layer = trace["per_layer_trace"]
    final_logit = trace["final_logit_metrics"]
    final_logit_max = float(final_logit["max_abs_error"])

    thresholds = {
        "1e_4": _first_layer_exceeding(per_layer, 1e-4),
        "1e_3": _first_layer_exceeding(per_layer, 1e-3),
        "1e_2": _first_layer_exceeding(per_layer, 1e-2),
        "1e_1": _first_layer_exceeding(per_layer, 1e-1),
    }

    root_cause = "unknown"
    if trace_mode == "free_running" and teacher_forced_trace is not None:
        root_cause = classify_divergence_root_cause(
            teacher_forced_trace=teacher_forced_trace,
            free_running_trace=per_layer,
            final_logit_max_abs=final_logit_max,
            depth_aware_tolerance=depth_tol,
            final_top1_agreement=bool(trace["final_top1_agreement"]),
        )
    elif trace_mode == "teacher_forced_layer_inputs":
        root_cause = classify_divergence_root_cause(
            teacher_forced_trace=per_layer,
            free_running_trace=per_layer,
            final_logit_max_abs=final_logit_max,
            depth_aware_tolerance=depth_tol,
            final_top1_agreement=bool(trace["final_top1_agreement"]),
        )

    streaming_pass = final_logit_max <= depth_tol

    return {
        "prompt_id": prompt_id,
        "target_token_length": target_token_length,
        "actual_token_length": actual_token_length,
        "chunk_size": chunk_size,
        "trace_mode": trace_mode,
        "per_layer_trace": per_layer,
        "first_layer_exceeding_1e_4": thresholds["1e_4"],
        "first_layer_exceeding_1e_3": thresholds["1e_3"],
        "first_layer_exceeding_1e_2": thresholds["1e_2"],
        "first_layer_exceeding_1e_1": thresholds["1e_1"],
        "final_hidden_metrics": trace["final_hidden_metrics"],
        "final_logit_metrics": final_logit,
        "final_top1_agreement": trace["final_top1_agreement"],
        "final_top5_overlap": trace["final_top5_overlap"],
        "final_top10_overlap": trace["final_top10_overlap"],
        "depth_aware_tolerance": depth_tol,
        "root_cause_classification": root_cause,
        "streaming_passed": streaming_pass,
        "passed": streaming_pass,
        "blockers": [],
    }


def run_exp072_probe(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    dtype: str = "float32",
    target_token_lengths: Sequence[int] = DEFAULT_TARGET_TOKEN_LENGTHS_072,
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES_072,
    max_prompts: int = 2,
    accumulator_mode: str = DEFAULT_ACCUMULATOR_MODE_071,
    local_files_only: bool = False,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    prompt_provider: Callable[[Any, Sequence[int], int], list[tuple[str, str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 072 full-depth divergence trace."""
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

    if load_ok and model is not None and tokenizer is not None:
        model.eval()
        if prompt_provider is not None:
            prompts = prompt_provider(tokenizer, target_token_lengths, max_prompts)
        else:
            prompts = long_context_prompts(
                tokenizer, target_token_lengths, max_prompts=max_prompts,
            )

        for prompt_id, text, target_len, actual_len in prompts:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            for chunk_size in chunk_sizes:
                tf_cell = run_exp072_trace_cell(
                    model=model,
                    input_ids=input_ids,
                    prompt_id=prompt_id,
                    target_token_length=target_len,
                    actual_token_length=actual_len,
                    chunk_size=chunk_size,
                    trace_mode="teacher_forced_layer_inputs",
                    accumulator_mode=accumulator_mode,
                )
                tf_cell["model_id"] = model_id
                cells.append(tf_cell)

                fr_cell = run_exp072_trace_cell(
                    model=model,
                    input_ids=input_ids,
                    prompt_id=prompt_id,
                    target_token_length=target_len,
                    actual_token_length=actual_len,
                    chunk_size=chunk_size,
                    trace_mode="free_running",
                    accumulator_mode=accumulator_mode,
                    teacher_forced_trace=tf_cell.get("per_layer_trace"),
                )
                fr_cell["model_id"] = model_id
                cells.append(fr_cell)

    successful = [c for c in cells if c.get("per_layer_trace") is not None]
    blocked = [c for c in cells if c.get("per_layer_trace") is None]

    def _summarize_mode(mode: str) -> dict[str, float]:
        mode_cells = [c for c in successful if c.get("trace_mode") == mode]
        if not mode_cells:
            return {"max_post_mlp_error": 0.0, "max_attn_context_error": 0.0, "cell_count": 0}
        post_mlp_max = 0.0
        attn_max = 0.0
        for c in mode_cells:
            for layer in c["per_layer_trace"]:
                sm = layer["streaming_vs_materialized"]
                post_mlp_max = max(post_mlp_max, sm["post_mlp_hidden"]["max_abs_error"])
                attn_max = max(attn_max, sm["attn_context"]["max_abs_error"])
        return {
            "max_post_mlp_error": post_mlp_max,
            "max_attn_context_error": attn_max,
            "cell_count": len(mode_cells),
        }

    tf_summary = _summarize_mode("teacher_forced_layer_inputs")
    fr_summary = _summarize_mode("free_running")

    fr_cells = [c for c in successful if c.get("trace_mode") == "free_running"]
    phase16f_reproduced = any(
        c.get("final_logit_metrics", {}).get("max_abs_error", 0) > PHASE16F_LOGIT_TOLERANCE_REF
        for c in fr_cells
    )

    threshold_summary: dict[str, Any] = {}
    for label, key in (
        ("1e_4", "first_layer_exceeding_1e_4"),
        ("1e_3", "first_layer_exceeding_1e_3"),
        ("1e_2", "first_layer_exceeding_1e_2"),
        ("1e_1", "first_layer_exceeding_1e_1"),
    ):
        vals = [c.get(key) for c in fr_cells if c.get(key) is not None]
        threshold_summary[label] = {
            "cells_with_crossing": len(vals),
            "earliest_layer_min": min(vals) if vals else None,
            "earliest_layer_max": max(vals) if vals else None,
        }

    final_logit_errors = [
        c["final_logit_metrics"]["max_abs_error"]
        for c in fr_cells
        if c.get("final_logit_metrics")
    ]
    final_logit_summary = {
        "max_abs_error": max(final_logit_errors) if final_logit_errors else 0.0,
        "mean_abs_error": (
            sum(c["final_logit_metrics"]["mean_abs_error"] for c in fr_cells) / len(fr_cells)
            if fr_cells
            else 0.0
        ),
        "cell_count": len(fr_cells),
    }

    top1_agree = sum(1 for c in fr_cells if c.get("final_top1_agreement"))
    topk_summary = {
        "free_running_top1_agreement_cells": top1_agree,
        "free_running_top5_overlap_mean": (
            sum(c.get("final_top5_overlap", 0) for c in fr_cells) / len(fr_cells)
            if fr_cells
            else 0.0
        ),
        "free_running_top10_overlap_mean": (
            sum(c.get("final_top10_overlap", 0) for c in fr_cells) / len(fr_cells)
            if fr_cells
            else 0.0
        ),
        "cell_count": len(fr_cells),
    }

    root_cause_counts: dict[str, int] = {}
    for c in fr_cells:
        rc = c.get("root_cause_classification", "unknown")
        root_cause_counts[rc] = root_cause_counts.get(rc, 0) + 1

    if not load_ok:
        status = "blocked"
    elif successful and all(c.get("passed") for c in successful):
        status = "pass"
    elif successful:
        status = "diagnostic_complete"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_072_ID,
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
        "phase16f_failure_reproduced": phase16f_reproduced,
        "teacher_forced_local_error_summary": tf_summary,
        "free_running_error_summary": fr_summary,
        "first_threshold_crossing_summary": threshold_summary,
        "final_logit_error_summary": final_logit_summary,
        "final_topk_agreement_summary": topk_summary,
        "root_cause_counts": root_cause_counts,
        "extraction_blockers": blockers,
        "cells": cells,
        "claim_note": EXP072_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": [
            "Offline divergence trace; not token generation integration.",
            "Top-k agreement is supplementary; not exact generation preservation.",
            "Root cause classification is heuristic and diagnostic only.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "prompt_count": len(prompts),
    }


def validate_exp072_report(report: dict[str, Any]) -> list[str]:
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
        "phase16f_failure_reproduced",
        "teacher_forced_local_error_summary",
        "free_running_error_summary",
        "first_threshold_crossing_summary",
        "final_logit_error_summary",
        "final_topk_agreement_summary",
        "root_cause_counts",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "forbidden_claims",
        "cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_072_ID:
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
            "chunk_size",
            "trace_mode",
            "root_cause_classification",
            "passed",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"cell {idx} missing {ck}")

    return errors


# --- Phase 16H: Qwen-family divergence panel ---

EXPERIMENT_073_ID = "exp073_qwen_family_divergence_panel"
DEFAULT_EXP073_REPORT = Path("reports/experiment_073_qwen_family_divergence_panel.json")
DEFAULT_MODEL_IDS_073: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
OPTIONAL_MODEL_IDS_073: tuple[str, ...] = (
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
DEFAULT_CHUNK_SIZES_073: tuple[int, ...] = (16, 64)
MODEL_PANEL_CLASSIFICATIONS: tuple[str, ...] = (
    "free_running_accumulation_confirmed",
    "local_attention_mismatch_detected",
    "parity_failure",
    "unsupported_architecture",
    "model_load_blocked",
    "unknown",
)
LOCAL_ATTENTION_MISMATCH_THRESHOLD = 1e-3
LOCAL_ATTENTION_TINY_THRESHOLD = 1e-4
FREE_RUNNING_TINY_THRESHOLD = 1e-3

EXP073_CLAIM_NOTE = (
    "Offline Qwen-family streaming/materialized divergence panel (Phase 16H). "
    "Reuses Phase 16G layer-by-layer trace across a small model panel. "
    "Not model generation integration, vLLM, CUDA/Triton kernels, or ExactKV "
    "default runtime. Unsupported or blocked models are reported explicitly. "
    "Top-k agreement is supplementary only. Theoretical memory accounting only; "
    "no measured active GPU memory, speed, throughput, latency, or serving claim."
)


def _summarize_trace_mode_cells(
    cells: Sequence[dict[str, Any]],
    *,
    trace_mode: str,
) -> dict[str, float]:
    mode_cells = [c for c in cells if c.get("trace_mode") == trace_mode and c.get("per_layer_trace")]
    if not mode_cells:
        return {"max_post_mlp_error": 0.0, "max_attn_context_error": 0.0, "cell_count": 0}
    post_mlp_max = 0.0
    attn_max = 0.0
    for cell in mode_cells:
        for layer in cell["per_layer_trace"]:
            sm = layer["streaming_vs_materialized"]
            post_mlp_max = max(post_mlp_max, sm["post_mlp_hidden"]["max_abs_error"])
            attn_max = max(attn_max, sm["attn_context"]["max_abs_error"])
    return {
        "max_post_mlp_error": post_mlp_max,
        "max_attn_context_error": attn_max,
        "cell_count": len(mode_cells),
    }


def _summarize_free_running_final_metrics(
    cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fr_cells = [
        c for c in cells
        if c.get("trace_mode") == "free_running" and c.get("final_logit_metrics")
    ]
    if not fr_cells:
        return {
            "max_final_hidden_error": 0.0,
            "max_final_logit_error": 0.0,
            "top1_agreement_cells": 0,
            "top5_overlap_mean": 0.0,
            "top10_overlap_mean": 0.0,
            "cell_count": 0,
        }
    hidden_errors = [
        c["final_hidden_metrics"]["max_abs_error"]
        for c in fr_cells
        if c.get("final_hidden_metrics")
    ]
    logit_errors = [c["final_logit_metrics"]["max_abs_error"] for c in fr_cells]
    top1_agree = sum(1 for c in fr_cells if c.get("final_top1_agreement"))
    return {
        "max_final_hidden_error": max(hidden_errors) if hidden_errors else 0.0,
        "max_final_logit_error": max(logit_errors) if logit_errors else 0.0,
        "top1_agreement_cells": top1_agree,
        "top5_overlap_mean": sum(c.get("final_top5_overlap", 0) for c in fr_cells) / len(fr_cells),
        "top10_overlap_mean": sum(c.get("final_top10_overlap", 0) for c in fr_cells) / len(fr_cells),
        "cell_count": len(fr_cells),
    }


def classify_model_panel_entry(
    *,
    model_load_succeeded: bool,
    architecture_supported: bool,
    parity_passed: bool,
    teacher_forced_max_attn: float,
    teacher_forced_max_post_mlp: float,
    free_running_max_post_mlp: float,
    free_running_root_cause_counts: dict[str, int],
) -> str:
    """Classify one model panel entry from aggregate trace metrics."""
    if not model_load_succeeded:
        return "model_load_blocked"
    if not architecture_supported:
        return "unsupported_architecture"
    if not parity_passed:
        return "parity_failure"
    if teacher_forced_max_attn > LOCAL_ATTENTION_MISMATCH_THRESHOLD:
        return "local_attention_mismatch_detected"
    accumulation_cells = free_running_root_cause_counts.get("free_running_accumulation", 0)
    if (
        accumulation_cells > 0
        and teacher_forced_max_attn <= LOCAL_ATTENTION_TINY_THRESHOLD
        and teacher_forced_max_post_mlp < FREE_RUNNING_TINY_THRESHOLD
        and free_running_max_post_mlp > FREE_RUNNING_TINY_THRESHOLD
    ):
        return "free_running_accumulation_confirmed"
    if teacher_forced_max_attn > LOCAL_ATTENTION_TINY_THRESHOLD:
        return "local_attention_mismatch_detected"
    if (
        free_running_max_post_mlp > FREE_RUNNING_TINY_THRESHOLD
        and teacher_forced_max_post_mlp < FREE_RUNNING_TINY_THRESHOLD
    ):
        return "free_running_accumulation_confirmed"
    return "unknown"


def _compute_trace_cell_memory_accounting(
    model: Any,
    input_ids: torch.Tensor,
    *,
    chunk_size: int,
    actual_token_length: int,
    accumulator_mode: str,
) -> dict[str, Any] | None:
    acc_arg = None if accumulator_mode == "default" else accumulator_mode
    try:
        _, _, mem_records = replay_full_decoder_stack(
            model,
            input_ids,
            attention_path="streaming_compressed",
            chunk_size=chunk_size,
            streaming_accumulator_dtype=acc_arg,
        )
    except Exception:  # noqa: BLE001
        return None
    return aggregate_full_stack_memory(mem_records, context_length=actual_token_length)


def run_model_full_parity_smoke(
    model: Any,
    input_ids: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    """Run one full-stack parity smoke for a model panel entry."""
    with torch.no_grad():
        hf_out = model(input_ids, output_hidden_states=True, use_cache=False)
    hf_logits = hf_out.logits[:, -1, :]
    hf_hidden = hf_out.hidden_states[-1] if hf_out.hidden_states else None
    if hf_hidden is None:
        return {
            "full_model_parity_status": "failed",
            "parity_passed": False,
            "blockers": ["HF forward missing hidden_states"],
        }

    full_hidden, full_logits, _ = replay_full_decoder_stack(
        model, input_ids, attention_path="full", chunk_size=chunk_size,
    )
    norm = _resolve_final_norm(model)
    manual_hidden_parity = norm(full_hidden[:, -1:, :]) if norm is not None else full_hidden[:, -1:, :]
    parity = check_full_model_parity(
        full_logits,
        hf_logits,
        manual_hidden_parity,
        hf_hidden[:, -1:, :],
    )
    return {
        **parity,
        "parity_passed": parity["full_model_parity_status"] == "passed",
    }


def run_exp073_trace_cell(
    *,
    model: Any,
    model_id: str,
    input_ids: torch.Tensor,
    prompt_id: str,
    target_token_length: int,
    actual_token_length: int,
    chunk_size: int,
    trace_mode: str,
    accumulator_mode: str = DEFAULT_ACCUMULATOR_MODE_071,
    teacher_forced_trace: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one panel trace cell with memory accounting."""
    cell = run_exp072_trace_cell(
        model=model,
        input_ids=input_ids,
        prompt_id=prompt_id,
        target_token_length=target_token_length,
        actual_token_length=actual_token_length,
        chunk_size=chunk_size,
        trace_mode=trace_mode,
        accumulator_mode=accumulator_mode,
        teacher_forced_trace=teacher_forced_trace,
    )
    cell["model_id"] = model_id
    if cell.get("per_layer_trace") is not None:
        cell["memory_accounting"] = _compute_trace_cell_memory_accounting(
            model,
            input_ids,
            chunk_size=chunk_size,
            actual_token_length=actual_token_length,
            accumulator_mode=accumulator_mode,
        )
    else:
        cell["memory_accounting"] = None
    return cell


def run_exp073_probe_for_model(
    *,
    model_id: str,
    model: Any,
    tokenizer: Any,
    device: str,
    target_token_lengths: Sequence[int],
    chunk_sizes: Sequence[int],
    max_prompts: int,
    accumulator_mode: str,
    prompt_provider: Callable[[Any, Sequence[int], int], list[tuple[str, str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Run divergence panel trace for one loaded model."""
    from exactkv.attention.hf_single_layer_probe import probe_qwen_architecture_support

    arch_supported, arch_blockers, arch = probe_qwen_architecture_support(model)
    cells: list[dict[str, Any]] = []
    blockers: list[str] = list(arch_blockers)
    parity_passed = False
    parity_status = "blocked"

    if not arch_supported:
        return {
            "model_id": model_id,
            "model_load_succeeded": True,
            "architecture_supported": False,
            "num_layers": arch.get("num_layers"),
            "num_attention_heads": arch.get("num_attention_heads"),
            "num_key_value_heads": arch.get("num_key_value_heads"),
            "hidden_size": arch.get("hidden_size"),
            "classification": "unsupported_architecture",
            "full_parity_passed": False,
            "full_model_parity_status": "blocked",
            "blockers": blockers,
            "cells": cells,
        }

    model.eval()
    if prompt_provider is not None:
        prompts = prompt_provider(tokenizer, target_token_lengths, max_prompts)
    else:
        prompts = long_context_prompts(
            tokenizer, target_token_lengths, max_prompts=max_prompts,
        )

    if prompts:
        first_prompt = prompts[0]
        first_ids = tokenizer(first_prompt[1], return_tensors="pt")["input_ids"].to(device)
        parity_info = run_model_full_parity_smoke(
            model, first_ids, chunk_size=chunk_sizes[0],
        )
        parity_passed = bool(parity_info.get("parity_passed"))
        parity_status = parity_info.get("full_model_parity_status", "failed")
        if not parity_passed:
            blockers.extend(parity_info.get("parity_failures", []) or parity_info.get("blockers", []))

    for prompt_id, text, target_len, actual_len in prompts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        for chunk_size in chunk_sizes:
            tf_cell = run_exp073_trace_cell(
                model=model,
                model_id=model_id,
                input_ids=input_ids,
                prompt_id=prompt_id,
                target_token_length=target_len,
                actual_token_length=actual_len,
                chunk_size=chunk_size,
                trace_mode="teacher_forced_layer_inputs",
                accumulator_mode=accumulator_mode,
            )
            cells.append(tf_cell)

            fr_cell = run_exp073_trace_cell(
                model=model,
                model_id=model_id,
                input_ids=input_ids,
                prompt_id=prompt_id,
                target_token_length=target_len,
                actual_token_length=actual_len,
                chunk_size=chunk_size,
                trace_mode="free_running",
                accumulator_mode=accumulator_mode,
                teacher_forced_trace=tf_cell.get("per_layer_trace"),
            )
            cells.append(fr_cell)

    tf_summary = _summarize_trace_mode_cells(cells, trace_mode="teacher_forced_layer_inputs")
    fr_summary = _summarize_trace_mode_cells(cells, trace_mode="free_running")
    fr_cells = [c for c in cells if c.get("trace_mode") == "free_running" and c.get("per_layer_trace")]
    root_cause_counts: dict[str, int] = {}
    for cell in fr_cells:
        rc = cell.get("root_cause_classification", "unknown")
        root_cause_counts[rc] = root_cause_counts.get(rc, 0) + 1

    classification = classify_model_panel_entry(
        model_load_succeeded=True,
        architecture_supported=True,
        parity_passed=parity_passed,
        teacher_forced_max_attn=tf_summary["max_attn_context_error"],
        teacher_forced_max_post_mlp=tf_summary["max_post_mlp_error"],
        free_running_max_post_mlp=fr_summary["max_post_mlp_error"],
        free_running_root_cause_counts=root_cause_counts,
    )

    return {
        "model_id": model_id,
        "model_load_succeeded": True,
        "architecture_supported": True,
        "num_layers": arch.get("num_layers"),
        "num_attention_heads": arch.get("num_attention_heads"),
        "num_key_value_heads": arch.get("num_key_value_heads"),
        "hidden_size": arch.get("hidden_size"),
        "classification": classification,
        "full_parity_passed": parity_passed,
        "full_model_parity_status": parity_status,
        "blockers": blockers,
        "cells": cells,
    }


def run_exp073_probe(
    *,
    model_ids: Sequence[str] | None = None,
    include_optional_models: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
    target_token_lengths: Sequence[int] = DEFAULT_TARGET_TOKEN_LENGTHS_072,
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES_073,
    max_prompts: int = 2,
    accumulator_mode: str = DEFAULT_ACCUMULATOR_MODE_071,
    local_files_only: bool = False,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    prompt_provider: Callable[[Any, Sequence[int], int], list[tuple[str, str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 073 Qwen-family divergence panel."""
    if model_ids is None:
        panel = list(DEFAULT_MODEL_IDS_073)
        if include_optional_models:
            panel.extend(OPTIONAL_MODEL_IDS_073)
        model_ids = panel

    torch_dtype = getattr(torch, dtype, torch.float32)
    model_entries: list[dict[str, Any]] = []
    loaded_models: list[str] = []
    blocked_models: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []

    for model_id in model_ids:
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
            load_blockers: list[str] = []
        except Exception as exc:  # noqa: BLE001
            load_ok = False
            model = None
            tokenizer = None
            load_blockers = [f"model load failed: {type(exc).__name__}: {exc}"]

        if not load_ok or model is None or tokenizer is None:
            entry = {
                "model_id": model_id,
                "model_load_succeeded": False,
                "architecture_supported": False,
                "num_layers": None,
                "num_attention_heads": None,
                "num_key_value_heads": None,
                "hidden_size": None,
                "classification": "model_load_blocked",
                "full_parity_passed": False,
                "full_model_parity_status": "blocked",
                "blockers": load_blockers,
                "cells": [],
            }
            model_entries.append(entry)
            blocked_models.append({
                "model_id": model_id,
                "classification": "model_load_blocked",
                "blockers": load_blockers,
            })
            continue

        loaded_models.append(model_id)
        entry = run_exp073_probe_for_model(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_token_lengths=target_token_lengths,
            chunk_sizes=chunk_sizes,
            max_prompts=max_prompts,
            accumulator_mode=accumulator_mode,
            prompt_provider=prompt_provider,
        )
        model_entries.append(entry)
        all_cells.extend(entry.get("cells", []))
        if entry["classification"] in ("unsupported_architecture",):
            blocked_models.append({
                "model_id": model_id,
                "classification": entry["classification"],
                "blockers": entry.get("blockers", []),
            })

    successful_cells = [c for c in all_cells if c.get("per_layer_trace") is not None]
    blocked_cells = [c for c in all_cells if c.get("per_layer_trace") is None]

    model_level_classifications = {
        entry["model_id"]: entry["classification"] for entry in model_entries
    }

    teacher_forced_by_model: dict[str, dict[str, float]] = {}
    free_running_by_model: dict[str, dict[str, float]] = {}
    topk_by_model: dict[str, dict[str, Any]] = {}
    root_cause_by_model: dict[str, dict[str, int]] = {}
    memory_by_model: dict[str, dict[str, Any]] = {}

    for entry in model_entries:
        mid = entry["model_id"]
        cells = entry.get("cells", [])
        teacher_forced_by_model[mid] = _summarize_trace_mode_cells(
            cells, trace_mode="teacher_forced_layer_inputs",
        )
        free_running_by_model[mid] = _summarize_trace_mode_cells(
            cells, trace_mode="free_running",
        )
        topk_by_model[mid] = _summarize_free_running_final_metrics(cells)
        rc_counts: dict[str, int] = {}
        for cell in cells:
            if cell.get("trace_mode") != "free_running":
                continue
            rc = cell.get("root_cause_classification", "unknown")
            rc_counts[rc] = rc_counts.get(rc, 0) + 1
        root_cause_by_model[mid] = rc_counts

        reductions = [
            c["memory_accounting"]["best_theoretical_streaming_reduction"]
            for c in cells
            if c.get("memory_accounting")
        ]
        memory_by_model[mid] = {
            "best_theoretical_streaming_reduction_max": max(reductions) if reductions else 0.0,
            "best_theoretical_streaming_reduction_mean": (
                sum(reductions) / len(reductions) if reductions else 0.0
            ),
            "cell_count": len([c for c in cells if c.get("memory_accounting")]),
        }

    panel_summary = {
        "models_requested": len(model_ids),
        "models_loaded": len(loaded_models),
        "models_blocked": len(model_ids) - len(loaded_models) + sum(
            1 for e in model_entries
            if e.get("model_load_succeeded") and e.get("classification") == "unsupported_architecture"
        ),
        "models_parity_pass": sum(1 for e in model_entries if e.get("full_parity_passed")),
        "models_free_running_accumulation": sum(
            1 for e in model_entries
            if e.get("classification") == "free_running_accumulation_confirmed"
        ),
    }

    if not loaded_models:
        status = "blocked"
    elif successful_cells:
        status = "diagnostic_complete"
    else:
        status = "blocked"

    return {
        "experiment_id": EXPERIMENT_073_ID,
        "status": status,
        "model_ids": list(model_ids),
        "loaded_models": loaded_models,
        "blocked_models": blocked_models,
        "device": device,
        "dtype": dtype,
        "target_token_lengths": list(target_token_lengths),
        "chunk_sizes": list(chunk_sizes),
        "accumulator_mode": accumulator_mode,
        "total_cells": len(all_cells),
        "successful_cells": len(successful_cells),
        "blocked_cells": len(blocked_cells),
        "panel_summary": panel_summary,
        "model_level_classifications": model_level_classifications,
        "model_entries": model_entries,
        "teacher_forced_local_error_summary_by_model": teacher_forced_by_model,
        "free_running_error_summary_by_model": free_running_by_model,
        "final_topk_agreement_summary_by_model": topk_by_model,
        "root_cause_counts_by_model": root_cause_by_model,
        "memory_accounting_summary_by_model": memory_by_model,
        "limitations": [
            "Offline Qwen-family divergence panel; not token generation integration.",
            "Top-k agreement is supplementary; not exact generation preservation.",
            "Blocked or unsupported models are reported explicitly.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP073_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
    }


def validate_exp073_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "model_ids",
        "loaded_models",
        "blocked_models",
        "device",
        "dtype",
        "target_token_lengths",
        "chunk_sizes",
        "total_cells",
        "successful_cells",
        "blocked_cells",
        "model_level_classifications",
        "teacher_forced_local_error_summary_by_model",
        "free_running_error_summary_by_model",
        "final_topk_agreement_summary_by_model",
        "root_cause_counts_by_model",
        "memory_accounting_summary_by_model",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "forbidden_claims",
        "model_entries",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_073_ID:
        errors.append("experiment_id mismatch")

    entries = report.get("model_entries")
    if not isinstance(entries, list):
        errors.append("model_entries must be a list")
        return errors

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"model entry {idx} not dict")
            continue
        for key in (
            "model_id",
            "model_load_succeeded",
            "architecture_supported",
            "classification",
            "blockers",
            "cells",
        ):
            if key not in entry:
                errors.append(f"model entry {idx} missing {key}")

        for cidx, cell in enumerate(entry.get("cells", [])):
            if not isinstance(cell, dict):
                errors.append(f"model entry {idx} cell {cidx} not dict")
                continue
            for ck in (
                "model_id",
                "prompt_id",
                "target_token_length",
                "chunk_size",
                "trace_mode",
                "root_cause_classification",
                "passed",
                "blockers",
            ):
                if ck not in cell:
                    errors.append(f"model entry {idx} cell {cidx} missing {ck}")

    return errors
