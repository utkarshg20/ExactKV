"""Reference-only streaming quantized-KV attention feasibility probe (Phase 16A).

Compares full-precision attention, materialized dequantized attention, and
chunked streaming dequantized attention without holding a full decompressed
K/V tensor at once. **Not** a production compressor or inference integration.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

EXPERIMENT_066_ID = "exp066_streaming_quant_attention_feasibility"
DEFAULT_EXP066_REPORT = Path("reports/experiment_066_streaming_quant_attention_feasibility.json")

EXP066_CLAIM_NOTE = (
    "Tensor-level streaming quantized-KV attention feasibility probe (Phase 16A). "
    "Reference int8 KV only — not model inference, vLLM, CUDA/Triton kernels, or "
    "ExactKV default generation. Theoretical memory accounting only; no measured "
    "active GPU memory, runtime speed, throughput, latency, or serving claim."
)

FORBIDDEN_ATTENTION_CLAIMS = (
    "throughput",
    "latency",
    "speedup",
    "tokens_per_second",
    "production_memory_savings",
    "active_gpu_memory_savings",
    "runtime memory savings",
    "serving support",
    "vericache throughput reproduced",
)

DEFAULT_STREAMING_TOLERANCE_FP32 = 5e-4
DEFAULT_STREAMING_TOLERANCE_FP16 = 2e-2

AccumulatorMode = str  # "default" | "float32" | "float64"


@dataclass
class StreamingAttentionDiagnostics:
    """Optional research diagnostics for chunked online-softmax attention."""

    num_chunks: int
    chunk_size: int
    accumulator_dtype: str
    running_max_min: float
    running_max_max: float
    running_den_min: float
    running_den_max: float
    has_nan: bool
    has_inf: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_accumulator_dtype(
    accumulator_dtype: AccumulatorMode | torch.dtype | None,
    query_dtype: torch.dtype,
) -> torch.dtype:
    """Map research accumulator mode to a torch dtype."""
    if accumulator_dtype is None or accumulator_dtype == "default":
        return query_dtype
    if accumulator_dtype == "float32":
        return torch.float32
    if accumulator_dtype == "float64":
        return torch.float64
    if isinstance(accumulator_dtype, torch.dtype):
        return accumulator_dtype
    raise ValueError(f"unsupported accumulator_dtype: {accumulator_dtype!r}")


def strict_streaming_tolerance(dtype: torch.dtype) -> float:
    """Fixed Phase 16D strict tolerance."""
    return _scale_from_dtype(dtype)


def dtype_aware_streaming_tolerance(dtype: torch.dtype) -> float:
    """Dtype-scaled tolerance (fp32 strict, fp16 relaxed)."""
    return _scale_from_dtype(dtype)


def layer_depth_aware_streaming_tolerance(
    dtype: torch.dtype,
    prefix_layer_count: int,
) -> float:
    """Scale strict tolerance by sqrt(prefix depth) for accumulated drift."""
    base = _scale_from_dtype(dtype)
    depth = max(1, prefix_layer_count)
    return base * math.sqrt(depth)


def reference_high_precision_tolerance(fp64_max_abs_error: float) -> float:
    """Tolerance anchored to float64 accumulator reference error."""
    ref = max(0.0, fp64_max_abs_error)
    return max(ref * 10.0, ref + 1e-6, DEFAULT_STREAMING_TOLERANCE_FP32)


def compute_candidate_tolerances(
    *,
    dtype: torch.dtype,
    prefix_layer_count: int,
    fp64_max_abs_error: float | None,
) -> dict[str, float]:
    """Return named candidate tolerance policies for audit reporting."""
    policies = {
        "strict_16d": strict_streaming_tolerance(dtype),
        "dtype_aware": dtype_aware_streaming_tolerance(dtype),
        "layer_depth_aware": layer_depth_aware_streaming_tolerance(dtype, prefix_layer_count),
    }
    if fp64_max_abs_error is not None:
        policies["reference_high_precision"] = reference_high_precision_tolerance(
            fp64_max_abs_error
        )
    return policies


@dataclass
class QuantizedKV:
    """Reference symmetric int8 KV container (not a production compressor)."""

    k_q: torch.Tensor
    v_q: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    axis_or_grouping: str
    group_size: int | None
    shape: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_or_grouping": self.axis_or_grouping,
            "group_size": self.group_size,
            "shape": list(self.shape),
            "k_q_dtype": str(self.k_q.dtype),
            "v_q_dtype": str(self.v_q.dtype),
            "k_scale_shape": list(self.k_scale.shape),
            "v_scale_shape": list(self.v_scale.shape),
        }


@dataclass
class MemoryAccounting:
    """Conservative theoretical tensor memory accounting (not measured GPU VRAM)."""

    full_kv_bytes: int
    stored_quantized_kv_bytes: int
    materialized_working_kv_bytes: int
    streaming_peak_chunk_working_kv_bytes: int
    metadata_bytes: int
    chunk_size: int
    num_chunks: int
    theoretical_streaming_working_reduction_vs_materialized: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttentionFeasibilityResult:
    """Per-cell attention feasibility comparison."""

    full_output: torch.Tensor
    materialized_compressed_output: torch.Tensor
    streaming_compressed_output: torch.Tensor
    max_abs_streaming_vs_materialized: float
    max_abs_full_vs_materialized: float
    max_abs_full_vs_streaming: float
    memory_accounting: MemoryAccounting
    passed: bool
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_streaming_vs_materialized": self.max_abs_streaming_vs_materialized,
            "max_abs_full_vs_materialized": self.max_abs_full_vs_materialized,
            "max_abs_full_vs_streaming": self.max_abs_full_vs_streaming,
            "memory_accounting": self.memory_accounting.to_dict(),
            "passed": self.passed,
            "tolerance": self.tolerance,
            "output_shape": list(self.full_output.shape),
        }


def _scale_from_dtype(dtype: torch.dtype) -> float:
    return float(DEFAULT_STREAMING_TOLERANCE_FP16 if dtype == torch.float16 else DEFAULT_STREAMING_TOLERANCE_FP32)


def _quantize_tensor_symmetric_int8(
    x: torch.Tensor,
    group_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 quantization along the head dimension."""
    if x.ndim != 4:
        raise ValueError(f"expected [B,H,T,D], got shape {tuple(x.shape)}")

    b, h, t, d = x.shape
    gs = d if group_size is None else group_size
    if d % gs != 0:
        raise ValueError(f"D={d} must be divisible by group_size={gs}")

    if gs == d:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        return q, scale

    n_groups = d // gs
    grouped = x.view(b, h, t, n_groups, gs)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(grouped / scale).clamp(-128, 127).to(torch.int8).view(b, h, t, d)
    scale = scale.squeeze(-1)
    return q, scale


def _dequantize_tensor(q: torch.Tensor, scale: torch.Tensor, group_size: int | None) -> torch.Tensor:
    d = q.shape[-1]
    gs = d if group_size is None else group_size
    if gs == d:
        return q.to(scale.dtype) * scale
    b, h, t, _ = q.shape
    n_groups = d // gs
    grouped = q.view(b, h, t, n_groups, gs).to(scale.dtype)
    scale_exp = scale.unsqueeze(-1)
    return (grouped * scale_exp).view(b, h, t, d)


def quantize_kv_int8_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    group_size: int | None = None,
) -> QuantizedKV:
    """Reference symmetric int8 KV quantizer (not a production compressor)."""
    if k.shape != v.shape or k.ndim != 4:
        raise ValueError(f"k and v must share shape [B,H,T,D], got {tuple(k.shape)} {tuple(v.shape)}")

    k_q, k_scale = _quantize_tensor_symmetric_int8(k, group_size)
    v_q, v_scale = _quantize_tensor_symmetric_int8(v, group_size)
    grouping = "per_token_symmetric_int8"
    if group_size is not None and group_size < k.shape[-1]:
        grouping = f"per_token_grouped_symmetric_int8_g{group_size}"

    return QuantizedKV(
        k_q=k_q,
        v_q=v_q,
        k_scale=k_scale,
        v_scale=v_scale,
        axis_or_grouping=grouping,
        group_size=group_size,
        shape=tuple(k.shape),
    )


def dequantize_kv_materialized(qkv: QuantizedKV) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize full dequantized K and V tensors."""
    k = _dequantize_tensor(qkv.k_q, qkv.k_scale, qkv.group_size)
    v = _dequantize_tensor(qkv.v_q, qkv.v_scale, qkv.group_size)
    return k, v


def _causal_mask_scores(
    scores: torch.Tensor,
    *,
    query_len: int,
    key_start: int,
    total_len: int,
) -> torch.Tensor:
    """Mask keys strictly after each query's absolute position.

    Assumes queries occupy the last ``query_len`` positions in a sequence of
    length ``total_len``.
    """
    if query_len > total_len:
        raise ValueError("query_len cannot exceed total_len for causal attention")
    device = scores.device
    query_positions = torch.arange(
        total_len - query_len,
        total_len,
        device=device,
    )
    key_positions = torch.arange(key_start, key_start + scores.shape[-1], device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    masked = scores.masked_fill(~allowed.view(1, 1, query_len, -1), float("-inf"))
    return masked


def attention_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention over full-precision K/V."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 [B,H,Q/T,D] tensors")
    d = q.shape[-1]
    scale = d**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        total_len = k.shape[-2]
        query_len = q.shape[-2]
        scores = _causal_mask_scores(
            scores,
            query_len=query_len,
            key_start=0,
            total_len=total_len,
        )
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, v)


def attention_materialized_compressed(
    q: torch.Tensor,
    qkv: QuantizedKV,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Dequantize full K/V, then run standard attention."""
    k, v = dequantize_kv_materialized(qkv)
    return attention_full(q, k, v, causal=causal)


def attention_streaming_compressed(
    q: torch.Tensor,
    qkv: QuantizedKV,
    chunk_size: int,
    *,
    causal: bool = False,
    accumulator_dtype: AccumulatorMode | torch.dtype | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Chunked dequantized attention without full K/V materialization."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if q.ndim != 4:
        raise ValueError(f"q must be [B,H,Q,D], got {tuple(q.shape)}")

    b, h, query_len, d = q.shape
    total_len = qkv.k_q.shape[2]
    if qkv.shape != (b, h, total_len, d):
        raise ValueError("q batch/heads/dim must match quantized KV")

    acc_dtype = resolve_accumulator_dtype(accumulator_dtype, q.dtype)
    scale = d**-0.5
    running_max = torch.full(
        (b, h, query_len, 1), float("-inf"), device=q.device, dtype=acc_dtype
    )
    running_den = torch.zeros((b, h, query_len, 1), device=q.device, dtype=acc_dtype)
    running_num = torch.zeros((b, h, query_len, d), device=q.device, dtype=acc_dtype)

    num_chunks = math.ceil(total_len / chunk_size)
    for key_start in range(0, total_len, chunk_size):
        key_end = min(key_start + chunk_size, total_len)
        k_chunk = _dequantize_tensor(
            qkv.k_q[:, :, key_start:key_end, :],
            qkv.k_scale[:, :, key_start:key_end, ...],
            qkv.group_size,
        )
        v_chunk = _dequantize_tensor(
            qkv.v_q[:, :, key_start:key_end, :],
            qkv.v_scale[:, :, key_start:key_end, ...],
            qkv.group_size,
        )

        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        if causal:
            scores = _causal_mask_scores(
                scores,
                query_len=query_len,
                key_start=key_start,
                total_len=total_len,
            )

        scores_acc = scores.to(acc_dtype)
        chunk_max = scores_acc.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        exp_old = torch.exp(running_max - new_max)
        exp_scores = torch.exp(scores_acc - new_max)
        v_acc = v_chunk.to(acc_dtype)

        running_den = running_den * exp_old + exp_scores.sum(dim=-1, keepdim=True)
        running_num = running_num * exp_old + torch.matmul(exp_scores, v_acc)
        running_max = new_max

    output = running_num / running_den.clamp(min=1e-12)
    output = output.to(q.dtype)
    output = torch.nan_to_num(output, nan=0.0)

    if not return_diagnostics:
        return output

    diag = StreamingAttentionDiagnostics(
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        accumulator_dtype=str(acc_dtype).replace("torch.", ""),
        running_max_min=float(running_max.min().item()),
        running_max_max=float(running_max.max().item()),
        running_den_min=float(running_den.min().item()),
        running_den_max=float(running_den.max().item()),
        has_nan=bool(torch.isnan(output).any().item()),
        has_inf=bool(torch.isinf(output).any().item()),
    )
    return output, diag.to_dict()


def estimate_attention_memory_bytes(
    *,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    element_size_fp: int,
    chunk_size: int,
    group_size: int | None = None,
) -> MemoryAccounting:
    """Theoretical tensor byte accounting; not measured active GPU memory."""
    if any(v <= 0 for v in (batch, heads, seq_len, head_dim, element_size_fp, chunk_size)):
        raise ValueError("memory accounting inputs must be positive")

    per_tensor = batch * heads * seq_len * head_dim
    full_kv_bytes = per_tensor * 2 * element_size_fp
    stored_quantized_kv_bytes = per_tensor * 2  # int8 k_q + v_q

    n_scale_elems = batch * heads * seq_len
    if group_size is not None and group_size < head_dim:
        n_scale_elems *= head_dim // group_size
    metadata_bytes = n_scale_elems * 4 * 2  # float32 k/v scales
    stored_quantized_kv_bytes += metadata_bytes

    materialized_working_kv_bytes = per_tensor * 2 * element_size_fp
    effective_chunk = min(chunk_size, seq_len)
    streaming_peak_chunk_working_kv_bytes = (
        batch * heads * effective_chunk * head_dim * 2 * element_size_fp
    )
    num_chunks = math.ceil(seq_len / chunk_size)
    if materialized_working_kv_bytes > 0:
        reduction = 1.0 - (
            streaming_peak_chunk_working_kv_bytes / materialized_working_kv_bytes
        )
    else:
        reduction = 0.0

    return MemoryAccounting(
        full_kv_bytes=full_kv_bytes,
        stored_quantized_kv_bytes=stored_quantized_kv_bytes,
        materialized_working_kv_bytes=materialized_working_kv_bytes,
        streaming_peak_chunk_working_kv_bytes=streaming_peak_chunk_working_kv_bytes,
        metadata_bytes=metadata_bytes,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        theoretical_streaming_working_reduction_vs_materialized=reduction,
    )


def run_attention_feasibility_cell(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    causal: bool = False,
    group_size: int | None = None,
    tolerance: float | None = None,
) -> AttentionFeasibilityResult:
    """Run one feasibility comparison cell."""
    qkv = quantize_kv_int8_reference(k, v, group_size=group_size)
    tol = tolerance if tolerance is not None else _scale_from_dtype(q.dtype)

    full_out = attention_full(q, k, v, causal=causal)
    mat_out = attention_materialized_compressed(q, qkv, causal=causal)
    stream_out = attention_streaming_compressed(q, qkv, chunk_size, causal=causal)

    max_sm = float((stream_out - mat_out).abs().max().item())
    max_fm = float((full_out - mat_out).abs().max().item())
    max_fs = float((full_out - stream_out).abs().max().item())

    elem_size = 4 if q.dtype == torch.float32 else 2
    mem = estimate_attention_memory_bytes(
        batch=q.shape[0],
        heads=q.shape[1],
        seq_len=k.shape[2],
        head_dim=k.shape[3],
        element_size_fp=elem_size,
        chunk_size=chunk_size,
        group_size=group_size,
    )

    return AttentionFeasibilityResult(
        full_output=full_out,
        materialized_compressed_output=mat_out,
        streaming_compressed_output=stream_out,
        max_abs_streaming_vs_materialized=max_sm,
        max_abs_full_vs_materialized=max_fm,
        max_abs_full_vs_streaming=max_fs,
        memory_accounting=mem,
        passed=max_sm <= tol,
        tolerance=tol,
    )


def validate_exp066_report(report: dict[str, Any]) -> list[str]:
    """Validate Experiment 066 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "total_cells",
        "pass_cells",
        "failed_cells",
        "max_streaming_vs_materialized_error",
        "max_full_vs_streaming_error",
        "best_theoretical_streaming_working_reduction",
        "worst_theoretical_streaming_working_reduction",
        "cells",
        "claim_note",
        "forbidden_claims",
        "limitations",
        "no_performance_claims_note",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_066_ID:
        errors.append("experiment_id mismatch")

    text_blob = str(report).lower()
    for term in FORBIDDEN_ATTENTION_CLAIMS:
        if term in text_blob and term not in (report.get("forbidden_claims") or []):
            # allow listing forbidden terms explicitly
            pass

    for term in ("throughput_improved", "latency_improved", "speedup_claim"):
        if term in text_blob:
            errors.append(f"forbidden positive claim phrase: {term}")

    cells = report.get("cells")
    if not isinstance(cells, list):
        errors.append("cells must be a list")
        return errors

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {idx} not a dict")
            continue
        for ck in (
            "dtype",
            "B",
            "H",
            "Q",
            "T",
            "D",
            "chunk_size",
            "passed",
            "max_abs_streaming_vs_materialized",
            "memory_accounting",
        ):
            if ck not in cell:
                errors.append(f"cell {idx} missing {ck}")
        mem = cell.get("memory_accounting")
        if isinstance(mem, dict):
            for mk in (
                "full_kv_bytes",
                "stored_quantized_kv_bytes",
                "materialized_working_kv_bytes",
                "streaming_peak_chunk_working_kv_bytes",
                "metadata_bytes",
                "chunk_size",
                "num_chunks",
                "theoretical_streaming_working_reduction_vs_materialized",
            ):
                if mk not in mem:
                    errors.append(f"cell {idx} memory_accounting missing {mk}")
                else:
                    val = mem[mk]
                    if mk != "theoretical_streaming_working_reduction_vs_materialized" and (
                        not isinstance(val, int) or val < 0
                    ):
                        errors.append(f"cell {idx} memory_accounting.{mk} must be non-negative int")

    return errors
