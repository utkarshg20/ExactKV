"""Offline attention tolerance policy for streaming-vs-materialized diagnostics (Phase 16I).

Formalizes strict numeric gates, depth-aware diagnostic tolerance, teacher-forced
local alignment, free-running accumulation interpretation, and supplementary
top-k agreement. **Not** wired into ExactKV default generation.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from exactkv.attention.streaming_quant_attention import (
    DEFAULT_STREAMING_TOLERANCE_FP32,
    FORBIDDEN_ATTENTION_CLAIMS,
    layer_depth_aware_streaming_tolerance,
    strict_streaming_tolerance,
)

EXPERIMENT_074_ID = "exp074_attention_tolerance_policy_panel"
DEFAULT_EXP074_REPORT = Path("reports/experiment_074_attention_tolerance_policy_panel.json")

DEFAULT_REPORT_PATHS: dict[str, Path] = {
    "exp070": Path("reports/experiment_070_streaming_multilayer_numerics_audit.json"),
    "exp071": Path("reports/experiment_071_full_prefix_logit_drift_smoke.json"),
    "exp072": Path("reports/experiment_072_full_depth_divergence_trace.json"),
    "exp073": Path("reports/experiment_073_qwen_family_divergence_panel.json"),
}

LOCAL_ALIGNMENT_ATTN_THRESHOLD = 1e-4
LOCAL_MISMATCH_ATTN_THRESHOLD = 1e-3
FREE_RUNNING_TINY_THRESHOLD = 1e-3

EXP074_CLAIM_NOTE = (
    "Offline attention-diagnostics tolerance policy panel (Phase 16I). "
    "Formalizes strict and depth-aware interpretation for Phases 16E–16H reports. "
    "Not model generation integration, vLLM, CUDA/Triton kernels, or ExactKV "
    "default runtime. Depth-aware tolerance is diagnostic only. Top-k agreement "
    "is supplementary and not exactness. No measured active GPU memory, speed, "
    "throughput, latency, or serving claim."
)


class MetricType(str, Enum):
    HIDDEN = "hidden"
    LOGITS = "logits"
    ATTENTION_CONTEXT = "attention_context"


class OfflineAttentionStatus(str, Enum):
    STRICT_NUMERIC_PASS = "strict_numeric_pass"
    STRICT_FAIL_DEPTH_AWARE_PASS = "strict_fail_depth_aware_pass"
    LOCAL_ALIGNMENT_PASS_FREE_RUNNING_ACCUMULATION = (
        "local_alignment_pass_free_running_accumulation"
    )
    TOPK_AGREES_NUMERIC_DRIFT_PRESENT = "topk_agrees_numeric_drift_present"
    LOCAL_ATTENTION_MISMATCH = "local_attention_mismatch"
    PARITY_FAILURE = "parity_failure"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass
class AttentionTolerancePolicy:
    """Offline attention tolerance policy configuration."""

    strict_base_tolerance: float = DEFAULT_STREAMING_TOLERANCE_FP32
    depth_aware_use_sqrt: bool = True
    dtype_name: str = "float32"

    def resolve_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_name, torch.float32)

    def strict_tolerance(self) -> float:
        return strict_streaming_tolerance(self.resolve_dtype())

    def depth_aware_tolerance(self, prefix_layers: int) -> float:
        if self.depth_aware_use_sqrt:
            return layer_depth_aware_streaming_tolerance(
                self.resolve_dtype(), prefix_layers,
            )
        return self.strict_tolerance()

    def tolerance_for_metric(
        self,
        metric_type: MetricType,
        prefix_layers: int,
    ) -> tuple[float, float]:
        """Return (strict_tolerance, depth_aware_diagnostic_tolerance)."""
        strict = self.strict_tolerance()
        if metric_type == MetricType.ATTENTION_CONTEXT:
            return strict, strict
        depth = self.depth_aware_tolerance(prefix_layers)
        return strict, depth

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TopKAgreementSummary:
    top1_agreement: bool | None
    top5_overlap: int | None = None
    top10_overlap: int | None = None

    @property
    def supplementary_pass(self) -> bool:
        """Top-k agreement is supplementary only — never implies exactness."""
        return bool(self.top1_agreement)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttentionToleranceDecision:
    strict_numeric_pass: bool
    depth_aware_numeric_pass: bool
    teacher_forced_local_alignment_pass: bool
    topk_supplementary_pass: bool
    overall_offline_status: OfflineAttentionStatus
    interpretation_note: str
    strict_tolerance: float
    depth_aware_tolerance: float
    max_abs_error: float
    metric_type: MetricType = MetricType.HIDDEN
    root_cause_classification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall_offline_status"] = self.overall_offline_status.value
        d["metric_type"] = self.metric_type.value
        return d


def evaluate_offline_attention_cell(
    *,
    policy: AttentionTolerancePolicy,
    metric_type: MetricType,
    prefix_layers: int,
    max_abs_error: float,
    blockers: Sequence[str] | None = None,
    parity_passed: bool | None = None,
    teacher_forced_max_attn_error: float | None = None,
    teacher_forced_max_post_mlp_error: float | None = None,
    free_running_max_post_mlp_error: float | None = None,
    root_cause_classification: str | None = None,
    topk: TopKAgreementSummary | None = None,
) -> AttentionToleranceDecision:
    """Apply offline tolerance policy to one diagnostic cell."""
    if blockers:
        return AttentionToleranceDecision(
            strict_numeric_pass=False,
            depth_aware_numeric_pass=False,
            teacher_forced_local_alignment_pass=False,
            topk_supplementary_pass=False,
            overall_offline_status=OfflineAttentionStatus.BLOCKED,
            interpretation_note="Cell blocked; strict failure remains visible.",
            strict_tolerance=policy.strict_tolerance(),
            depth_aware_tolerance=policy.depth_aware_tolerance(prefix_layers),
            max_abs_error=max_abs_error,
            metric_type=metric_type,
            root_cause_classification=root_cause_classification,
        )

    if parity_passed is False:
        return AttentionToleranceDecision(
            strict_numeric_pass=False,
            depth_aware_numeric_pass=False,
            teacher_forced_local_alignment_pass=False,
            topk_supplementary_pass=bool(topk and topk.supplementary_pass),
            overall_offline_status=OfflineAttentionStatus.PARITY_FAILURE,
            interpretation_note=(
                "Full manual replay parity failed; do not interpret streaming drift."
            ),
            strict_tolerance=policy.strict_tolerance(),
            depth_aware_tolerance=policy.depth_aware_tolerance(prefix_layers),
            max_abs_error=max_abs_error,
            metric_type=metric_type,
            root_cause_classification=root_cause_classification,
        )

    strict_tol, depth_tol = policy.tolerance_for_metric(metric_type, prefix_layers)
    strict_pass = max_abs_error <= strict_tol
    depth_pass = max_abs_error <= depth_tol

    tf_attn = teacher_forced_max_attn_error if teacher_forced_max_attn_error is not None else 0.0
    tf_post = teacher_forced_max_post_mlp_error if teacher_forced_max_post_mlp_error is not None else 0.0
    fr_post = free_running_max_post_mlp_error if free_running_max_post_mlp_error is not None else 0.0

    tf_local_pass = tf_attn <= LOCAL_ALIGNMENT_ATTN_THRESHOLD and tf_post < FREE_RUNNING_TINY_THRESHOLD
    topk_pass = bool(topk and topk.supplementary_pass)

    if tf_attn > LOCAL_MISMATCH_ATTN_THRESHOLD or root_cause_classification == "local_attention_mismatch":
        status = OfflineAttentionStatus.LOCAL_ATTENTION_MISMATCH
        note = (
            "Teacher-forced local attention/context error exceeds mismatch threshold; "
            "investigate per-layer streaming vs materialized attention."
        )
    elif strict_pass:
        status = OfflineAttentionStatus.STRICT_NUMERIC_PASS
        note = "Streaming-vs-materialized error within strict offline tolerance."
    elif (
        tf_local_pass
        and fr_post > FREE_RUNNING_TINY_THRESHOLD
        and (
            root_cause_classification == "free_running_accumulation"
            or (tf_post < FREE_RUNNING_TINY_THRESHOLD and fr_post > tf_post * 5)
        )
    ):
        status = OfflineAttentionStatus.LOCAL_ALIGNMENT_PASS_FREE_RUNNING_ACCUMULATION
        note = (
            "Teacher-forced local alignment holds; free-running divergence accumulates "
            "through depth. Diagnostic only — not exact generation preservation."
        )
    elif not strict_pass and depth_pass:
        status = OfflineAttentionStatus.STRICT_FAIL_DEPTH_AWARE_PASS
        note = (
            "Strict numeric tolerance failed but depth-aware diagnostic tolerance passed. "
            "Depth-aware gate is diagnostic only — not a production correctness guarantee."
        )
    elif topk_pass and not strict_pass:
        status = OfflineAttentionStatus.TOPK_AGREES_NUMERIC_DRIFT_PRESENT
        note = (
            "Top-k agreement holds but numeric drift exceeds strict tolerance. "
            "Top-k is supplementary only and does not imply exactness."
        )
    else:
        status = OfflineAttentionStatus.UNKNOWN
        note = "Cell does not match a known offline interpretation category."

    return AttentionToleranceDecision(
        strict_numeric_pass=strict_pass,
        depth_aware_numeric_pass=depth_pass,
        teacher_forced_local_alignment_pass=tf_local_pass,
        topk_supplementary_pass=topk_pass,
        overall_offline_status=status,
        interpretation_note=note,
        strict_tolerance=strict_tol,
        depth_aware_tolerance=depth_tol,
        max_abs_error=max_abs_error,
        metric_type=metric_type,
        root_cause_classification=root_cause_classification,
    )


def _max_attn_from_trace(per_layer_trace: Sequence[dict[str, Any]] | None) -> float:
    if not per_layer_trace:
        return 0.0
    vals = [
        layer["streaming_vs_materialized"]["attn_context"]["max_abs_error"]
        for layer in per_layer_trace
        if "attn_context" in layer.get("streaming_vs_materialized", {})
    ]
    return max(vals) if vals else 0.0


def _max_post_mlp_from_trace(per_layer_trace: Sequence[dict[str, Any]] | None) -> float:
    if not per_layer_trace:
        return 0.0
    vals = [
        layer["streaming_vs_materialized"]["post_mlp_hidden"]["max_abs_error"]
        for layer in per_layer_trace
        if "post_mlp_hidden" in layer.get("streaming_vs_materialized", {})
    ]
    return max(vals) if vals else 0.0


def _cell_record(
    *,
    decision: AttentionToleranceDecision,
    source_experiment: str,
    model_id: str,
    prompt_id: str,
    target_token_length: int,
    chunk_size: int,
    prefix_layers: int,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "source_experiment": source_experiment,
        "model_id": model_id,
        "prompt_id": prompt_id,
        "target_token_length": target_token_length,
        "chunk_size": chunk_size,
        "prefix_layers": prefix_layers,
        "metric_type": decision.metric_type.value,
        "max_abs_error": decision.max_abs_error,
        "strict_tolerance": decision.strict_tolerance,
        "depth_aware_tolerance": decision.depth_aware_tolerance,
        "strict_numeric_pass": decision.strict_numeric_pass,
        "depth_aware_numeric_pass": decision.depth_aware_numeric_pass,
        "teacher_forced_local_alignment_pass": decision.teacher_forced_local_alignment_pass,
        "topk_supplementary_pass": decision.topk_supplementary_pass,
        "overall_offline_status": decision.overall_offline_status.value,
        "root_cause_classification": decision.root_cause_classification,
        "interpretation_note": decision.interpretation_note,
        "blockers": blockers,
    }


def extract_policy_cells_from_exp070(
    report: dict[str, Any],
    *,
    policy: AttentionTolerancePolicy,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model_id = report.get("model_id", "unknown")
    for cell in report.get("cells", []):
        metrics = cell.get("streaming_vs_materialized_hidden_metrics")
        if not metrics:
            out.append(_cell_record(
                decision=evaluate_offline_attention_cell(
                    policy=policy,
                    metric_type=MetricType.HIDDEN,
                    prefix_layers=int(cell.get("prefix_layer_count", 1)),
                    max_abs_error=0.0,
                    blockers=cell.get("blockers") or ["missing metrics"],
                ),
                source_experiment="exp070",
                model_id=model_id,
                prompt_id=str(cell.get("prompt_id", "")),
                target_token_length=int(cell.get("target_token_length", 0)),
                chunk_size=int(cell.get("chunk_size", 0)),
                prefix_layers=int(cell.get("prefix_layer_count", 1)),
                blockers=list(cell.get("blockers") or ["missing metrics"]),
            ))
            continue
        prefix = int(cell.get("prefix_layer_count", 1))
        decision = evaluate_offline_attention_cell(
            policy=policy,
            metric_type=MetricType.HIDDEN,
            prefix_layers=prefix,
            max_abs_error=float(metrics["max_abs_error"]),
            blockers=cell.get("blockers"),
            root_cause_classification=None,
        )
        out.append(_cell_record(
            decision=decision,
            source_experiment="exp070",
            model_id=model_id,
            prompt_id=str(cell.get("prompt_id", "")),
            target_token_length=int(cell.get("target_token_length", 0)),
            chunk_size=int(cell.get("chunk_size", 0)),
            prefix_layers=prefix,
            blockers=list(cell.get("blockers") or []),
        ))
    return out


def extract_policy_cells_from_exp071(
    report: dict[str, Any],
    *,
    policy: AttentionTolerancePolicy,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model_id = report.get("model_id", "unknown")
    for cell in report.get("cells", []):
        blockers = list(cell.get("blockers") or [])
        parity_ok = cell.get("full_model_parity_status") == "passed"
        prefix = int(cell.get("num_layers_replayed", 24))

        for metric_type, key in (
            (MetricType.HIDDEN, "streaming_vs_materialized_hidden_metrics"),
            (MetricType.LOGITS, "streaming_vs_materialized_logit_metrics"),
        ):
            metrics = cell.get(key)
            if not metrics:
                out.append(_cell_record(
                    decision=evaluate_offline_attention_cell(
                        policy=policy,
                        metric_type=metric_type,
                        prefix_layers=prefix,
                        max_abs_error=0.0,
                        blockers=blockers or ["missing metrics"],
                        parity_passed=parity_ok,
                    ),
                    source_experiment="exp071",
                    model_id=model_id,
                    prompt_id=str(cell.get("prompt_id", "")),
                    target_token_length=int(cell.get("target_token_length", 0)),
                    chunk_size=int(cell.get("chunk_size", 0)),
                    prefix_layers=prefix,
                    blockers=blockers or ["missing metrics"],
                ))
                continue
            topk = TopKAgreementSummary(
                top1_agreement=metrics.get("top1_agreement"),
                top5_overlap=metrics.get("top5_overlap"),
                top10_overlap=metrics.get("top10_overlap"),
            ) if metric_type == MetricType.LOGITS else None
            decision = evaluate_offline_attention_cell(
                policy=policy,
                metric_type=metric_type,
                prefix_layers=prefix,
                max_abs_error=float(metrics["max_abs_error"]),
                blockers=blockers,
                parity_passed=parity_ok,
                topk=topk,
            )
            out.append(_cell_record(
                decision=decision,
                source_experiment="exp071",
                model_id=model_id,
                prompt_id=str(cell.get("prompt_id", "")),
                target_token_length=int(cell.get("target_token_length", 0)),
                chunk_size=int(cell.get("chunk_size", 0)),
                prefix_layers=prefix,
                blockers=blockers,
            ))
    return out


def extract_policy_cells_from_exp072(
    report: dict[str, Any],
    *,
    policy: AttentionTolerancePolicy,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model_id = report.get("model_id", "unknown")
    cells = report.get("cells", [])
    tf_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        if cell.get("trace_mode") == "teacher_forced_layer_inputs":
            tf_by_key[(str(cell.get("prompt_id")), int(cell.get("chunk_size", 0)))] = cell

    for cell in cells:
        if cell.get("trace_mode") != "free_running":
            continue
        blockers = list(cell.get("blockers") or [])
        prefix = len(cell.get("per_layer_trace") or []) or int(report.get("num_layers") or 24)
        tf_cell = tf_by_key.get((str(cell.get("prompt_id")), int(cell.get("chunk_size", 0))))
        tf_trace = tf_cell.get("per_layer_trace") if tf_cell else None
        tf_attn = _max_attn_from_trace(tf_trace)
        tf_post = _max_post_mlp_from_trace(tf_trace)
        fr_post = _max_post_mlp_from_trace(cell.get("per_layer_trace"))

        logit_metrics = cell.get("final_logit_metrics") or {}
        hidden_metrics = cell.get("final_hidden_metrics") or {}
        topk = TopKAgreementSummary(
            top1_agreement=cell.get("final_top1_agreement"),
            top5_overlap=cell.get("final_top5_overlap"),
            top10_overlap=cell.get("final_top10_overlap"),
        )

        for metric_type, metrics in (
            (MetricType.HIDDEN, hidden_metrics),
            (MetricType.LOGITS, logit_metrics),
        ):
            if not metrics:
                out.append(_cell_record(
                    decision=evaluate_offline_attention_cell(
                        policy=policy,
                        metric_type=metric_type,
                        prefix_layers=prefix,
                        max_abs_error=0.0,
                        blockers=blockers or ["missing metrics"],
                    ),
                    source_experiment="exp072",
                    model_id=model_id,
                    prompt_id=str(cell.get("prompt_id", "")),
                    target_token_length=int(cell.get("target_token_length", 0)),
                    chunk_size=int(cell.get("chunk_size", 0)),
                    prefix_layers=prefix,
                    blockers=blockers or ["missing metrics"],
                ))
                continue
            decision = evaluate_offline_attention_cell(
                policy=policy,
                metric_type=metric_type,
                prefix_layers=prefix,
                max_abs_error=float(metrics["max_abs_error"]),
                blockers=blockers,
                teacher_forced_max_attn_error=tf_attn,
                teacher_forced_max_post_mlp_error=tf_post,
                free_running_max_post_mlp_error=fr_post,
                root_cause_classification=cell.get("root_cause_classification"),
                topk=topk if metric_type == MetricType.LOGITS else None,
            )
            out.append(_cell_record(
                decision=decision,
                source_experiment="exp072",
                model_id=model_id,
                prompt_id=str(cell.get("prompt_id", "")),
                target_token_length=int(cell.get("target_token_length", 0)),
                chunk_size=int(cell.get("chunk_size", 0)),
                prefix_layers=prefix,
                blockers=blockers,
            ))
    return out


def extract_policy_cells_from_exp073(
    report: dict[str, Any],
    *,
    policy: AttentionTolerancePolicy,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in report.get("model_entries", []):
        model_id = str(entry.get("model_id", "unknown"))
        if not entry.get("model_load_succeeded"):
            out.append(_cell_record(
                decision=evaluate_offline_attention_cell(
                    policy=policy,
                    metric_type=MetricType.HIDDEN,
                    prefix_layers=0,
                    max_abs_error=0.0,
                    blockers=entry.get("blockers") or ["model load blocked"],
                ),
                source_experiment="exp073",
                model_id=model_id,
                prompt_id="",
                target_token_length=0,
                chunk_size=0,
                prefix_layers=0,
                blockers=list(entry.get("blockers") or ["model load blocked"]),
            ))
            continue
        parity_ok = entry.get("full_parity_passed")
        prefix = int(entry.get("num_layers") or 24)
        cells = entry.get("cells", [])
        tf_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for cell in cells:
            if cell.get("trace_mode") == "teacher_forced_layer_inputs":
                tf_by_key[(str(cell.get("prompt_id")), int(cell.get("chunk_size", 0)))] = cell

        for cell in cells:
            if cell.get("trace_mode") != "free_running":
                continue
            blockers = list(cell.get("blockers") or [])
            tf_cell = tf_by_key.get((str(cell.get("prompt_id")), int(cell.get("chunk_size", 0))))
            tf_trace = tf_cell.get("per_layer_trace") if tf_cell else None
            tf_attn = _max_attn_from_trace(tf_trace)
            tf_post = _max_post_mlp_from_trace(tf_trace)
            fr_post = _max_post_mlp_from_trace(cell.get("per_layer_trace"))
            logit_metrics = cell.get("final_logit_metrics") or {}
            hidden_metrics = cell.get("final_hidden_metrics") or {}
            topk = TopKAgreementSummary(
                top1_agreement=cell.get("final_top1_agreement"),
                top5_overlap=cell.get("final_top5_overlap"),
                top10_overlap=cell.get("final_top10_overlap"),
            )
            for metric_type, metrics in (
                (MetricType.HIDDEN, hidden_metrics),
                (MetricType.LOGITS, logit_metrics),
            ):
                if not metrics:
                    out.append(_cell_record(
                        decision=evaluate_offline_attention_cell(
                            policy=policy,
                            metric_type=metric_type,
                            prefix_layers=prefix,
                            max_abs_error=0.0,
                            blockers=blockers or ["missing metrics"],
                            parity_passed=parity_ok,
                        ),
                        source_experiment="exp073",
                        model_id=model_id,
                        prompt_id=str(cell.get("prompt_id", "")),
                        target_token_length=int(cell.get("target_token_length", 0)),
                        chunk_size=int(cell.get("chunk_size", 0)),
                        prefix_layers=prefix,
                        blockers=blockers or ["missing metrics"],
                    ))
                    continue
                decision = evaluate_offline_attention_cell(
                    policy=policy,
                    metric_type=metric_type,
                    prefix_layers=prefix,
                    max_abs_error=float(metrics["max_abs_error"]),
                    blockers=blockers,
                    parity_passed=parity_ok,
                    teacher_forced_max_attn_error=tf_attn,
                    teacher_forced_max_post_mlp_error=tf_post,
                    free_running_max_post_mlp_error=fr_post,
                    root_cause_classification=cell.get("root_cause_classification"),
                    topk=topk if metric_type == MetricType.LOGITS else None,
                )
                out.append(_cell_record(
                    decision=decision,
                    source_experiment="exp073",
                    model_id=model_id,
                    prompt_id=str(cell.get("prompt_id", "")),
                    target_token_length=int(cell.get("target_token_length", 0)),
                    chunk_size=int(cell.get("chunk_size", 0)),
                    prefix_layers=prefix,
                    blockers=blockers,
                ))
    return out


REPORT_EXTRACTORS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "exp070": extract_policy_cells_from_exp070,
    "exp071": extract_policy_cells_from_exp071,
    "exp072": extract_policy_cells_from_exp072,
    "exp073": extract_policy_cells_from_exp073,
}


def load_report_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_policy_panel_cells(*, policy: AttentionTolerancePolicy) -> list[dict[str, Any]]:
    """Small mock panel when prior experiment reports are missing."""
    scenarios = [
        {
            "source": "synthetic",
            "model_id": "mock/strict_pass",
            "prompt_id": "p_strict",
            "target_token_length": 32,
            "chunk_size": 16,
            "prefix_layers": 4,
            "metric_type": MetricType.HIDDEN,
            "max_abs_error": 1e-5,
        },
        {
            "source": "synthetic",
            "model_id": "mock/depth_aware",
            "prompt_id": "p_depth",
            "target_token_length": 32,
            "chunk_size": 16,
            "prefix_layers": 24,
            "metric_type": MetricType.LOGITS,
            "max_abs_error": 0.002,
            "topk": TopKAgreementSummary(top1_agreement=True, top5_overlap=5, top10_overlap=10),
            "tf_attn": 1e-6,
            "tf_post": 1e-6,
            "fr_post": 0.5,
            "root_cause": "free_running_accumulation",
        },
        {
            "source": "synthetic",
            "model_id": "mock/topk_drift",
            "prompt_id": "p_topk",
            "target_token_length": 64,
            "chunk_size": 32,
            "prefix_layers": 24,
            "metric_type": MetricType.LOGITS,
            "max_abs_error": 0.1,
            "topk": TopKAgreementSummary(top1_agreement=True, top5_overlap=5, top10_overlap=10),
        },
        {
            "source": "synthetic",
            "model_id": "mock/local_mismatch",
            "prompt_id": "p_local",
            "target_token_length": 32,
            "chunk_size": 16,
            "prefix_layers": 8,
            "metric_type": MetricType.ATTENTION_CONTEXT,
            "max_abs_error": 0.02,
            "tf_attn": 0.02,
            "root_cause": "local_attention_mismatch",
        },
        {
            "source": "synthetic",
            "model_id": "mock/blocked",
            "prompt_id": "p_blocked",
            "target_token_length": 0,
            "chunk_size": 0,
            "prefix_layers": 0,
            "metric_type": MetricType.HIDDEN,
            "max_abs_error": 0.0,
            "blockers": ["synthetic blocked cell"],
        },
    ]
    out: list[dict[str, Any]] = []
    for sc in scenarios:
        decision = evaluate_offline_attention_cell(
            policy=policy,
            metric_type=sc["metric_type"],
            prefix_layers=int(sc["prefix_layers"]),
            max_abs_error=float(sc["max_abs_error"]),
            blockers=sc.get("blockers"),
            teacher_forced_max_attn_error=sc.get("tf_attn"),
            teacher_forced_max_post_mlp_error=sc.get("tf_post"),
            free_running_max_post_mlp_error=sc.get("fr_post"),
            root_cause_classification=sc.get("root_cause"),
            topk=sc.get("topk"),
        )
        out.append(_cell_record(
            decision=decision,
            source_experiment=str(sc["source"]),
            model_id=str(sc["model_id"]),
            prompt_id=str(sc["prompt_id"]),
            target_token_length=int(sc["target_token_length"]),
            chunk_size=int(sc["chunk_size"]),
            prefix_layers=int(sc["prefix_layers"]),
            blockers=list(sc.get("blockers") or []),
        ))
    return out


def _count_by_status(cells: Sequence[dict[str, Any]], status: OfflineAttentionStatus) -> int:
    return sum(1 for c in cells if c.get("overall_offline_status") == status.value)


def _interpretation_summary(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(cells)
    strict_pass = sum(1 for c in cells if c.get("strict_numeric_pass"))
    depth_pass = sum(1 for c in cells if c.get("depth_aware_numeric_pass"))
    return {
        "total_evaluated": total,
        "strict_numeric_pass_rate": strict_pass / total if total else 0.0,
        "depth_aware_pass_rate": depth_pass / total if total else 0.0,
        "dominant_statuses": _dominant_status_counts(cells),
        "policy_guidance": (
            "Use strict tolerance for single-layer and attention-context gates. "
            "Use depth-aware tolerance only as a diagnostic interpretive aid for "
            "accumulated hidden/logit drift. Top-k agreement never upgrades a cell "
            "to exactness or production correctness."
        ),
    }


def _dominant_status_counts(cells: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        st = str(cell.get("overall_offline_status", "unknown"))
        counts[st] = counts.get(st, 0) + 1
    return counts


def run_exp074_panel(
    *,
    report_paths: dict[str, Path] | None = None,
    policy: AttentionTolerancePolicy | None = None,
    include_optional_models: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
    local_files_only: bool = False,
    optional_probe_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Experiment 074 tolerance policy panel."""
    policy = policy or AttentionTolerancePolicy(dtype_name=dtype)
    paths = report_paths or DEFAULT_REPORT_PATHS
    reports_loaded: list[str] = []
    reports_missing: list[str] = []
    evaluated: list[dict[str, Any]] = []

    for exp_id, path in paths.items():
        report = load_report_json(path)
        if report is None:
            reports_missing.append(exp_id)
            continue
        reports_loaded.append(exp_id)
        extractor = REPORT_EXTRACTORS[exp_id]
        evaluated.extend(extractor(report, policy=policy))

    if not evaluated:
        evaluated = synthetic_policy_panel_cells(policy=policy)
        reports_missing = list(paths.keys())

    optional_requested = include_optional_models
    optional_loaded: list[str] = []
    optional_blocked: list[dict[str, Any]] = []

    if include_optional_models:
        if optional_probe_runner is not None:
            opt_report = optional_probe_runner(
                include_optional_models=True,
                device=device,
                dtype=dtype,
                local_files_only=local_files_only,
            )
        else:
            from exactkv.attention.hf_full_replay_probe import (
                OPTIONAL_MODEL_IDS_073,
                run_exp073_probe,
            )

            opt_report = run_exp073_probe(
                model_ids=list(OPTIONAL_MODEL_IDS_073),
                target_token_lengths=[32],
                chunk_sizes=[16, 64],
                max_prompts=1,
                accumulator_mode="float32",
                device=device,
                dtype=dtype,
                local_files_only=local_files_only,
            )
        optional_loaded = list(opt_report.get("loaded_models", []))
        for blocked in opt_report.get("blocked_models", []):
            optional_blocked.append(blocked)
        evaluated.extend(extract_policy_cells_from_exp073(opt_report, policy=policy))

    status = "diagnostic_complete" if evaluated else "blocked"

    return {
        "experiment_id": EXPERIMENT_074_ID,
        "status": status,
        "reports_loaded": reports_loaded,
        "reports_missing": reports_missing,
        "policy": policy.to_dict(),
        "policy_formula": {
            "strict_base_tolerance": policy.strict_base_tolerance,
            "depth_aware_formula": "5e-4 * sqrt(prefix_layers)",
            "attention_context_uses_strict_only": True,
            "hidden_logits_allow_depth_aware_diagnostic": True,
            "topk_never_implies_exactness": True,
            "not_production_correctness_guarantee": True,
        },
        "total_cells_evaluated": len(evaluated),
        "strict_numeric_pass_cells": _count_by_status(
            evaluated, OfflineAttentionStatus.STRICT_NUMERIC_PASS,
        ),
        "strict_fail_depth_aware_pass_cells": _count_by_status(
            evaluated, OfflineAttentionStatus.STRICT_FAIL_DEPTH_AWARE_PASS,
        ),
        "local_alignment_pass_free_running_accumulation_cells": _count_by_status(
            evaluated, OfflineAttentionStatus.LOCAL_ALIGNMENT_PASS_FREE_RUNNING_ACCUMULATION,
        ),
        "topk_agrees_numeric_drift_present_cells": _count_by_status(
            evaluated, OfflineAttentionStatus.TOPK_AGREES_NUMERIC_DRIFT_PRESENT,
        ),
        "local_attention_mismatch_cells": _count_by_status(
            evaluated, OfflineAttentionStatus.LOCAL_ATTENTION_MISMATCH,
        ),
        "blocked_cells": _count_by_status(evaluated, OfflineAttentionStatus.BLOCKED),
        "parity_failure_cells": _count_by_status(
            evaluated, OfflineAttentionStatus.PARITY_FAILURE,
        ),
        "unknown_cells": _count_by_status(evaluated, OfflineAttentionStatus.UNKNOWN),
        "optional_models_requested": optional_requested,
        "optional_models_loaded": optional_loaded,
        "optional_models_blocked": optional_blocked,
        "interpretation_summary": _interpretation_summary(evaluated),
        "evaluated_cells": evaluated,
        "limitations": [
            "Offline attention-diagnostics policy only; not generation integration.",
            "Depth-aware tolerance is diagnostic only; not production correctness.",
            "Top-k agreement is supplementary; not exactness.",
            "Strict numeric failures remain visible in per-cell records.",
            "No CUDA/Triton/vLLM/serving integration.",
        ],
        "no_performance_claims_note": (
            "No speed, throughput, latency, serving, measured active GPU memory, "
            "or production-memory claim is made."
        ),
        "claim_note": EXP074_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
    }


def validate_exp074_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "reports_loaded",
        "reports_missing",
        "policy",
        "total_cells_evaluated",
        "strict_numeric_pass_cells",
        "strict_fail_depth_aware_pass_cells",
        "local_alignment_pass_free_running_accumulation_cells",
        "topk_agrees_numeric_drift_present_cells",
        "local_attention_mismatch_cells",
        "blocked_cells",
        "optional_models_requested",
        "optional_models_loaded",
        "optional_models_blocked",
        "interpretation_summary",
        "limitations",
        "no_performance_claims_note",
        "claim_note",
        "forbidden_claims",
        "evaluated_cells",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing key: {key}")

    if report.get("experiment_id") != EXPERIMENT_074_ID:
        errors.append("experiment_id mismatch")

    cells = report.get("evaluated_cells")
    if not isinstance(cells, list):
        errors.append("evaluated_cells must be a list")
        return errors

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {idx} not dict")
            continue
        for ck in (
            "source_experiment",
            "model_id",
            "metric_type",
            "strict_numeric_pass",
            "depth_aware_numeric_pass",
            "overall_offline_status",
            "interpretation_note",
            "blockers",
        ):
            if ck not in cell:
                errors.append(f"cell {idx} missing {ck}")

    return errors
