"""Tests for Experiment 067 HF single-layer attention drift docs/reports."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.attention.hf_single_layer_probe import (
    EXPERIMENT_067_ID,
    EXP067_CLAIM_NOTE,
    validate_exp067_report,
)
from exactkv.attention.streaming_quant_attention import FORBIDDEN_ATTENTION_CLAIMS

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_067_HF_SINGLE_LAYER_ATTENTION_DRIFT.md"


def _sample_cell(**overrides: object) -> dict[str, object]:
    mem = {
        "full_kv_bytes": 8192,
        "stored_quantized_kv_bytes": 4608,
        "materialized_working_kv_bytes": 8192,
        "streaming_peak_chunk_working_kv_bytes": 2048,
        "metadata_bytes": 512,
        "chunk_size": 16,
        "num_chunks": 4,
        "theoretical_streaming_working_reduction_vs_materialized": 0.75,
    }
    metrics = {
        "max_abs_error": 1e-6,
        "mean_abs_error": 1e-7,
        "cosine_similarity": 1.0,
        "relative_l2_error": 1e-6,
        "top_dim_max_abs": 1e-6,
    }
    base: dict[str, object] = {
        "model_id": "mock",
        "prompt_id": "prompt_0",
        "prompt_preview": "hello",
        "layer_idx": 0,
        "extraction_status": "success",
        "extraction_mode": "projection_only",
        "rope_status": "skipped",
        "grouped_query_status": "repeated",
        "q_shape": [1, 4, 8, 16],
        "k_shape": [1, 4, 8, 16],
        "v_shape": [1, 4, 8, 16],
        "chunk_size": 16,
        "streaming_vs_materialized": dict(metrics),
        "full_vs_materialized": dict(metrics),
        "full_vs_streaming": {**metrics, "max_abs_error": 0.05},
        "memory_accounting": mem,
        "passed": True,
        "tolerance": 5e-4,
        "blockers": [],
    }
    base.update(overrides)
    return base


def _blocked_cell() -> dict[str, object]:
    return {
        "model_id": "mock",
        "prompt_id": "prompt_0",
        "prompt_preview": "hello",
        "layer_idx": 0,
        "extraction_status": "blocked",
        "extraction_mode": "blocked",
        "rope_status": "unsupported",
        "grouped_query_status": "unsupported",
        "chunk_size": 16,
        "passed": False,
        "blockers": ["model load failed"],
    }


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_067_ID,
        "status": "pass",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "model_load_succeeded": True,
        "total_cells": 1,
        "successful_cells": 1,
        "blocked_cells": 0,
        "streaming_vs_materialized_pass_cells": 1,
        "max_streaming_vs_materialized_error": 1e-6,
        "full_vs_streaming_drift_summary": {
            "max_abs_error": 0.05,
            "mean_abs_error": 0.01,
            "cell_count": 1,
        },
        "output_projection_drift_summary": {
            "cells_with_output_projection": 1,
            "max_full_vs_streaming_after_o_proj": 0.04,
            "mean_full_vs_streaming_after_o_proj": 0.01,
        },
        "memory_accounting_summary": {
            "best_theoretical_streaming_working_reduction": 0.9,
            "worst_theoretical_streaming_working_reduction": 0.5,
        },
        "extraction_blockers": [],
        "cells": [_sample_cell()],
        "claim_note": EXP067_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "limitations": ["offline only"],
        "no_performance_claims_note": "no throughput claim",
        "prompt_count": 1,
        "chunk_sizes": [16],
    }
    base.update(overrides)
    return base


def test_exp067_success_report_validates() -> None:
    assert validate_exp067_report(_report()) == []


def test_exp067_blocked_report_validates() -> None:
    assert validate_exp067_report(
        _report(
            status="blocked",
            successful_cells=0,
            blocked_cells=1,
            streaming_vs_materialized_pass_cells=0,
            cells=[_blocked_cell()],
            model_load_succeeded=False,
            extraction_blockers=["model load failed"],
        )
    ) == []


def test_forbidden_claim_terms_listed() -> None:
    for term in (
        "throughput",
        "latency",
        "speedup",
        "tokens_per_second",
        "production_memory_savings",
        "active_gpu_memory_savings",
    ):
        assert term in FORBIDDEN_ATTENTION_CLAIMS


def test_report_has_no_forbidden_performance_fields() -> None:
    blob = json.dumps(_report()).lower()
    for term in ("throughput_improved", "latency_improved", "speedup_claim"):
        assert term not in blob


def test_doc_required_wording() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "offline single-layer attention-drift probe",
        "not model generation integration",
        "not wired into exactkv generation",
        "projection-only",
        "no cuda kernel",
        "no triton kernel",
        "no vllm integration",
        "vericache",
        "phase 16a",
        "drift metrics",
    ):
        assert phrase in text, phrase
