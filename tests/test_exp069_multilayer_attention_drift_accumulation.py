"""Tests for Experiment 069 multi-layer drift accumulation docs/reports."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.attention.hf_multilayer_probe import (
    EXPERIMENT_069_ID,
    EXP069_CLAIM_NOTE,
    validate_exp069_report,
)
from exactkv.attention.streaming_quant_attention import FORBIDDEN_ATTENTION_CLAIMS

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_069_MULTILAYER_ATTENTION_DRIFT_ACCUMULATION.md"


def _metrics(**overrides: float) -> dict[str, float]:
    base = {
        "max_abs_error": 1e-6,
        "mean_abs_error": 1e-7,
        "cosine_similarity": 1.0,
        "relative_l2_error": 1e-6,
        "top_dim_max_abs": 1e-6,
    }
    base.update(overrides)
    return base


def _agg_mem() -> dict[str, object]:
    return {
        "aggregate_full_kv_bytes": 8192,
        "aggregate_stored_quantized_kv_bytes": 4608,
        "aggregate_materialized_working_kv_bytes": 8192,
        "aggregate_streaming_peak_working_kv_bytes_conservative": 2048,
        "best_theoretical_streaming_reduction": 0.75,
        "layer_count": 2,
    }


def _sample_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model_id": "mock",
        "prompt_id": "long_64",
        "prompt_preview": "hello",
        "target_token_length": 64,
        "actual_token_length": 64,
        "prefix_layer_count": 2,
        "chunk_size": 16,
        "full_block_parity_status": "passed",
        "full_block_parity_metrics": _metrics(),
        "streaming_vs_materialized_hidden_metrics": _metrics(),
        "full_vs_streaming_hidden_metrics": {**_metrics(), "max_abs_error": 0.05},
        "full_vs_materialized_hidden_metrics": {**_metrics(), "max_abs_error": 0.05},
        "per_layer_memory_accounting": {
            "streaming_path": [{"num_chunks": 4, "full_kv_bytes": 4096}],
        },
        "aggregate_memory_accounting": {
            "streaming_path": _agg_mem(),
            "materialized_path": _agg_mem(),
            "full_path": _agg_mem(),
        },
        "streaming_passed": True,
        "passed": True,
        "blockers": [],
    }
    base.update(overrides)
    return base


def _blocked_cell() -> dict[str, object]:
    return {
        "model_id": "mock",
        "prompt_id": "long_64",
        "prompt_preview": "hello",
        "target_token_length": 64,
        "actual_token_length": 64,
        "prefix_layer_count": 2,
        "chunk_size": 16,
        "full_block_parity_status": "blocked",
        "streaming_vs_materialized_hidden_metrics": None,
        "passed": False,
        "blockers": ["model load failed"],
    }


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_069_ID,
        "status": "pass",
        "model_id": "mock",
        "device": "cpu",
        "dtype": "float32",
        "model_load_succeeded": True,
        "target_token_lengths": [64, 128],
        "prefix_layer_counts": [1, 2, 4],
        "chunk_sizes": [16, 32, 64],
        "total_cells": 1,
        "successful_cells": 1,
        "blocked_cells": 0,
        "full_block_parity_pass_cells": 1,
        "streaming_vs_materialized_pass_cells": 1,
        "max_streaming_vs_materialized_error": 1e-6,
        "full_vs_streaming_drift_summary": {
            "max_abs_error": 0.05,
            "mean_abs_error": 0.01,
            "cell_count": 1,
        },
        "full_vs_materialized_drift_summary": {
            "max_abs_error": 0.05,
            "mean_abs_error": 0.01,
            "cell_count": 1,
        },
        "memory_accounting_summary": {
            "best_theoretical_streaming_reduction": 0.875,
            "worst_theoretical_streaming_reduction": 0.5,
            "cells_with_reduction_gt_zero": 1,
        },
        "longest_context_tested": 128,
        "max_prefix_layers_tested": 4,
        "max_num_chunks": 8,
        "limitations": ["offline only"],
        "no_performance_claims_note": "no throughput claim",
        "claim_note": EXP069_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_ATTENTION_CLAIMS),
        "cells": [_sample_cell()],
    }
    base.update(overrides)
    return base


def test_exp069_success_report_validates() -> None:
    assert validate_exp069_report(_report()) == []


def test_exp069_parity_fail_report_validates() -> None:
    assert validate_exp069_report(
        _report(
            cells=[
                _sample_cell(
                    full_block_parity_status="failed",
                    passed=False,
                    blockers=["full_block_parity failed"],
                )
            ],
            full_block_parity_pass_cells=0,
            streaming_vs_materialized_pass_cells=1,
            status="failed",
        )
    ) == []


def test_exp069_blocked_report_validates() -> None:
    assert validate_exp069_report(
        _report(
            status="blocked",
            successful_cells=0,
            blocked_cells=1,
            full_block_parity_pass_cells=0,
            streaming_vs_materialized_pass_cells=0,
            cells=[_blocked_cell()],
            model_load_succeeded=False,
        )
    ) == []


def test_memory_accounting_fields_present() -> None:
    cell = _sample_cell()
    agg = cell["aggregate_memory_accounting"]
    assert isinstance(agg, dict)
    stream = agg["streaming_path"]
    assert stream["aggregate_full_kv_bytes"] >= 0
    assert stream["best_theoretical_streaming_reduction"] >= 0


def test_no_forbidden_performance_fields() -> None:
    body = {k: v for k, v in _report().items() if k not in ("forbidden_claims", "claim_note")}
    blob = json.dumps(body).lower()
    for term in ("throughput_improved", "latency_improved", "speedup_claim", "active_gpu_memory_savings"):
        assert term not in blob


def test_doc_required_wording() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "offline multi-layer drift accumulation probe",
        "not model generation integration",
        "not wired into exactkv generation",
        "full-block parity",
        "no cuda",
        "no triton",
        "no vllm integration",
        "vericache",
        "phase 16c",
        "theoretical memory accounting",
    ):
        assert phrase in text, phrase
