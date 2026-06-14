"""Tests for Experiment 047 full-KV restore panel (no model download by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from exactkv.cache.hf_kv_restore import (
    DEFAULT_MODEL,
    EXPERIMENT_047_ID,
    FORBIDDEN_CLAIMS,
    PANEL_CLAIM_NOTE,
    default_panel_prompts,
    reconcile_panel_cell_counts,
    resolve_panel_device_dtype_configs,
    validate_exp047_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_047_FULL_KV_RESTORE_PANEL.md"
_REPORT = (
    Path(__file__).resolve().parents[1] / "reports" / "experiment_047_full_kv_restore_panel.json"
)


def _synthetic_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prompt_id": "panel_001",
        "prompt": "test",
        "category": "short_natural",
        "backend_name": "in_memory_kv_storage",
        "device": "cpu",
        "dtype": "torch.float32",
        "cache_format": "dynamic_v5",
        "prompt_length": 5,
        "continuation_token_count": 8,
        "layer_count": 24,
        "shape_summary": "[(1, 2, 5, 64)]",
        "dtype_summary": "torch.float32",
        "payload_byte_summary": 1024,
        "token_exact_match": True,
        "live_token_ids": [1, 2, 3],
        "restored_token_ids": [1, 2, 3],
        "live_decoded": "abc",
        "restored_decoded": "abc",
        "first_divergence_idx": None,
        "restore_blocker": "",
        "cell_status": "passed",
    }
    base.update(overrides)
    return base


def _synthetic_report(**overrides: object) -> dict[str, object]:
    configs = resolve_panel_device_dtype_configs()
    for cfg in configs:
        if cfg.status == "pending":
            cfg.status = "tested" if cfg.device == "cpu" else "skipped"
            if cfg.status == "skipped" and not cfg.skip_reason:
                cfg.skip_reason = "CUDA unavailable"
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_047_ID,
        "model": DEFAULT_MODEL,
        "transformers_version": "5.0.0",
        "total_cells": 2,
        "passed_cells": 2,
        "failed_cells": 0,
        "skipped_cells": 0,
        "storage_backends_tested": ["in_memory_kv_storage", "file_kv_storage"],
        "device_dtype_configs_tested": [c.to_dict() for c in configs],
        "cache_formats_detected": ["dynamic_v5"],
        "aggregate_exactness": {
            "token_exact_match_count": 2,
            "failures_count": 0,
        },
        "per_cell": [
            _synthetic_cell(),
            _synthetic_cell(
                prompt_id="panel_002",
                backend_name="file_kv_storage",
            ),
        ],
        "restore_blockers": [],
        "claim_note": PANEL_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_panel_prompt_count_in_range() -> None:
    prompts = default_panel_prompts()
    assert 8 <= len(prompts) <= 16
    categories = {p["category"] for p in prompts}
    for cat in (
        "short_natural",
        "structured_json",
        "retrieval_copy",
        "code_like",
        "long_context_summary",
        "tool_call_style",
    ):
        assert cat in categories


def test_validate_exp047_report_schema() -> None:
    report = _synthetic_report()
    assert validate_exp047_report(report) == []


def test_aggregate_counts_reconcile() -> None:
    report = _synthetic_report(
        total_cells=3,
        passed_cells=2,
        failed_cells=1,
        skipped_cells=0,
        per_cell=[
            _synthetic_cell(),
            _synthetic_cell(prompt_id="panel_002"),
            _synthetic_cell(
                prompt_id="panel_003",
                token_exact_match=False,
                cell_status="failed",
                first_divergence_idx=0,
            ),
        ],
        aggregate_exactness={"token_exact_match_count": 2, "failures_count": 1},
    )
    assert reconcile_panel_cell_counts(report) == []
    report["passed_cells"] = 1
    assert reconcile_panel_cell_counts(report) != []


def test_skipped_cuda_configs_when_unavailable() -> None:
    configs = resolve_panel_device_dtype_configs()
    if torch.cuda.is_available():
        cuda_cfgs = [c for c in configs if c.device == "cuda"]
        assert cuda_cfgs
        assert all(c.status == "pending" for c in cuda_cfgs)
    else:
        skipped = [c for c in configs if c.status == "skipped"]
        assert len(skipped) >= 2
        assert all("CUDA unavailable" in c.skip_reason for c in skipped)


def test_per_cell_includes_backend_device_dtype_cache_format() -> None:
    report = _synthetic_report()
    cell = report["per_cell"][0]
    assert isinstance(cell, dict)
    for field in ("backend_name", "device", "dtype", "cache_format"):
        assert field in cell


def test_first_divergence_schema() -> None:
    report = _synthetic_report(
        per_cell=[
            _synthetic_cell(first_divergence_idx=3, cell_status="failed"),
        ],
        total_cells=1,
        passed_cells=0,
        failed_cells=1,
        aggregate_exactness={"token_exact_match_count": 0, "failures_count": 1},
    )
    assert validate_exp047_report(report) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "full-kv restore panel",
        "not a serving runtime",
        "vllm",
        "lmcache",
        "remote prefix",
        "generation and verification behavior is unchanged",
        "throughput",
        "vericache",
        "active memory savings",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
        assert phrase not in text


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model restore panel",
)
def test_model_restore_panel_cpu_cell() -> None:
    from exactkv.cache.hf_kv_restore import run_restore_panel_cell
    from exactkv.cache.storage import InMemoryKVStorageBackend
    from exactkv.runtime.model_runtime import ModelRuntime

    runtime = ModelRuntime(DEFAULT_MODEL, device="cpu", dtype="float32")
    backend = InMemoryKVStorageBackend()
    entry = default_panel_prompts()[0]
    result = run_restore_panel_cell(
        runtime,
        prompt_id=entry["prompt_id"],
        prompt=entry["prompt"],
        category=entry["category"],
        backend=backend,
        max_new_tokens=8,
    )
    assert not result.restore_blocker, result.restore_blocker
    assert result.token_exact_match
    assert result.cell_status == "passed"
    assert result.device
    assert result.dtype
    assert result.cache_format != "unknown"


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp047_full_kv_restore_panel.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp047_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_047_ID
    assert report["failed_cells"] == 0
