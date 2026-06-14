"""Tests for Experiment 051 offline verifier CUDA drift panel (no CUDA by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from exactkv.cache.offline_verifier import (
    DEFAULT_MODEL,
    EXPERIMENT_051_ID,
    FORBIDDEN_CLAIMS,
    OFFLINE_CUDA_DRIFT_CLAIM_NOTE,
    VERIFIER_SOURCE,
    default_cuda_drift_prompts,
    resolve_cuda_drift_dtype_configs,
    validate_exp051_report,
)

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_051_OFFLINE_VERIFIER_CUDA_DRIFT_PANEL.md"
)
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_051_offline_verifier_cuda_drift_panel.json"
)


def _cuda_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prompt_id": "drift_001",
        "prompt": "test",
        "category": "pharmacy_semantic",
        "backend_name": "in_memory_kv_storage",
        "compressor_name": "int4_sim",
        "draft_len": 8,
        "device": "cuda",
        "dtype": "float16",
        "cache_format": "dynamic_v5",
        "draft_source": "int4_sim",
        "verifier_source": VERIFIER_SOURCE,
        "live_reference_token_ids": [1, 2, 3, 4],
        "offline_output_token_ids": [1, 2, 3, 4],
        "token_exact_match": True,
        "exactkv_failures": 0,
        "accepted_prefix_lengths": [2, 2],
        "mean_acceptance": 0.75,
        "draft_divergence_count": 1,
        "semantic_divergence_count": 1,
        "first_divergence_idx": None,
        "restore_blocker": "",
        "draft_blocker": "",
        "verification_blocker": "",
        "exactness_blocker": "",
    }
    base.update(overrides)
    return base


def _synthetic_report(**overrides: object) -> dict[str, object]:
    configs = resolve_cuda_drift_dtype_configs()
    dtype_supported = {c.dtype: c.dtype_supported for c in configs}
    skipped = [c.to_dict() for c in configs if c.status == "skipped"]
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_051_ID,
        "model": DEFAULT_MODEL,
        "device": "cuda",
        "dtype": "float16,bfloat16",
        "prompt_count": 6,
        "storage_backends": ["in_memory_kv_storage", "file_kv_storage"],
        "compressor_names": ["int4_sim", "k8_v4_sim", "k8_v4_boundary4_v8_sim", "int8"],
        "draft_len_values": [4, 8],
        "max_new_tokens": 32,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [
            _cuda_cell(),
            _cuda_cell(
                prompt_id="drift_002",
                dtype="bfloat16",
                draft_divergence_count=0,
                semantic_divergence_count=0,
            ),
        ],
        "exactkv_failures": 0,
        "token_exact_match_count": 2,
        "accepted_prefix_lengths": [[2, 2], [4]],
        "mean_acceptance": 0.7,
        "draft_divergence_count": 1,
        "semantic_divergence_count": 1,
        "first_divergences": [],
        "cuda_available": True,
        "dtype_supported": dtype_supported,
        "skipped_configs": skipped,
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
        "exactness_blockers": [],
        "claim_note": OFFLINE_CUDA_DRIFT_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def _blocked_report(**overrides: object) -> dict[str, object]:
    configs = resolve_cuda_drift_dtype_configs()
    dtype_supported = {c.dtype: c.dtype_supported for c in configs}
    skipped = [c.to_dict() for c in configs if c.status == "skipped"]
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_051_ID,
        "status": "blocked",
        "model": DEFAULT_MODEL,
        "device": "unknown",
        "dtype": "none",
        "prompt_count": 0,
        "storage_backends": [],
        "compressor_names": [],
        "draft_len_values": [4, 8],
        "max_new_tokens": 32,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [],
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "accepted_prefix_lengths": [],
        "mean_acceptance": 0.0,
        "draft_divergence_count": 0,
        "semantic_divergence_count": 0,
        "first_divergences": [],
        "cuda_available": False,
        "dtype_supported": dtype_supported,
        "skipped_configs": skipped,
        "restore_blockers": ["CUDA unavailable — experiment skipped"],
        "draft_blockers": [],
        "verification_blockers": [],
        "exactness_blockers": [],
        "claim_note": OFFLINE_CUDA_DRIFT_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_validate_exp051_report_schema() -> None:
    assert validate_exp051_report(_synthetic_report()) == []


def test_blocked_cuda_unavailable_validates() -> None:
    report = _blocked_report()
    assert validate_exp051_report(report) == []
    assert report["cuda_available"] is False
    assert report["cells"] == []


def test_dtype_skipped_case_validates() -> None:
    report = _synthetic_report(
        skipped_configs=[
            {
                "device": "cuda",
                "dtype": "bfloat16",
                "dtype_supported": False,
                "status": "skipped",
                "skip_reason": "bfloat16 not supported on this CUDA device",
            }
        ],
        dtype="float16",
    )
    assert validate_exp051_report(report) == []


def test_exactness_failure_schema_validates() -> None:
    report = _synthetic_report(
        cells=[
            _cuda_cell(),
            _cuda_cell(
                prompt_id="drift_002",
                token_exact_match=False,
                exactkv_failures=1,
                first_divergence_idx=2,
                exactness_blocker="offline output diverged from live full greedy at token index 2",
            ),
        ],
        exactkv_failures=1,
        token_exact_match_count=1,
        exactness_blockers=[
            "float16/in_memory_kv_storage/int4_sim/dl8/drift_002: "
            "offline output diverged from live full greedy at token index 2"
        ],
        first_divergences=[
            {
                "prompt_id": "drift_002",
                "backend_name": "in_memory_kv_storage",
                "compressor_name": "int4_sim",
                "draft_len": 8,
                "dtype": "float16",
                "first_divergence_idx": 2,
            }
        ],
    )
    assert validate_exp051_report(report) == []


def test_exact_pass_schema_validates() -> None:
    report = _synthetic_report(
        cells=[_cuda_cell(draft_divergence_count=0, semantic_divergence_count=0)],
        token_exact_match_count=1,
        draft_divergence_count=0,
        semantic_divergence_count=0,
    )
    assert validate_exp051_report(report) == []


def test_restore_blocker_schema() -> None:
    report = _synthetic_report(
        cells=[_cuda_cell(restore_blocker="HfKvRestoreError", exactkv_failures=1)],
        exactkv_failures=1,
        token_exact_match_count=0,
        restore_blockers=["float16/in_memory/int4_sim/dl8/drift_001: HfKvRestoreError"],
    )
    assert validate_exp051_report(report) == []


def test_draft_and_verification_blocker_schemas() -> None:
    report = _synthetic_report(
        cells=[
            _cuda_cell(
                draft_blocker="RuntimeError: compress failed",
                exactkv_failures=1,
            )
        ],
        exactkv_failures=1,
        token_exact_match_count=0,
        draft_blockers=["float16/in_memory/int4_sim/dl8/drift_001: RuntimeError: compress failed"],
        verification_blockers=[],
    )
    assert validate_exp051_report(report) == []


def test_default_cuda_prompt_panel_size() -> None:
    prompts = default_cuda_drift_prompts()
    assert len(prompts) == 6
    categories = {p["category"] for p in prompts}
    assert "pharmacy_semantic" in categories
    assert "longbench_style" in categories


def test_skipped_cuda_configs_when_unavailable() -> None:
    configs = resolve_cuda_drift_dtype_configs()
    if torch.cuda.is_available():
        pending = [c for c in configs if c.status == "pending"]
        assert pending
    else:
        skipped = [c for c in configs if c.status == "skipped"]
        assert len(skipped) >= 2
        assert all("CUDA unavailable" in c.skip_reason for c in skipped)


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "cuda exactness panel",
        "not default runtime integration",
        "isolated experiment path",
        "existing compressor logic",
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
    os.environ.get("EXACTKV_RUN_CUDA_SMOKE") != "1",
    reason="Set EXACTKV_RUN_CUDA_SMOKE=1 to run CUDA drift smoke",
)
def test_cuda_smoke_one_cell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from exactkv.cache.offline_verifier import configure_cuda_determinism, run_offline_cuda_drift_cell
    from exactkv.cache.storage import InMemoryKVStorageBackend
    from exactkv.runtime.model_runtime import ModelRuntime

    configure_cuda_determinism()
    runtime = ModelRuntime(DEFAULT_MODEL, device="cuda", dtype="float16")
    backend = InMemoryKVStorageBackend()
    entry = default_cuda_drift_prompts()[0]
    result = run_offline_cuda_drift_cell(
        runtime,
        prompt_id=entry["prompt_id"],
        prompt=entry["prompt"],
        category=entry["category"],
        backend=backend,
        compressor_name="int4_sim",
        draft_len=4,
        dtype="float16",
        max_new_tokens=16,
    )
    assert not result.restore_blocker, result.restore_blocker
    assert not result.draft_blocker, result.draft_blocker
    assert not result.verification_blocker, result.verification_blocker
    assert result.verifier_source == VERIFIER_SOURCE
    assert result.device == "cuda"
    assert result.dtype == "float16"
    assert result.token_exact_match, result.exactness_blocker
    assert result.exactkv_failures == 0


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_CUDA_SMOKE") != "1",
    reason="Set EXACTKV_RUN_CUDA_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp051_offline_verifier_cuda_drift_panel.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp051_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_051_ID
