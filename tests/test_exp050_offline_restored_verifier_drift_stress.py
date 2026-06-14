"""Tests for Experiment 050 offline restored-verifier drift stress (no model by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exactkv.cache.offline_verifier import (
    DEFAULT_MODEL,
    EXPERIMENT_050_ID,
    FORBIDDEN_CLAIMS,
    OFFLINE_DRIFT_STRESS_CLAIM_NOTE,
    VERIFIER_SOURCE,
    OfflineVerifierRoundTrace,
    default_drift_stress_compressors,
    default_drift_stress_prompts,
    draft_divergence_count_from_traces,
    semantic_divergence_count_from_traces,
    validate_exp050_report,
)

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_050_OFFLINE_RESTORED_VERIFIER_DRIFT_STRESS.md"
)
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_050_offline_restored_verifier_drift_stress.json"
)


def _drift_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prompt_id": "drift_001",
        "prompt": "test",
        "category": "pharmacy_semantic",
        "backend_name": "in_memory_kv_storage",
        "compressor_name": "int4_sim",
        "draft_len": 8,
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
    }
    base.update(overrides)
    return base


def _synthetic_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_050_ID,
        "model": DEFAULT_MODEL,
        "device": "cpu",
        "dtype": "torch.float32",
        "prompt_count": 12,
        "storage_backends": ["in_memory_kv_storage", "file_kv_storage"],
        "compressor_names": ["int4_sim", "k8_v4_sim", "k8_v4_boundary4_v8_sim", "int8"],
        "draft_len_values": [4, 8],
        "max_new_tokens": 32,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [
            _drift_cell(),
            _drift_cell(
                prompt_id="drift_002",
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
        "no_real_drift_observed": False,
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
        "claim_note": OFFLINE_DRIFT_STRESS_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_validate_exp050_report_schema_with_drift() -> None:
    assert validate_exp050_report(_synthetic_report()) == []


def test_no_drift_case_validates() -> None:
    report = _synthetic_report(
        cells=[
            _drift_cell(draft_divergence_count=0, semantic_divergence_count=0),
        ],
        draft_divergence_count=0,
        semantic_divergence_count=0,
        no_real_drift_observed=True,
        token_exact_match_count=1,
    )
    assert validate_exp050_report(report) == []


def test_exactkv_failures_reconcile() -> None:
    report = _synthetic_report(
        cells=[
            _drift_cell(),
            _drift_cell(
                prompt_id="drift_002",
                token_exact_match=False,
                exactkv_failures=1,
                first_divergence_idx=2,
            ),
        ],
        exactkv_failures=1,
        token_exact_match_count=1,
        first_divergences=[
            {
                "prompt_id": "drift_002",
                "backend_name": "in_memory_kv_storage",
                "compressor_name": "int4_sim",
                "draft_len": 8,
                "first_divergence_idx": 2,
            }
        ],
    )
    assert validate_exp050_report(report) == []


def test_draft_divergence_helpers() -> None:
    traces = [
        OfflineVerifierRoundTrace(
            round_idx=0,
            draft_tokens=[1, 2],
            verifier_tokens=[1, 9],
            accepted_prefix_length=1,
            correction_token=9,
            committed_tokens=[1, 9],
            all_matched=False,
            num_rejected=1,
        ),
        OfflineVerifierRoundTrace(
            round_idx=1,
            draft_tokens=[3, 4],
            verifier_tokens=[3, 4],
            accepted_prefix_length=2,
            correction_token=None,
            committed_tokens=[3, 4],
            all_matched=True,
            num_rejected=0,
        ),
    ]
    assert draft_divergence_count_from_traces(traces) == 1
    assert semantic_divergence_count_from_traces(traces, category="pharmacy_semantic") == 1
    assert semantic_divergence_count_from_traces(traces, category="code_like") == 0


def test_accepted_prefix_aggregation_field() -> None:
    report = _synthetic_report()
    assert isinstance(report["accepted_prefix_lengths"], list)


def test_restore_blocker_schema() -> None:
    report = _synthetic_report(
        cells=[_drift_cell(restore_blocker="HfKvRestoreError", exactkv_failures=1)],
        exactkv_failures=1,
        token_exact_match_count=0,
        restore_blockers=["in_memory/int4_sim/dl8/drift_001: HfKvRestoreError"],
    )
    assert validate_exp050_report(report) == []


def test_draft_and_verification_blocker_schemas() -> None:
    report = _synthetic_report(
        cells=[
            _drift_cell(
                draft_blocker="RuntimeError: compress failed",
                exactkv_failures=1,
            )
        ],
        exactkv_failures=1,
        token_exact_match_count=0,
        draft_blockers=["in_memory/int4_sim/dl8/drift_001: RuntimeError: compress failed"],
        verification_blockers=[],
    )
    assert validate_exp050_report(report) == []


def test_drift_prompt_panel_size() -> None:
    prompts = default_drift_stress_prompts()
    assert 8 <= len(prompts) <= 16
    categories = {p["category"] for p in prompts}
    assert "pharmacy_semantic" in categories
    assert "longbench_style" in categories


def test_default_drift_compressors_include_boundary4() -> None:
    names = default_drift_stress_compressors()
    assert "int4_sim" in names
    assert "k8_v4_boundary4_v8_sim" in names


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "offline restored-verifier drift stress experiment",
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
        "no real drift was observed",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
        assert phrase not in text


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model drift stress",
)
def test_model_drift_stress_pharmacy_cell() -> None:
    from exactkv.cache.offline_verifier import run_offline_drift_stress_cell
    from exactkv.cache.storage import InMemoryKVStorageBackend
    from exactkv.runtime.model_runtime import ModelRuntime

    runtime = ModelRuntime(DEFAULT_MODEL, device="cpu", dtype="float32")
    backend = InMemoryKVStorageBackend()
    entry = default_drift_stress_prompts()[0]
    result = run_offline_drift_stress_cell(
        runtime,
        prompt_id=entry["prompt_id"],
        prompt=entry["prompt"],
        category=entry["category"],
        backend=backend,
        compressor_name="k8_v4_sim",
        draft_len=8,
        max_new_tokens=32,
    )
    assert not result.restore_blocker, result.restore_blocker
    assert not result.draft_blocker, result.draft_blocker
    assert not result.verification_blocker, result.verification_blocker
    assert result.verifier_source == VERIFIER_SOURCE
    assert result.token_exact_match
    assert result.exactkv_failures == 0


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp050_offline_restored_verifier_drift_stress.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp050_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_050_ID
    assert report["exactkv_failures"] == 0
