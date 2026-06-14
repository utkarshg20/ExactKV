"""Tests for Experiment 049 offline verifier lossy draft (no model download by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exactkv.cache.offline_verifier import (
    DEFAULT_MODEL,
    EXPERIMENT_049_ID,
    FORBIDDEN_CLAIMS,
    OFFLINE_LOSSY_CLAIM_NOTE,
    VERIFIER_SOURCE,
    OfflineVerifierRoundTrace,
    default_lossy_compressors,
    mean_acceptance_from_traces,
    validate_exp049_report,
)

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_049_OFFLINE_VERIFIER_LOSSY_DRAFT.md"
)
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_049_offline_verifier_lossy_draft.json"
)


def _lossy_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prompt_id": "offline_001",
        "prompt": "test",
        "backend_name": "in_memory_kv_storage",
        "compressor_name": "int8",
        "cache_format": "dynamic_v5",
        "draft_source": "int8",
        "verifier_source": VERIFIER_SOURCE,
        "live_reference_token_ids": [1, 2, 3, 4],
        "offline_output_token_ids": [1, 2, 3, 4],
        "token_exact_match": True,
        "exactkv_failures": 0,
        "accepted_prefix_lengths": [2, 2],
        "mean_acceptance": 0.75,
        "first_divergence_idx": None,
        "restore_blocker": "",
        "draft_blocker": "",
        "verification_blocker": "",
    }
    base.update(overrides)
    return base


def _synthetic_report(**overrides: object) -> dict[str, object]:
    cells = [
        _lossy_cell(),
        _lossy_cell(
            prompt_id="offline_002",
            backend_name="file_kv_storage",
            compressor_name="int4_sim",
            draft_source="int4_sim",
            mean_acceptance=0.5,
        ),
    ]
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_049_ID,
        "model": DEFAULT_MODEL,
        "device": "cpu",
        "dtype": "torch.float32",
        "prompt_count": 2,
        "storage_backends": ["in_memory_kv_storage", "file_kv_storage"],
        "compressor_names": ["int8", "int4_sim"],
        "draft_len": 4,
        "max_new_tokens": 12,
        "verifier_source": VERIFIER_SOURCE,
        "cells": cells,
        "exactkv_failures": 0,
        "token_exact_match_count": 2,
        "accepted_prefix_lengths": [[2, 2], [1, 3]],
        "mean_acceptance": 0.625,
        "first_divergences": [],
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
        "claim_note": OFFLINE_LOSSY_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_validate_exp049_report_schema() -> None:
    assert validate_exp049_report(_synthetic_report()) == []


def test_lossy_cell_row_schema_fields() -> None:
    report = _synthetic_report()
    cell = report["cells"][0]
    for field in (
        "compressor_name",
        "draft_source",
        "verifier_source",
        "mean_acceptance",
        "exactkv_failures",
        "accepted_prefix_lengths",
    ):
        assert field in cell


def test_report_counts_reconcile() -> None:
    report = _synthetic_report(
        cells=[
            _lossy_cell(),
            _lossy_cell(
                prompt_id="offline_002",
                token_exact_match=False,
                exactkv_failures=1,
                first_divergence_idx=1,
            ),
        ],
        exactkv_failures=1,
        token_exact_match_count=1,
        first_divergences=[
            {
                "prompt_id": "offline_002",
                "backend_name": "in_memory_kv_storage",
                "compressor_name": "int8",
                "first_divergence_idx": 1,
            }
        ],
    )
    assert validate_exp049_report(report) == []


def test_mean_acceptance_aggregation_helper() -> None:
    traces = [
        OfflineVerifierRoundTrace(
            round_idx=0,
            draft_tokens=[1, 2, 3, 4],
            verifier_tokens=[1, 2],
            accepted_prefix_length=2,
            correction_token=5,
            committed_tokens=[1, 2, 5],
            all_matched=False,
            num_rejected=2,
        ),
        OfflineVerifierRoundTrace(
            round_idx=1,
            draft_tokens=[6, 7],
            verifier_tokens=[6, 7],
            accepted_prefix_length=2,
            correction_token=None,
            committed_tokens=[6, 7],
            all_matched=True,
            num_rejected=0,
        ),
    ]
    assert mean_acceptance_from_traces(traces) == pytest.approx(4 / 6)


def test_restore_blocker_schema() -> None:
    report = _synthetic_report(
        cells=[_lossy_cell(restore_blocker="HfKvRestoreError", exactkv_failures=1)],
        exactkv_failures=1,
        token_exact_match_count=0,
        restore_blockers=["in_memory/int8/offline_001: HfKvRestoreError"],
    )
    assert validate_exp049_report(report) == []


def test_draft_blocker_schema() -> None:
    report = _synthetic_report(
        cells=[_lossy_cell(draft_blocker="RuntimeError: compress failed", exactkv_failures=1)],
        exactkv_failures=1,
        token_exact_match_count=0,
        draft_blockers=["in_memory/int8/offline_001: RuntimeError: compress failed"],
    )
    assert validate_exp049_report(report) == []


def test_verification_blocker_schema() -> None:
    report = _synthetic_report(
        cells=[
            _lossy_cell(
                verification_blocker="round 0: RuntimeError",
                exactkv_failures=1,
            )
        ],
        exactkv_failures=1,
        token_exact_match_count=0,
        verification_blockers=["in_memory/int8/offline_001: round 0: RuntimeError"],
    )
    assert validate_exp049_report(report) == []


def test_default_lossy_compressors_include_int8() -> None:
    names = default_lossy_compressors()
    assert "int8" in names


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "offline verifier restore experiment",
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
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model lossy offline verifier",
)
def test_model_lossy_offline_verifier_int8() -> None:
    from exactkv.cache.offline_verifier import default_offline_prompts, run_offline_lossy_verifier_cell
    from exactkv.cache.storage import InMemoryKVStorageBackend
    from exactkv.runtime.model_runtime import ModelRuntime

    runtime = ModelRuntime(DEFAULT_MODEL, device="cpu", dtype="float32")
    backend = InMemoryKVStorageBackend()
    entry = default_offline_prompts()[0]
    result = run_offline_lossy_verifier_cell(
        runtime,
        prompt_id=entry["prompt_id"],
        prompt=entry["prompt"],
        backend=backend,
        compressor_name="int8",
        max_new_tokens=12,
        draft_len=4,
    )
    assert not result.restore_blocker, result.restore_blocker
    assert not result.draft_blocker, result.draft_blocker
    assert not result.verification_blocker, result.verification_blocker
    assert result.verifier_source == VERIFIER_SOURCE
    assert result.draft_source == "int8"
    assert result.token_exact_match
    assert result.exactkv_failures == 0
    assert result.live_reference_token_ids == result.offline_output_token_ids


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp049_offline_verifier_lossy_draft.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp049_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_049_ID
    assert report["exactkv_failures"] == 0
