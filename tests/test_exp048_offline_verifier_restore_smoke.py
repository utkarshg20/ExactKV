"""Integration tests for Experiment 048 (model smoke behind env flag)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exactkv.cache.offline_verifier import (
    DEFAULT_MODEL,
    EXPERIMENT_048_ID,
    VERIFIER_SOURCE,
    default_offline_prompts,
    run_offline_verifier_cell,
    validate_exp048_report,
)
from exactkv.cache.storage import InMemoryKVStorageBackend

_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_048_offline_verifier_restore_smoke.json"
)


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model offline verifier smoke",
)
def test_model_offline_verifier_in_memory() -> None:
    from exactkv.runtime.model_runtime import ModelRuntime

    runtime = ModelRuntime(DEFAULT_MODEL, device="cpu", dtype="float32")
    backend = InMemoryKVStorageBackend()
    entry = default_offline_prompts()[0]
    result = run_offline_verifier_cell(
        runtime,
        prompt_id=entry["prompt_id"],
        prompt=entry["prompt"],
        backend=backend,
        max_new_tokens=12,
        draft_len=4,
    )
    assert not result.restore_blocker, result.restore_blocker
    assert not result.verification_blocker, result.verification_blocker
    assert result.verifier_source == VERIFIER_SOURCE
    assert result.token_exact_match
    assert result.exactkv_failures == 0
    assert result.live_reference_token_ids == result.offline_output_token_ids


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp048_offline_verifier_restore_smoke.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp048_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_048_ID
    assert report["exactkv_failures"] == 0
