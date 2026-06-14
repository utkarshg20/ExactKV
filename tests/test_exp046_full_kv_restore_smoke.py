"""Integration tests for Experiment 046 (model smoke behind env flag)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exactkv.cache.hf_kv_restore import (
    DEFAULT_MODEL,
    EXPERIMENT_046_ID,
    run_restore_equivalence_for_prompt,
    validate_exp046_report,
)
from exactkv.cache.storage import InMemoryKVStorageBackend

_REPORT = Path(__file__).resolve().parents[1] / "reports" / "experiment_046_full_kv_restore_smoke.json"


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model restore smoke",
)
def test_model_restore_smoke_in_memory() -> None:
    from exactkv.cache.hf_kv_restore import default_smoke_prompts
    from exactkv.runtime.model_runtime import ModelRuntime

    runtime = ModelRuntime(DEFAULT_MODEL, device="cpu", dtype="float32")
    backend = InMemoryKVStorageBackend()
    entry = default_smoke_prompts()[0]
    result = run_restore_equivalence_for_prompt(
        runtime,
        prompt_id=entry["prompt_id"],
        prompt=entry["prompt"],
        backend=backend,
        max_new_tokens=8,
    )
    assert not result.restore_blocker, result.restore_blocker
    assert result.token_exact_match
    assert result.live_token_ids == result.restored_token_ids


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp046_full_kv_restore_smoke.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp046_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_046_ID
    assert report["failures_count"] == 0
