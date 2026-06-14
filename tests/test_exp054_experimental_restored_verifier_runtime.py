"""Tests for Experiment 054 experimental restored-verifier runtime (no model by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL, FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import VERIFIER_SOURCE
from exactkv.runtime.experimental import (
    EXPERIMENT_054_ID,
    EXP054_CLAIM_NOTE,
    default_experimental_smoke_config,
    report_to_exp054_json,
    run_experimental_restored_verifier,
    validate_exp054_report,
    ExperimentalRestoredVerifierConfig,
)

_DOC_EXP = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_054_EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md"
)
_DOC_RUNTIME = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENTAL_RESTORED_VERIFIER_RUNTIME.md"
)
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_054_experimental_restored_verifier_runtime.json"
)


def _synthetic_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_054_ID,
        "status": "pass",
        "runtime_mode": "restored_verifier_offline",
        "enabled": True,
        "model": DEFAULT_MODEL,
        "device": "cpu",
        "dtype": "float32",
        "prompt_count": 4,
        "storage_backends": ["in_memory_kv_storage"],
        "compressor_names": ["int4_sim", "k8_v4_sim", "int8"],
        "draft_len": 4,
        "draft_lens": [4],
        "max_new_tokens": 12,
        "verifier_source": VERIFIER_SOURCE,
        "total_cells": 12,
        "exactkv_failures": 0,
        "token_exact_match_count": 12,
        "mean_acceptance": 0.82,
        "draft_divergence_count": 5,
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
        "runner_called": True,
        "claim_note": EXP054_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_exp054_report_schema() -> None:
    assert validate_exp054_report(_synthetic_report()) == []


def test_disabled_exp054_report_schema() -> None:
    result = run_experimental_restored_verifier(ExperimentalRestoredVerifierConfig.disabled())
    payload = report_to_exp054_json(result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp054_report(payload) == []


def test_default_smoke_config_explicit_opt_in() -> None:
    cfg = default_experimental_smoke_config()
    assert cfg.enabled is True
    assert cfg.mode.value == "restored_verifier_offline"
    assert cfg.verifier_source == VERIFIER_SOURCE
    assert len(cfg.prompt_ids) >= 4


def test_doc_exp054_caveats() -> None:
    text = _DOC_EXP.read_text(encoding="utf-8").lower()
    for phrase in (
        "non-default experimental restored-verifier runtime",
        "explicitly enabled",
        "default exactkv generation behavior is unchanged",
        "vllm",
        "lmcache",
        "remote prefix",
        "throughput",
        "vericache",
        "active memory savings",
    ):
        assert phrase in text, phrase


def test_doc_runtime_caveats() -> None:
    text = _DOC_RUNTIME.read_text(encoding="utf-8").lower()
    for phrase in (
        "experimental",
        "explicit opt-in",
        "default exactkv",
        "vllm",
        "lmcache",
        "remote prefix",
        "not production serving",
        "does not prove",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    for doc in (_DOC_EXP, _DOC_RUNTIME):
        text = doc.read_text(encoding="utf-8").lower()
        for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
            assert phrase not in text


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model smoke",
)
def test_model_smoke_via_experimental_runtime() -> None:
    config = default_experimental_smoke_config(max_new_tokens=12)
    result = run_experimental_restored_verifier(config)
    assert result.runner_called
    assert result.runner_report is not None
    assert result.runner_report.exactkv_failures == 0
    assert result.runner_report.token_exact_match_count == result.runner_report.total_cells


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp054_experimental_restored_verifier_runtime.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp054_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_054_ID
    assert report["enabled"] is True
    assert report["exactkv_failures"] == 0
