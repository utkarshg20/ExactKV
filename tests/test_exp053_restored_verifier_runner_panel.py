"""Tests for Experiment 053 runner-backed drift panel (no model by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL, FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import VERIFIER_SOURCE
from exactkv.cache.restored_verifier_runner import (
    EXPERIMENT_053_ID,
    EXP053_CLAIM_NOTE,
    RestoredVerifierRunConfig,
    default_panel_config,
    default_panel_prompt_ids,
    report_to_exp053_json,
    validate_exp053_report,
    RestoredVerifierRunReport,
    RestoredVerifierCellResult,
    aggregate_blockers,
)

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_053_RESTORED_VERIFIER_RUNNER_PANEL.md"
)
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_053_restored_verifier_runner_panel.json"
)


def _panel_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prompt_id": "drift_001",
        "compressor_name": "int4_sim",
        "storage_backend": "in_memory_kv_storage",
        "draft_len": 8,
        "token_exact_match": True,
        "exactkv_failure": 0,
        "accepted_prefix_lengths": [2, 2],
        "first_divergence": None,
        "mean_acceptance": 0.75,
        "draft_divergence_count": 1,
        "semantic_divergence_count": 1,
        "category": "pharmacy_semantic",
        "restore_blocker": "",
        "draft_blocker": "",
        "verification_blocker": "",
    }
    base.update(overrides)
    return base


def _synthetic_report(**overrides: object) -> dict[str, object]:
    cfg = default_panel_config()
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_053_ID,
        "status": "pass",
        "config": cfg.to_dict(),
        "model": DEFAULT_MODEL,
        "device": "cpu",
        "dtype": "float32",
        "prompt_count": 12,
        "storage_backends": ["in_memory_kv_storage", "file_kv_storage"],
        "compressor_names": ["int4_sim", "k8_v4_sim", "k8_v4_boundary4_v8_sim", "int8"],
        "draft_len_values": [4, 8],
        "max_new_tokens": 32,
        "verifier_source": VERIFIER_SOURCE,
        "cells": [
            _panel_cell(),
            _panel_cell(
                prompt_id="drift_002",
                draft_divergence_count=0,
                semantic_divergence_count=0,
            ),
        ],
        "total_cells": 2,
        "exactkv_failures": 0,
        "token_exact_match_count": 2,
        "mean_acceptance": 0.7,
        "draft_divergence_count": 1,
        "semantic_divergence_count": 1,
        "no_real_drift_observed": False,
        "first_divergences": [],
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
        "blockers": {
            "restore_blockers": [],
            "draft_blockers": [],
            "verification_blockers": [],
        },
        "claim_note": EXP053_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_validate_exp053_report_schema() -> None:
    assert validate_exp053_report(_synthetic_report()) == []


def test_runner_config_in_report() -> None:
    report = _synthetic_report()
    assert isinstance(report["config"], dict)
    assert report["config"]["namespace_prefix"] == "exp053"
    assert report["config"]["verifier_source"] == VERIFIER_SOURCE


def test_exactness_pass_schema() -> None:
    assert validate_exp053_report(_synthetic_report()) == []


def test_exactness_failure_schema() -> None:
    report = _synthetic_report(
        status="failed",
        cells=[
            _panel_cell(),
            _panel_cell(
                prompt_id="drift_003",
                token_exact_match=False,
                exactkv_failure=1,
                first_divergence=2,
            ),
        ],
        total_cells=2,
        exactkv_failures=1,
        token_exact_match_count=1,
        first_divergences=[
            {
                "prompt_id": "drift_003",
                "compressor_name": "int4_sim",
                "storage_backend": "in_memory_kv_storage",
                "draft_len": 8,
                "first_divergence_idx": 2,
            }
        ],
    )
    assert validate_exp053_report(report) == []


def test_no_drift_case_validates() -> None:
    report = _synthetic_report(
        cells=[_panel_cell(draft_divergence_count=0, semantic_divergence_count=0)],
        total_cells=1,
        token_exact_match_count=1,
        draft_divergence_count=0,
        semantic_divergence_count=0,
        no_real_drift_observed=True,
    )
    assert validate_exp053_report(report) == []


def test_drift_case_validates() -> None:
    report = _synthetic_report(
        draft_divergence_count=3,
        semantic_divergence_count=2,
        no_real_drift_observed=False,
    )
    assert validate_exp053_report(report) == []


def test_blocker_aggregation_in_runner_report() -> None:
    cells = [
        RestoredVerifierCellResult(
            prompt_id="drift_001",
            compressor_name="int4_sim",
            storage_backend="in_memory_kv_storage",
            draft_len=4,
            token_exact_match=False,
            exactkv_failure=1,
            accepted_prefix_lengths=[],
            restore_blocker="HfKvRestoreError",
        )
    ]
    blockers = aggregate_blockers(cells)
    report = RestoredVerifierRunReport(
        experiment_id=EXPERIMENT_053_ID,
        config=default_panel_config(),
        cells=cells,
        total_cells=1,
        token_exact_match_count=0,
        exactkv_failures=1,
        mean_acceptance=0.0,
        draft_divergence_count=0,
        semantic_divergence_count=0,
        no_real_drift_observed=True,
        blockers=blockers,
        claim_note=EXP053_CLAIM_NOTE,
        status="failed",
    )
    payload = report_to_exp053_json(report)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert payload["restore_blockers"]
    assert validate_exp053_report(payload) == []


def test_default_panel_prompt_count() -> None:
    full = default_panel_prompt_ids(full_panel=True)
    reduced = default_panel_prompt_ids(full_panel=False)
    assert 8 <= len(full) <= 16
    assert 8 <= len(reduced) <= 12


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "runner-backed offline restored-verifier panel",
        "not default runtime integration",
        "isolated experiment path",
        "vllm",
        "lmcache",
        "remote prefix",
        "generation and verification behavior is unchanged",
        "throughput",
        "vericache",
        "active memory savings",
        "no real drift",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
        assert phrase not in text


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to run real model panel smoke",
)
def test_model_smoke_one_cell_via_runner() -> None:
    from exactkv.cache.restored_verifier_runner import run_restored_verifier

    config = default_panel_config(
        prompt_ids=default_panel_prompt_ids(full_panel=False)[:1],
        compressor_names=["int8"],
        draft_len_values=[4],
        max_new_tokens=12,
    )
    report = run_restored_verifier(config, experiment_id=EXPERIMENT_053_ID)
    assert report.total_cells == 1
    assert report.exactkv_failures == 0
    assert report.cells[0].token_exact_match


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_MODEL_SMOKE") != "1",
    reason="Set EXACTKV_RUN_MODEL_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp053_restored_verifier_runner_panel.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp053_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_053_ID
    assert report["exactkv_failures"] == 0
