"""Tests for Experiment 056 CUDA restored-verifier runtime gate (no CUDA by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from exactkv.cache.hf_kv_restore import DEFAULT_MODEL, FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import VERIFIER_SOURCE
from exactkv.runtime.experimental import (
    EXPERIMENT_056_ID,
    EXP056_CLAIM_NOTE,
    RUNTIME_PATH_EXPERIMENTAL,
    CLI_OPT_IN_REQUIRED,
    report_to_exp056_json,
    run_cuda_restored_verifier_runtime_gate,
    validate_exp056_report,
)

_DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "EXPERIMENT_056_CUDA_RESTORED_VERIFIER_RUNTIME_GATE.md"
)
_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_056_cuda_restored_verifier_runtime_gate.json"
)


def _blocked_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_056_ID,
        "status": "blocked",
        "cuda_available": False,
        "model": DEFAULT_MODEL,
        "runtime_path": RUNTIME_PATH_EXPERIMENTAL,
        "cli_opt_in_required": True,
        "device": "unknown",
        "dtype_configs": [],
        "dtype_supported": {"float16": False, "bfloat16": False},
        "prompt_count": 4,
        "storage_backend": "in_memory_kv_storage",
        "compressor_names": ["int4_sim", "k8_v4_sim", "int8"],
        "draft_len": 4,
        "max_new_tokens": 12,
        "verifier_source": VERIFIER_SOURCE,
        "total_cells": 0,
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "mean_acceptance": 0.0,
        "draft_divergence_count": 0,
        "accepted_prefix_lengths": [],
        "first_divergences": [],
        "skipped_configs": [
            {
                "device": "cuda",
                "dtype": "float16",
                "dtype_supported": False,
                "status": "skipped",
                "skip_reason": "CUDA unavailable",
            }
        ],
        "cuda_blockers": ["CUDA unavailable"],
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
        "claim_note": EXP056_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def _pass_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_056_ID,
        "status": "pass",
        "cuda_available": True,
        "model": DEFAULT_MODEL,
        "runtime_path": RUNTIME_PATH_EXPERIMENTAL,
        "cli_opt_in_required": True,
        "device": "cuda",
        "dtype_configs": ["float16"],
        "dtype_supported": {"float16": True, "bfloat16": False},
        "prompt_count": 4,
        "storage_backend": "in_memory_kv_storage",
        "compressor_names": ["int4_sim", "k8_v4_sim", "int8"],
        "draft_len": 4,
        "max_new_tokens": 12,
        "verifier_source": VERIFIER_SOURCE,
        "total_cells": 12,
        "exactkv_failures": 0,
        "token_exact_match_count": 12,
        "mean_acceptance": 0.82,
        "draft_divergence_count": 5,
        "accepted_prefix_lengths": [[2, 2], [4]],
        "first_divergences": [],
        "skipped_configs": [],
        "cuda_blockers": [],
        "restore_blockers": [],
        "draft_blockers": [],
        "verification_blockers": [],
        "claim_note": EXP056_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_cuda_unavailable_blocked_report_validates() -> None:
    assert validate_exp056_report(_blocked_report()) == []


def test_blocked_report_not_marked_pass() -> None:
    report = _blocked_report()
    assert report["status"] == "blocked"
    assert report["cuda_available"] is False
    assert report["total_cells"] == 0


def test_dtype_skipped_report_validates() -> None:
    report = _pass_report(
        dtype_configs=["float16"],
        skipped_configs=[
            {
                "device": "cuda",
                "dtype": "bfloat16",
                "dtype_supported": False,
                "status": "skipped",
                "skip_reason": "bfloat16 not supported on this CUDA device",
            }
        ],
    )
    assert validate_exp056_report(report) == []


def test_cuda_pass_report_schema_validates() -> None:
    assert validate_exp056_report(_pass_report()) == []


def test_exactness_failure_report_schema_validates() -> None:
    report = _pass_report(
        status="failed",
        total_cells=2,
        exactkv_failures=1,
        token_exact_match_count=1,
        cuda_blockers=["float16: exactkv_failures=1"],
        first_divergences=[
            {
                "prompt_id": "offline_001",
                "compressor_name": "int4_sim",
                "storage_backend": "in_memory_kv_storage",
                "draft_len": 4,
                "first_divergence_idx": 2,
            }
        ],
    )
    assert validate_exp056_report(report) == []


def test_runtime_path_field_present() -> None:
    report = _pass_report()
    assert report["runtime_path"] == "run_experimental_restored_verifier"


def test_cli_opt_in_required_true() -> None:
    report = _pass_report()
    assert report["cli_opt_in_required"] is True


def test_skipped_configs_explicit() -> None:
    report = _blocked_report()
    assert report["skipped_configs"]
    assert report["skipped_configs"][0]["skip_reason"]


def test_run_gate_blocked_when_no_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available — use EXACTKV_RUN_CUDA_SMOKE for live gate")
    result = run_cuda_restored_verifier_runtime_gate()
    assert not result.cuda_available
    assert result.status == "blocked"
    assert result.total_cells == 0
    payload = report_to_exp056_json(result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp056_report(payload) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "cuda exactness gate",
        "explicit experimental restored-verifier runtime",
        "explicitly enabled",
        "default exactkv generation behavior is unchanged",
        "vllm",
        "lmcache",
        "remote prefix",
        "throughput",
        "vericache",
        "active memory savings",
        "not a performance result",
    ):
        assert phrase in text, phrase


def test_doc_no_positive_forbidden_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("achieves speedup", "memory savings claim", "production serving ready"):
        assert phrase not in text


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_CUDA_SMOKE") != "1",
    reason="Set EXACTKV_RUN_CUDA_SMOKE=1 to run CUDA runtime gate smoke",
)
def test_cuda_smoke_runtime_gate() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    result = run_cuda_restored_verifier_runtime_gate()
    payload = report_to_exp056_json(result)
    payload["forbidden_claims"] = list(FORBIDDEN_CLAIMS)
    assert validate_exp056_report(payload) == []
    assert result.cuda_available
    if result.total_cells > 0:
        assert result.exactkv_failures == 0


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_CUDA_SMOKE") != "1",
    reason="Set EXACTKV_RUN_CUDA_SMOKE=1 to validate on-disk report",
)
def test_report_file_if_present() -> None:
    if not _REPORT.is_file():
        pytest.skip("Run scripts/research/run_exp056_cuda_restored_verifier_runtime_gate.py first")
    report = json.loads(_REPORT.read_text(encoding="utf-8"))
    assert validate_exp056_report(report) == []
    assert report["experiment_id"] == EXPERIMENT_056_ID
    if report.get("cuda_available") and report.get("total_cells", 0) > 0:
        assert report["exactkv_failures"] == 0
