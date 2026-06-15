"""Tests for Experiment 058 expanded GPU memory panel (no CUDA by default)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.metrics.gpu_memory_accounting import (
    EXPERIMENT_058_ID,
    EXP058_CLAIM_NOTE,
    ExpandedMemorySlice,
    check_exp056_exactness_gate,
    run_expanded_gpu_memory_panel,
    validate_exp058_report,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_058_EXPANDED_GPU_MEMORY_PANEL.md"
_EXP056 = Path(__file__).resolve().parents[1] / "reports/experiment_056_cuda_restored_verifier_runtime_gate.json"


def _pass_exp058_report(**overrides: object) -> dict[str, object]:
    slice_entry = {
        "dtype": "float16",
        "storage_backend": "in_memory_kv_storage",
        "draft_len": 4,
        "prompt_count": 4,
        "compressors": ["int4_sim", "k8_v4_sim", "int8"],
        "measurements": [
            {
                "label": "float16/in_memory_kv_storage/dl4/restored_verifier_runtime",
                "before": {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0,
                           "max_reserved_bytes": 0, "device": "cuda", "dtype": "float16", "label": "b"},
                "after": {"allocated_bytes": 100, "reserved_bytes": 200, "max_allocated_bytes": 100,
                          "max_reserved_bytes": 200, "device": "cuda", "dtype": "float16", "label": "a"},
                "peak_allocated_bytes": 2_000_000_000,
                "peak_reserved_bytes": 2_100_000_000,
                "delta_allocated_bytes": 100,
                "delta_reserved_bytes": 200,
                "notes": "diagnostic",
            }
        ],
        "peak_allocated_by_label": {"restored_verifier_runtime": 2_000_000_000},
        "exactkv_failures": 0,
        "token_exact_match_count": 12,
        "total_cells": 12,
        "full_kv_payload_bytes": 147552,
        "stored_kv_payload_bytes": 147552,
        "blockers": [],
    }
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_058_ID,
        "status": "pass",
        "cuda_available": True,
        "device_name": "NVIDIA RTX A5000",
        "torch_version": "2.8.0",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "dtype_configs": ["float16"],
        "prompt_ids": ["offline_001", "offline_002", "offline_003", "offline_004"],
        "prompt_count": 4,
        "draft_lens": [4, 8],
        "storage_backends": ["in_memory_kv_storage", "file_kv_storage"],
        "compressors": ["int4_sim", "k8_v4_sim", "int8"],
        "max_new_tokens": 12,
        "verifier_source": "reloaded_full_kv",
        "exp056_gate_passed": True,
        "exp057_baseline_loaded": True,
        "exactness_gate_passed": True,
        "exactkv_failures": 0,
        "token_exact_match_count": 12,
        "total_cells": 12,
        "slices": [slice_entry],
        "baseline_measurements": [],
        "aggregate_peak_stats": {
            "full_greedy": {"count": 1, "min": 1_000_000_000, "max": 1_000_000_000, "mean": 1_000_000_000},
            "restored_verifier_runtime": {"count": 1, "min": 2_000_000_000, "max": 2_000_000_000, "mean": 2_000_000_000},
            "kv_capture_store_reload": {"count": 1, "min": 1_010_000_000, "max": 1_010_000_000, "mean": 1_010_000_000},
        },
        "exp057_baseline_peaks": {},
        "stability_notes": ["diagnostic only — not a memory savings claim"],
        "blockers": [],
        "claim_note": EXP058_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "active_gpu_memory_savings_claimed": False,
        "speedup_claimed": False,
        "throughput_claimed": False,
        "latency_claimed": False,
        "production_serving_claimed": False,
    }
    base.update(overrides)
    return base


def test_expanded_slice_schema() -> None:
    sl = ExpandedMemorySlice(
        dtype="float16",
        storage_backend="in_memory_kv_storage",
        draft_len=4,
        prompt_count=4,
        compressors=["int8"],
    )
    data = sl.to_dict()
    assert data["dtype"] == "float16"
    assert data["draft_len"] == 4


def test_blocked_report_validates() -> None:
    report = _pass_exp058_report(
        status="blocked",
        cuda_available=False,
        exactness_gate_passed=False,
        slices=[],
        total_cells=0,
        token_exact_match_count=0,
    )
    assert validate_exp058_report(report) == []


def test_pass_report_validates() -> None:
    assert validate_exp058_report(_pass_exp058_report()) == []


def test_exactness_failure_blocks_panel(tmp_path: Path) -> None:
    bad = tmp_path / "exp056_bad.json"
    bad.write_text(
        json.dumps(
            {
                "experiment_id": "exp056_cuda_restored_verifier_runtime_gate",
                "status": "failed",
                "cuda_available": True,
                "total_cells": 24,
                "exactkv_failures": 1,
                "token_exact_match_count": 23,
                "cuda_blockers": [],
                "restore_blockers": [],
                "draft_blockers": [],
                "verification_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    result = run_expanded_gpu_memory_panel(exp056_report_path=bad)
    assert result.status == "blocked"
    assert not result.exp056_gate_passed


def test_run_blocked_without_cuda(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available")
    good = tmp_path / "exp056_pass.json"
    good.write_text(
        json.dumps(
            {
                "experiment_id": "exp056_cuda_restored_verifier_runtime_gate",
                "status": "pass",
                "cuda_available": True,
                "total_cells": 24,
                "exactkv_failures": 0,
                "token_exact_match_count": 24,
                "cuda_blockers": [],
                "restore_blockers": [],
                "draft_blockers": [],
                "verification_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    result = run_expanded_gpu_memory_panel(exp056_report_path=good)
    assert result.status == "blocked"
    assert validate_exp058_report(result.to_dict()) == []


def test_claim_flags_false() -> None:
    report = _pass_exp058_report(active_gpu_memory_savings_claimed=True)
    errors = validate_exp058_report(report)
    assert any("active_gpu_memory_savings_claimed" in e for e in errors)


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "expanded gpu memory",
        "not a memory savings claim",
        "active gpu memory savings are not claimed",
        "explicit experimental",
        "vllm",
        "lmcache",
        "default exactkv generation behavior is unchanged",
        "stability",
    ):
        assert phrase in text, phrase


@pytest.mark.skipif(
    os.environ.get("EXACTKV_RUN_CUDA_SMOKE") != "1",
    reason="Set EXACTKV_RUN_CUDA_SMOKE=1 for CUDA smoke",
)
def test_cuda_smoke_expanded_panel() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not _EXP056.is_file():
        pytest.skip("Exp 056 report required")
    ok, blockers = check_exp056_exactness_gate(_EXP056)
    if not ok:
        pytest.skip(f"Exp 056 gate not passed: {blockers}")
    result = run_expanded_gpu_memory_panel(exp056_report_path=_EXP056)
    payload = result.to_dict()
    assert validate_exp058_report(payload) == []
    if result.status == "pass":
        assert result.exactkv_failures == 0
