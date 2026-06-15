"""Unit tests for GPU memory accounting helpers (no CUDA required)."""
from __future__ import annotations

import pytest
import torch

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.metrics.gpu_memory_accounting import (
    EXPERIMENT_057_ID,
    EXP057_CLAIM_NOTE,
    CudaMemoryMeasurement,
    CudaMemorySnapshot,
    check_exp056_exactness_gate,
    measure_cuda_memory,
    snapshot_cuda_memory,
    validate_exp057_report,
)


def _snapshot(**overrides: object) -> CudaMemorySnapshot:
    base = {
        "allocated_bytes": 100,
        "reserved_bytes": 200,
        "max_allocated_bytes": 150,
        "max_reserved_bytes": 250,
        "device": "cuda",
        "dtype": "float16",
        "label": "test_before",
    }
    base.update(overrides)
    return CudaMemorySnapshot(**base)  # type: ignore[arg-type]


def test_memory_snapshot_schema() -> None:
    snap = _snapshot()
    data = snap.to_dict()
    for key in (
        "allocated_bytes",
        "reserved_bytes",
        "max_allocated_bytes",
        "max_reserved_bytes",
        "device",
        "dtype",
        "label",
    ):
        assert key in data
    assert data["allocated_bytes"] >= 0


def test_measurement_schema() -> None:
    before = _snapshot(label="before")
    after = _snapshot(label="after", allocated_bytes=120, reserved_bytes=220)
    meas = CudaMemoryMeasurement(
        label="full_greedy",
        before=before,
        after=after,
        peak_allocated_bytes=180,
        peak_reserved_bytes=280,
        delta_allocated_bytes=20,
        delta_reserved_bytes=20,
        notes="diagnostic only",
    )
    data = meas.to_dict()
    assert data["label"] == "full_greedy"
    assert data["peak_allocated_bytes"] == 180
    assert data["before"]["label"] == "before"


def test_snapshot_cuda_memory_requires_cuda() -> None:
    if torch.cuda.is_available():
        snap = snapshot_cuda_memory("probe", dtype="float16")
        assert snap.allocated_bytes >= 0
    else:
        with pytest.raises(RuntimeError, match="CUDA unavailable"):
            snapshot_cuda_memory("probe")


def test_measure_cuda_memory_cpu_fn_without_cuda() -> None:
    if torch.cuda.is_available():
        result, meas = measure_cuda_memory("noop", lambda: 42, dtype="float16")
        assert result == 42
        assert meas.peak_allocated_bytes >= 0
    else:
        with pytest.raises(RuntimeError, match="CUDA unavailable"):
            measure_cuda_memory("noop", lambda: None)


def test_check_exp056_gate_missing_report() -> None:
    ok, blockers = check_exp056_exactness_gate(
        __import__("pathlib").Path("/nonexistent/exp056.json")
    )
    assert not ok
    assert blockers


def test_validate_exp057_blocked_report() -> None:
    report = {
        "experiment_id": EXPERIMENT_057_ID,
        "status": "blocked",
        "cuda_available": False,
        "device_name": "",
        "torch_version": "2.0.0",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "dtype_configs": [],
        "prompt_count": 2,
        "storage_backend": "in_memory_kv_storage",
        "compressors": ["int4_sim", "k8_v4_sim", "int8"],
        "verifier_source": "reloaded_full_kv",
        "exactness_gate_passed": False,
        "exactkv_failures": 0,
        "token_exact_match_count": 0,
        "measurements": [],
        "full_kv_payload_bytes": 0,
        "stored_kv_payload_bytes": 0,
        "active_gpu_memory_savings_claimed": False,
        "speedup_claimed": False,
        "throughput_claimed": False,
        "latency_claimed": False,
        "production_serving_claimed": False,
        "blockers": ["CUDA unavailable"],
        "claim_note": EXP057_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    assert validate_exp057_report(report) == []


def test_validate_exp057_pass_report() -> None:
    meas = {
        "label": "model_loaded",
        "before": _snapshot().to_dict(),
        "after": _snapshot(label="after").to_dict(),
        "peak_allocated_bytes": 500_000_000,
        "peak_reserved_bytes": 600_000_000,
        "delta_allocated_bytes": 0,
        "delta_reserved_bytes": 0,
        "notes": "diagnostic",
    }
    report = {
        "experiment_id": EXPERIMENT_057_ID,
        "status": "pass",
        "cuda_available": True,
        "device_name": "NVIDIA RTX A5000",
        "torch_version": "2.8.0",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "dtype_configs": ["float16"],
        "prompt_count": 2,
        "storage_backend": "in_memory_kv_storage",
        "compressors": ["int4_sim", "k8_v4_sim", "int8"],
        "verifier_source": "reloaded_full_kv",
        "exactness_gate_passed": True,
        "exactkv_failures": 0,
        "token_exact_match_count": 6,
        "measurements": [meas],
        "full_kv_payload_bytes": 1_048_576,
        "stored_kv_payload_bytes": 1_048_576,
        "active_gpu_memory_savings_claimed": False,
        "speedup_claimed": False,
        "throughput_claimed": False,
        "latency_claimed": False,
        "production_serving_claimed": False,
        "blockers": [],
        "claim_note": EXP057_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    assert validate_exp057_report(report) == []


def test_claim_flags_must_be_false() -> None:
    report = {
        "experiment_id": EXPERIMENT_057_ID,
        "cuda_available": True,
        "device_name": "GPU",
        "torch_version": "2.8.0",
        "model_id": "m",
        "dtype_configs": [],
        "prompt_count": 2,
        "storage_backend": "in_memory_kv_storage",
        "compressors": ["int8"],
        "verifier_source": "reloaded_full_kv",
        "exactness_gate_passed": False,
        "exactkv_failures": 1,
        "token_exact_match_count": 0,
        "measurements": [],
        "full_kv_payload_bytes": 0,
        "stored_kv_payload_bytes": 0,
        "active_gpu_memory_savings_claimed": True,
        "speedup_claimed": False,
        "throughput_claimed": False,
        "latency_claimed": False,
        "production_serving_claimed": False,
        "blockers": ["exactness"],
        "claim_note": EXP057_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    errors = validate_exp057_report(report)
    assert any("active_gpu_memory_savings_claimed" in e for e in errors)
