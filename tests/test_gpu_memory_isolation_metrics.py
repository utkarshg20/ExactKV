"""Tests for Experiment 031 GPU memory isolation metrics helpers."""
from __future__ import annotations

import pytest

from exactkv.metrics.gpu_memory_isolation import (
    CudaMemoryReading,
    assert_memory_artifact_safe,
    collect_gpu_memory_environment,
    summarize_memory_trials,
)


def test_summarize_memory_trials_empty() -> None:
    stats = summarize_memory_trials([])
    assert stats["num_trials"] == 0
    assert stats["mean_peak_allocated"] == 0


def test_summarize_memory_trials_mean_max() -> None:
    trials = [
        CudaMemoryReading(100, 200, 150, 250, 120, 220, 50, 50),
        CudaMemoryReading(100, 200, 170, 270, 130, 230, 70, 70),
    ]
    stats = summarize_memory_trials(trials)
    assert stats["num_trials"] == 2
    assert stats["mean_peak_allocated"] == 160.0
    assert stats["max_peak_allocated"] == 170
    assert stats["mean_delta_peak_allocated_vs_model_loaded"] == 60.0


def test_assert_memory_artifact_safe_rejects_forbidden() -> None:
    with pytest.raises(ValueError, match="Forbidden fields"):
        assert_memory_artifact_safe({"tokens_per_second": 1.0})
    with pytest.raises(ValueError, match="Forbidden fields"):
        assert_memory_artifact_safe({"throughput": 1.0})


def test_assert_memory_artifact_safe_accepts_valid() -> None:
    assert_memory_artifact_safe({
        "peak_allocated": 1024,
        "v5_accounting": {"total_kv_footprint_bytes": 512},
    })


def test_collect_gpu_memory_environment_cpu() -> None:
    meta = collect_gpu_memory_environment("cpu")
    assert "torch_version" in meta
    assert "transformers_version" in meta
