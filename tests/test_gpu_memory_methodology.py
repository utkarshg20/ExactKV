"""Tests for Experiment 018 GPU memory pilot helpers."""
from __future__ import annotations

import pytest

from exactkv.metrics.gpu_memory_pilot import (
    PILOT_GPU_FIELD_NAMES,
    GpuMemorySnapshot,
    assert_pilot_artifact_safe,
    cuda_available,
    pilot_fields_not_in_standard_schema,
)


class TestGpuMemoryMethodology:
    def test_cuda_available_does_not_raise_without_cuda(self):
        # Should return bool on any platform.
        assert isinstance(cuda_available(), bool)

    def test_pilot_field_names_exclude_active_gpu_kv_bytes(self):
        assert "active_gpu_kv_bytes" not in PILOT_GPU_FIELD_NAMES
        assert "gpu_peak_allocated_during_run_bytes" in PILOT_GPU_FIELD_NAMES

    def test_pilot_fields_not_in_standard_schema(self):
        assert pilot_fields_not_in_standard_schema() is True

    def test_snapshot_to_dict_uses_pilot_names_only(self):
        snap = GpuMemorySnapshot(
            gpu_baseline_model_loaded_bytes=100,
            gpu_allocated_after_prefill_bytes=110,
            gpu_peak_allocated_during_run_bytes=150,
            gpu_allocated_after_run_bytes=140,
            gpu_allocated_after_cleanup_bytes=105,
        )
        d = snap.to_dict()
        assert set(d.keys()) == PILOT_GPU_FIELD_NAMES
        assert "active_gpu_kv_bytes" not in d

    def test_assert_pilot_artifact_safe_rejects_forbidden_fields(self):
        with pytest.raises(ValueError, match="throughput"):
            assert_pilot_artifact_safe({"throughput": 1.0})
        with pytest.raises(ValueError, match="active_gpu_kv_bytes"):
            assert_pilot_artifact_safe({"active_gpu_kv_bytes": 100})

    def test_assert_pilot_artifact_safe_accepts_pilot_structure(self):
        artifact = {
            "manifest": {"experiment": "018"},
            "cells": [
                {
                    "gpu_pilot_observations": {
                        "gpu_baseline_model_loaded_bytes": 1,
                        "gpu_allocated_after_prefill_bytes": 2,
                        "gpu_peak_allocated_during_run_bytes": 3,
                        "gpu_allocated_after_run_bytes": 4,
                        "gpu_allocated_after_cleanup_bytes": 5,
                    },
                    "v5_accounting": {"total_kv_footprint_bytes": 100},
                }
            ],
        }
        assert_pilot_artifact_safe(artifact)

    @pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
    def test_measure_raises_without_cuda_guard(self):
        # cuda_available is True here; just verify import path works.
        from exactkv.metrics.gpu_memory_pilot import measure_exactkv_cell_gpu_memory

        assert callable(measure_exactkv_cell_gpu_memory)
