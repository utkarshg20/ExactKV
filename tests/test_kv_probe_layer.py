"""Tests for Phase D runtime KV probe layer."""
from __future__ import annotations

import json
from pathlib import Path

from exactkv.runtime.kv_probe_layer import (
    PHASE_D_ID,
    PROBE_MODES,
    build_deterministic_probe_cell,
    build_layer_drift_report,
    build_memory_profile_report,
    run_phase_d_runtime_probe,
    simulate_compression_on_kv,
    validate_phase_d_report,
    write_phase_d_reports,
)


def test_deterministic_probe_grid() -> None:
    report = run_phase_d_runtime_probe(deterministic_mode=True, seed=42)
    assert validate_phase_d_report(report).valid
    assert report["phase_id"] == PHASE_D_ID
    expected = len(report["models_evaluated"]) * 4 * len(PROBE_MODES)
    assert report["total_cells"] == expected


def test_compression_sim_no_mutation() -> None:
    import torch

    k = [torch.randn(1, 4, 8, 16)]
    v = [torch.randn(1, 4, 8, 16)]
    k_orig = k[0].clone()
    simulate_compression_on_kv(k, v, "int4_sim", seed=0)
    assert torch.allclose(k[0], k_orig)


def test_three_report_outputs(tmp_path: Path) -> None:
    report = run_phase_d_runtime_probe(deterministic_mode=True)
    paths = write_phase_d_reports(
        report,
        runtime_path=tmp_path / "phaseD_runtime_probe.json",
        memory_path=tmp_path / "phaseD_memory_profile.json",
        layer_path=tmp_path / "phaseD_layer_drift.json",
        visuals_dir=tmp_path / "visuals",
    )
    assert Path(paths["phaseD_runtime_probe"]).exists()
    assert Path(paths["phaseD_memory_profile"]).exists()
    assert Path(paths["phaseD_layer_drift"]).exists()
    memory = json.loads(Path(paths["phaseD_memory_profile"]).read_text())
    layer = json.loads(Path(paths["phaseD_layer_drift"]).read_text())
    assert memory["report_type"] == "memory_profile"
    assert layer["report_type"] == "layer_drift"


def test_memory_and_layer_reports() -> None:
    report = run_phase_d_runtime_probe(deterministic_mode=True)
    memory = build_memory_profile_report(report)
    layer = build_layer_drift_report(report)
    assert memory.get("aggregates")
    assert layer.get("compression_sensitivity_by_mode")


def test_probe_cell_fields() -> None:
    cell = build_deterministic_probe_cell(
        model_name="Qwen/Qwen2.5-0.5B",
        prompt_id="p0",
        prompt_text="test",
        compression_mode="int4_sim",
        max_new_tokens=4,
        seed=42,
    )
    assert cell.memory_proxy["kv_total_bytes"] > 0
    assert cell.divergence_metrics["first_divergence_index"] is not None or cell.token_exact_match


def test_no_runtime_commit_flags() -> None:
    report = run_phase_d_runtime_probe(deterministic_mode=True)
    assert report["exactkv_generator_modified"] is False
    assert report["runtime_commit_authorized"] is False
    assert report["instrumentation_only"] is True
