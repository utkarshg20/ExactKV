"""Tests for Phase H unified benchmark platform."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from exactkv.benchmark.unified_benchmark_runner import (
    run_unified_benchmark,
    write_benchmark_outputs,
)
from exactkv.core.compressor_interface import CompressedKV
from exactkv.platform.public_leaderboard import (
    SCORE_WEIGHTS,
    run_public_leaderboard,
    validate_public_leaderboard,
    write_public_leaderboard_outputs,
)
from exactkv.registry.compressor_registry import get_compressor, list_compressors, register_compressor
from exactkv.schema.benchmark_schema import BenchmarkCell, BenchmarkConfig, BenchmarkRun


def test_compressor_registry_lists_builtins() -> None:
    names = list_compressors()
    assert "noop" in names
    assert "int8" in names
    assert "spectralquant" in names
    assert "spectralquant_real" in names
    assert "shard" in names
    assert "shard_real" in names
    assert "kvquant" in names
    assert "turboquant" in names


def test_kv_compressor_interface_roundtrip() -> None:
    comp = get_compressor("int8")
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    compressed = comp.compress(k, v, seed=0)
    assert isinstance(compressed, CompressedKV)
    k2, v2 = comp.decompress(compressed)
    assert k2.shape == compressed.k.shape
    assert comp.name() == "int8"


def test_dynamic_registration() -> None:
    from exactkv.core.compressor_interface import KVCompressor

    class EchoCompressor(KVCompressor):
        def name(self) -> str:
            return "echo_test"

        def compress(self, k, v, **kwargs):
            return CompressedKV(k=k, v=v, metadata={"echo": True})

        def decompress(self, compressed_kv, **kwargs):
            return compressed_kv.k, compressed_kv.v

    register_compressor("echo_test", EchoCompressor)
    assert "echo_test" in list_compressors()
    inst = get_compressor("echo_test")
    assert inst.name() == "echo_test"


def test_benchmark_config_hash_stable() -> None:
    a = BenchmarkConfig(models=("m",), compressors=("noop",), prompt_ids=("p0",), max_new_tokens_values=(4,))
    b = BenchmarkConfig(models=("m",), compressors=("noop",), prompt_ids=("p0",), max_new_tokens_values=(4,))
    assert a.config_hash() == b.config_hash()


def test_benchmark_cell_from_phase_a() -> None:
    cell = BenchmarkCell.from_phase_a_cell(
        {
            "model_name": "Qwen/Qwen2.5-0.5B",
            "compressor_name": "int8",
            "prompt_id": "p0_capital_france",
            "max_new_tokens": 4,
            "exactkv_failure": False,
            "metrics": {
                "acceptance_rate": 0.99,
                "first_divergence_index": None,
                "verifier_agreement_score": 0.99,
                "compression_ratio": 0.25,
            },
            "exactkv": {"acceptance": {"total_accepted": 4, "total_rounds": 1}},
        },
    )
    assert cell.model == "Qwen/Qwen2.5-0.5B"
    assert cell.acceptance_rate == 0.99
    assert cell.avg_accepted_span == 4.0


def test_unified_benchmark_deterministic() -> None:
    result = run_unified_benchmark(deterministic_mode=True, device="cpu")
    assert result.benchmark_run.total_cells > 0
    assert result.benchmark_run.config_hash
    assert result.validation.get("valid") is True
    assert all(isinstance(c, BenchmarkCell) for c in result.benchmark_run.cells)


def test_public_leaderboard_from_phase_a() -> None:
    phase_a = Path("reports/phaseA_benchmark.json")
    if not phase_a.is_file():
        return
    report = run_public_leaderboard(phase_a_path=phase_a)
    assert report["status"] == "leaderboard_complete"
    assert report["score_weights"] == SCORE_WEIGHTS
    assert validate_public_leaderboard(report).valid
    assert len(report["entries"]) > 0


def test_leaderboard_scoring_weights_locked() -> None:
    assert SCORE_WEIGHTS["acceptance_rate"] == 0.35
    assert SCORE_WEIGHTS["verifier_agreement"] == 0.25
    assert SCORE_WEIGHTS["first_divergence_normalized"] == 0.20
    assert SCORE_WEIGHTS["exactkv_success"] == 0.10
    assert SCORE_WEIGHTS["stability_score"] == 0.10


def test_write_outputs_contract(tmp_path: Path) -> None:
    result = run_unified_benchmark(deterministic_mode=True)
    bench_paths = write_benchmark_outputs(
        result,
        json_path=tmp_path / "benchmark.json",
        markdown_path=tmp_path / "benchmark.md",
    )
    assert bench_paths["benchmark_json"].exists()
    data = json.loads(bench_paths["benchmark_json"].read_text())
    assert data["phase_id"] == "phaseH_unified_benchmark_runner"

    phase_a = Path("reports/phaseA_benchmark.json")
    if phase_a.is_file():
        lb = run_public_leaderboard(phase_a_path=phase_a)
        lb_paths = write_public_leaderboard_outputs(
            lb,
            json_path=tmp_path / "leaderboard.json",
            markdown_path=tmp_path / "leaderboard.md",
            csv_path=tmp_path / "leaderboard.csv",
        )
        assert lb_paths["leaderboard_csv"].exists()
        csv_text = lb_paths["leaderboard_csv"].read_text()
        assert "compressor" in csv_text


def test_benchmark_run_roundtrip_schema() -> None:
    run = BenchmarkRun(
        run_id="test",
        config=BenchmarkConfig(
            models=("m",),
            compressors=("noop",),
            prompt_ids=("p",),
            max_new_tokens_values=(4,),
        ),
        config_hash="abc",
        git_commit=None,
        cells=[],
        status="ok",
    )
    d = run.to_dict()
    assert d["config_hash"] == "abc"
