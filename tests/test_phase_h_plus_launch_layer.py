"""Tests for Phase H+ launch layer (adapters, scale config, release package)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from exactkv.adapters.shard_real_adapter import ShardRealKVCompressor
from exactkv.adapters.spectralquant_real_adapter import (
    SpectralQuantRealKVCompressor,
    spectralquant_available,
)
from exactkv.benchmark.scale_7b_benchmark import expand_prompt_suite
from exactkv.configs.load_scale_config import (
    DEFAULT_SCALE_CONFIG,
    load_scale_config,
    map_compressors,
    resolve_device,
)
from exactkv.core.compressor_interface import CompressedKV
from exactkv.platform.release_packager import build_release_package
from exactkv.registry.compressor_registry import get_compressor, list_compressors


def test_registry_includes_phase_h_plus_compressors() -> None:
    names = list_compressors()
    assert "spectralquant_real" in names
    assert "shard_real" in names


def test_spectralquant_real_adapter_contract() -> None:
    comp = SpectralQuantRealKVCompressor()
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = comp.compress(k, v, seed=0)
    assert isinstance(out, CompressedKV)
    assert "approximation_mode" in out.metadata
    assert "compression_ratio" in out.metadata
    k2, v2 = comp.decompress(out)
    assert k2.shape == out.k.shape
    assert v2.shape == out.v.shape
    assert comp.name() == "spectralquant_real"
    _ = spectralquant_available()


def test_shard_real_adapter_probe_scores() -> None:
    comp = ShardRealKVCompressor()
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = comp.compress(k, v, seed=0)
    assert out.metadata.get("stability_score_estimate") is not None
    assert out.metadata.get("divergence_risk_estimate") is not None
    assert out.metadata.get("acceptance_proxy_score") is not None
    assert out.metadata.get("probe_only") is True
    k2, v2 = comp.decompress(out)
    assert k2.shape == out.k.shape


def test_get_compressor_phase_h_plus_roundtrip() -> None:
    for name in ("spectralquant_real", "shard_real"):
        comp = get_compressor(name)
        assert comp.name() == name


def test_scale_config_loads() -> None:
    cfg = load_scale_config(DEFAULT_SCALE_CONFIG)
    assert "meta-llama/Llama-3.1-8B" in cfg["models"]
    assert int(cfg["prompts"]) == 50
    assert cfg["max_new_tokens"] == [4, 8, 16]
    mapped = map_compressors(
        list(cfg["compressors"]),
        cfg.get("compressor_map") or {},
    )
    assert "int4_sim" in mapped
    assert "spectralquant" in mapped
    assert "shard" in mapped


def test_expand_prompt_suite_to_50() -> None:
    prompts = expand_prompt_suite(50)
    assert len(prompts) == 50
    assert prompts[0]["prompt_id"]
    assert prompts[49]["prompt_id"] != prompts[0]["prompt_id"]


def test_resolve_device_auto() -> None:
    device = resolve_device("auto")
    assert device in ("cpu", "cuda")


def test_release_packager_writes_bundle(tmp_path: Path) -> None:
    phase_a = Path("reports/phaseA_benchmark.json")
    if phase_a.is_file():
        manifest = build_release_package(release_dir=tmp_path / "release")
        assert manifest["release_dir"]
        out = tmp_path / "release"
        for fname in (
            "README_PUBLIC.md",
            "benchmark_summary.md",
            "methodology.md",
            "repro_command.sh",
            "leaderboard_final.json",
        ):
            assert (out / fname).is_file(), fname
        repro = (out / "repro_command.sh").read_text()
        assert "exactkv_repro.py --reports-only" in repro


def test_cli_spectralquant_check_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/exactkv.py", "run", "spectralquant-check"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "approximation_mode=" in proc.stdout


def test_cli_shard_analysis_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/exactkv.py", "run", "shard-analysis"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "stability_score_estimate=" in proc.stdout


def test_scale_summary_schema_if_present() -> None:
    path = Path("reports/scale_7b/scale_summary.json")
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    assert data.get("phase_id") == "phaseH_plus_scale_7b_8b_benchmark"
    assert "outputs" in data
