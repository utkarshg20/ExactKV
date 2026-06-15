"""Unit tests for vLLM surface recon helpers (no vLLM/CUDA/network)."""
from __future__ import annotations

import ast
from pathlib import Path

from exactkv.integrations.vllm_surface_recon import (
    KV_CACHE_ACCESS_STATUSES,
    build_possible_adapter_path,
    classify_kv_cache_access_status,
    validate_exp063_report,
)

_MODULE = Path(__file__).resolve().parents[1] / "exactkv" / "integrations" / "vllm_surface_recon.py"


def _base_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": "exp063_vllm_api_surface_recon",
        "status": "pass",
        "environment_note": "test",
        "python_executable": "/usr/bin/python3",
        "python_version": "3.12.0",
        "platform": "Linux",
        "torch_version": "2.11.0+cu130",
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX A5000",
        "gpu_memory_summary": "free=10.00GiB used=10.00GiB total=20.00GiB",
        "running_server_detected": False,
        "stopped_processes": [],
        "vllm_importable": True,
        "vllm_version": "0.23.0",
        "llm_class_importable": True,
        "sampling_params_importable": True,
        "generation_smoke_attempted": False,
        "generation_smoke_passed": False,
        "generation_smoke_error": "",
        "llm_object_initialized": False,
        "visible_top_level_modules": ["vllm"],
        "visible_config_surfaces": ["CacheConfig"],
        "visible_engine_surfaces": ["LLM"],
        "visible_scheduler_surfaces": [],
        "visible_cache_surfaces": ["CacheEngine"],
        "kv_cache_access_status": "module_names_visible",
        "possible_adapter_path": "potential adapter path — cache module names visible",
        "blockers": [],
        "claim_note": "test",
        "forbidden_claims": [
            "speedup",
            "latency improvement",
            "throughput improvement",
            "memory savings",
            "active memory savings",
            "production serving",
            "vLLM integrated",
            "vLLM integration exists",
            "vericache throughput reproduced",
            "full vericache reproduction",
        ],
    }
    base.update(overrides)
    return base


def test_kv_cache_status_enum() -> None:
    assert classify_kv_cache_access_status(
        vllm_importable=False,
        llm_importable=False,
        cache_modules=[],
        object_attrs=[],
        generation_blocked_by_oom=False,
        generation_blocked_by_server=False,
    ) == "not_importable"
    assert classify_kv_cache_access_status(
        vllm_importable=True,
        llm_importable=True,
        cache_modules=["CacheEngine"],
        object_attrs=[],
        generation_blocked_by_oom=False,
        generation_blocked_by_server=False,
    ) == "module_names_visible"
    assert classify_kv_cache_access_status(
        vllm_importable=True,
        llm_importable=True,
        cache_modules=[],
        object_attrs=[],
        generation_blocked_by_oom=True,
        generation_blocked_by_server=False,
    ) == "blocked_by_oom"
    for status in KV_CACHE_ACCESS_STATUSES:
        assert isinstance(status, str)


def test_possible_adapter_path_schema() -> None:
    path = build_possible_adapter_path(
        kv_status="module_names_visible",
        cache_surfaces=["CacheEngine"],
        object_attrs=[],
        running_server=False,
        oom_blocked=False,
    )
    assert "potential adapter path" in path
    assert "integration works" not in path.lower()
    assert validate_exp063_report(_base_report(possible_adapter_path=path)) == []


def test_import_only_report_validates() -> None:
    report = _base_report(
        status="pass",
        kv_cache_access_status="import_only_unknown",
        visible_cache_surfaces=[],
        possible_adapter_path="cache access remains unknown",
    )
    assert validate_exp063_report(report) == []


def test_running_server_blocked_report_validates() -> None:
    report = _base_report(
        status="pass",
        running_server_detected=True,
        kv_cache_access_status="blocked_by_running_server",
        blockers=["GPU busy — vLLM/OpenAI server detected"],
        generation_smoke_attempted=False,
    )
    assert validate_exp063_report(report) == []


def test_oom_blocked_report_validates() -> None:
    report = _base_report(
        status="pass",
        gpu_memory_summary="free=0.24GiB used=23.32GiB total=23.56GiB",
        kv_cache_access_status="blocked_by_oom",
        blockers=["GPU memory low"],
    )
    assert validate_exp063_report(report) == []


def test_object_level_recon_pass_report_validates() -> None:
    report = _base_report(
        llm_object_initialized=True,
        generation_smoke_attempted=True,
        generation_smoke_passed=True,
        object_level_attr_names=["llm_engine", "cache_config"],
        kv_cache_access_status="object_attrs_visible_private_only",
    )
    report["object_level_attr_names"] = report.get("object_level_attr_names", [])
    # schema uses visible lists only; object attrs not in required top-level - they're in result
    assert validate_exp063_report(report) == []


def test_no_module_level_vllm_import() -> None:
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "vllm"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "vllm"
