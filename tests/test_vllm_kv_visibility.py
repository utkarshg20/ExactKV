"""Unit tests for vLLM KV visibility helpers (no vLLM/CUDA/network)."""
from __future__ import annotations

import ast
from pathlib import Path

from exactkv.integrations.vllm_kv_visibility import (
    KV_VISIBILITY_STATUSES,
    RAW_KV_EXPORT_STATUSES,
    BoundedInspectResult,
    build_possible_adapter_path,
    bounded_inspect_surfaces,
    classify_kv_visibility_status,
    validate_exp064_report,
)

_MODULE = Path(__file__).resolve().parents[1] / "exactkv" / "integrations" / "vllm_kv_visibility.py"


class _NestedEngine:
    def __init__(self) -> None:
        self.cache_config = _CacheConfig()
        self.model_executor = object()
        self.scheduler = object()


class _CacheConfig:
    block_size = 16
    num_gpu_blocks = 100


class _FakeLlm:
    llm_engine = _NestedEngine()

    def get_kv_cache(self) -> None:
        pass


def _base_report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": "exp064_vllm_kv_visibility_probe",
        "status": "pass",
        "environment_note": "test",
        "python_executable": "/usr/bin/python3",
        "torch_version": "2.11.0+cu130",
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX A5000",
        "gpu_memory_before": "free=0.88GiB used=23.11GiB total=23.99GiB",
        "gpu_memory_after": "free=22.00GiB used=1.00GiB total=23.00GiB",
        "running_server_detected": False,
        "stopped_processes": [],
        "vllm_version": "0.23.0",
        "llm_object_initialized": True,
        "generation_smoke_attempted": True,
        "generation_smoke_passed": True,
        "generated_text_preview": "hello",
        "visible_llm_attrs": ["llm_engine"],
        "visible_engine_attrs": ["llm_engine.cache_config"],
        "visible_scheduler_attrs": [],
        "visible_cache_attrs": ["llm_engine.cache_config"],
        "visible_block_attrs": [],
        "cache_config_summary": "block_size=16",
        "kv_cache_visibility_status": "metadata_only_probe_success",
        "raw_kv_export_status": "raw_kv_export_not_available",
        "possible_adapter_path": "potential adapter path — cache metadata visible",
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


def test_kv_visibility_status_enum() -> None:
    assert len(KV_VISIBILITY_STATUSES) >= 9
    assert classify_kv_visibility_status(
        llm_initialized=False,
        running_server=True,
        oom_blocked=False,
        inspect_result=BoundedInspectResult(),
        cache_config_summary="",
        raw_kv_status="blocked_by_running_server",
    ) == "blocked_by_running_server"
    assert classify_kv_visibility_status(
        llm_initialized=False,
        running_server=False,
        oom_blocked=True,
        inspect_result=BoundedInspectResult(),
        cache_config_summary="",
        raw_kv_status="blocked_by_oom",
    ) == "blocked_by_oom"


def test_blocked_running_server_report_validates() -> None:
    assert validate_exp064_report(
        _base_report(
            status="blocked",
            llm_object_initialized=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            running_server_detected=True,
            kv_cache_visibility_status="blocked_by_running_server",
            raw_kv_export_status="blocked_by_running_server",
            possible_adapter_path="adapter blocked pending stable hook",
        )
    ) == []


def test_oom_blocked_report_validates() -> None:
    assert validate_exp064_report(
        _base_report(
            status="blocked",
            llm_object_initialized=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            kv_cache_visibility_status="blocked_by_oom",
            raw_kv_export_status="blocked_by_oom",
            blockers=["OutOfMemoryError"],
        )
    ) == []


def test_llm_init_pass_report_validates() -> None:
    assert validate_exp064_report(_base_report()) == []


def test_metadata_only_cache_visibility_report_validates() -> None:
    assert validate_exp064_report(
        _base_report(
            kv_cache_visibility_status="private_cache_attrs_visible",
            visible_cache_attrs=["llm_engine.cache_config"],
            possible_adapter_path=(
                "potential adapter path — cache metadata visible; "
                "private attrs require validation"
            ),
        )
    ) == []


def test_raw_kv_export_unavailable_report_validates() -> None:
    assert validate_exp064_report(
        _base_report(
            raw_kv_export_status="raw_kv_export_not_available",
            kv_cache_visibility_status="metadata_only_probe_success",
            possible_adapter_path="raw KV export not available",
        )
    ) == []


def test_possible_adapter_path_schema_validates() -> None:
    path = build_possible_adapter_path(
        kv_status="cache_config_visible",
        raw_kv_status="raw_kv_export_not_available",
        cache_config_summary="block_size=16",
        inspect_result=BoundedInspectResult(cache_attrs=["cache_config"]),
    )
    assert "potential adapter path" in path
    assert validate_exp064_report(_base_report(possible_adapter_path=path)) == []


def test_bounded_inspector_catches_exceptions() -> None:
    class BadDir:
        def __dir__(self) -> list[str]:
            raise RuntimeError("boom")

    result = bounded_inspect_surfaces(BadDir())
    assert result.errors
    assert validate_exp064_report(_base_report()) == []


def test_bounded_inspector_finds_cache_attrs() -> None:
    result = bounded_inspect_surfaces(_FakeLlm())
    assert any("cache" in a for a in result.cache_attrs + result.engine_attrs)


def test_no_module_level_vllm_import() -> None:
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "vllm"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "vllm"


def test_raw_kv_export_status_enum() -> None:
    assert "raw_kv_export_not_available" in RAW_KV_EXPORT_STATUSES
