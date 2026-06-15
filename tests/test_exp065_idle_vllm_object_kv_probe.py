"""Tests for Experiment 065 idle-GPU vLLM object KV probe docs and reports."""
from __future__ import annotations

from pathlib import Path

from exactkv.integrations.vllm_kv_visibility import (
    EXPERIMENT_065_ID,
    EXP065_CLAIM_NOTE,
    EXP065_KV_VISIBILITY_STATUSES,
    build_exp065_possible_adapter_path,
    classify_exp065_kv_visibility_status,
    validate_exp065_report,
    BoundedInspectResult,
)
from exactkv.integrations.vllm_probe import FORBIDDEN_CLAIMS

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md"


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_065_ID,
        "status": "pass",
        "environment_note": "idle vLLM CUDA-13 pod",
        "python_executable": "/usr/bin/python3",
        "torch_version": "2.11.0+cu130",
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX A5000",
        "gpu_memory_before": "free=22.00GiB used=1.00GiB total=23.00GiB",
        "gpu_memory_after": "free=18.00GiB used=5.00GiB total=23.00GiB",
        "running_server_detected": False,
        "stopped_processes": [],
        "vllm_version": "0.23.0",
        "llm_object_initialized": True,
        "generation_smoke_attempted": True,
        "generation_smoke_passed": True,
        "generated_text_preview": "hello world",
        "visible_llm_attrs": ["llm_engine"],
        "visible_engine_attrs": ["llm_engine.cache_config"],
        "visible_model_executor_attrs": ["model_executor"],
        "visible_scheduler_attrs": [],
        "visible_cache_attrs": ["llm_engine.cache_config"],
        "visible_block_attrs": ["block_size"],
        "cache_config_summary": "block_size=16",
        "kv_cache_visibility_status": "engine_cache_metadata_visible",
        "raw_kv_export_status": "raw_kv_export_not_available",
        "possible_adapter_path": "potential adapter path — engine/cache metadata visible",
        "blockers": [],
        "claim_note": EXP065_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_exp065_id_and_statuses() -> None:
    assert EXPERIMENT_065_ID == "exp065_idle_vllm_object_kv_probe"
    assert len(EXP065_KV_VISIBILITY_STATUSES) >= 9


def test_idle_pass_report_validates() -> None:
    assert validate_exp065_report(_report()) == []


def test_running_server_blocked_report_validates() -> None:
    assert validate_exp065_report(
        _report(
            status="blocked",
            llm_object_initialized=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            running_server_detected=True,
            kv_cache_visibility_status="blocked_by_running_server",
            raw_kv_export_status="blocked_by_running_server",
            possible_adapter_path="adapter blocked pending idle GPU",
            blockers=["Running vLLM/OpenAI/model server detected"],
        )
    ) == []


def test_oom_blocked_report_validates() -> None:
    assert validate_exp065_report(
        _report(
            status="blocked",
            llm_object_initialized=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            kv_cache_visibility_status="blocked_by_oom",
            raw_kv_export_status="blocked_by_oom",
            blockers=["OutOfMemoryError"],
        )
    ) == []


def test_llm_init_failure_validates() -> None:
    assert validate_exp065_report(
        _report(
            status="blocked",
            llm_object_initialized=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            kv_cache_visibility_status="llm_init_failed",
            raw_kv_export_status="raw_kv_export_not_available",
            blockers=["RuntimeError: Engine core initialization failed"],
        )
    ) == []


def test_cache_config_visible_status_validates() -> None:
    assert validate_exp065_report(
        _report(
            kv_cache_visibility_status="cache_config_visible",
            visible_cache_attrs=[],
            cache_config_summary="block_size=16",
        )
    ) == []


def test_private_cache_attrs_visible_validates() -> None:
    assert validate_exp065_report(
        _report(
            kv_cache_visibility_status="private_cache_attrs_visible",
            visible_cache_attrs=["llm_engine.kv_cache"],
            possible_adapter_path="private attrs require validation",
        )
    ) == []


def test_raw_kv_unavailable_validates() -> None:
    assert validate_exp065_report(
        _report(
            raw_kv_export_status="raw_kv_export_not_available",
            kv_cache_visibility_status="engine_cache_metadata_visible",
        )
    ) == []


def test_possible_adapter_path_schema() -> None:
    path = build_exp065_possible_adapter_path(
        kv_status="engine_cache_metadata_visible",
        raw_kv_status="raw_kv_export_not_available",
        cache_config_summary="block_size=16",
        inspect_result=BoundedInspectResult(
            cache_attrs=["cache_config"],
            engine_attrs=["llm_engine"],
        ),
    )
    assert "potential adapter path" in path
    assert validate_exp065_report(_report(possible_adapter_path=path)) == []


def test_classify_engine_cache_metadata() -> None:
    assert (
        classify_exp065_kv_visibility_status(
            llm_initialized=True,
            init_attempted=True,
            running_server=False,
            oom_blocked=False,
            init_failed=False,
            inspect_result=BoundedInspectResult(
                engine_attrs=["llm_engine"],
                cache_attrs=["cache_config"],
            ),
            cache_config_summary="block_size=16",
            raw_kv_status="raw_kv_export_not_available",
        )
        == "engine_cache_metadata_visible"
    )


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "idle-gpu",
        "not exactkv",
        "not vllm integration",
        "default runtime",
        "private",
        "raw kv export",
        "throughput",
        "vericache",
    ):
        assert phrase in text, phrase


def test_doc_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("exactkv supports vllm", "vllm integration works", "throughput improved"):
        assert phrase not in text
