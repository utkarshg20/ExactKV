"""Tests for Experiment 063 vLLM API surface recon docs and reports."""
from __future__ import annotations

from pathlib import Path

from exactkv.integrations.vllm_surface_recon import (
    EXPERIMENT_063_ID,
    EXP063_CLAIM_NOTE,
    KV_CACHE_ACCESS_STATUSES,
    validate_exp063_report,
)
from exactkv.integrations.vllm_probe import FORBIDDEN_CLAIMS

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_063_VLLM_API_SURFACE_RECON.md"


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_063_ID,
        "status": "pass",
        "environment_note": "RunPod vLLM template",
        "python_executable": "/usr/bin/python3",
        "python_version": "3.12.13",
        "platform": "Linux",
        "torch_version": "2.11.0+cu130",
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX A5000",
        "gpu_memory_summary": "free=0.24GiB used=23.32GiB total=23.56GiB",
        "running_server_detected": True,
        "stopped_processes": [],
        "vllm_importable": True,
        "vllm_version": "0.23.0",
        "llm_class_importable": True,
        "sampling_params_importable": True,
        "generation_smoke_attempted": False,
        "generation_smoke_passed": False,
        "generation_smoke_error": "",
        "llm_object_initialized": False,
        "visible_top_level_modules": ["vllm", "vllm.config"],
        "visible_config_surfaces": ["CacheConfig"],
        "visible_engine_surfaces": ["LLM"],
        "visible_scheduler_surfaces": ["Scheduler"],
        "visible_cache_surfaces": ["CacheEngine"],
        "kv_cache_access_status": "blocked_by_running_server",
        "possible_adapter_path": "potential adapter path requires idle GPU prototype validation",
        "blockers": ["GPU busy — vLLM/OpenAI server detected"],
        "claim_note": EXP063_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_exp063_id_and_kv_statuses() -> None:
    assert EXPERIMENT_063_ID == "exp063_vllm_api_surface_recon"
    assert len(KV_CACHE_ACCESS_STATUSES) >= 7


def test_import_only_report_validates() -> None:
    assert validate_exp063_report(
        _report(
            kv_cache_access_status="import_only_unknown",
            running_server_detected=False,
            blockers=[],
        )
    ) == []


def test_generation_fail_report_validates() -> None:
    assert validate_exp063_report(
        _report(
            status="failed",
            generation_smoke_attempted=True,
            generation_smoke_passed=False,
            generation_smoke_error="RuntimeError: OOM",
            blockers=["generation_smoke: RuntimeError: OOM"],
            kv_cache_access_status="blocked_by_oom",
        )
    ) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "api surface reconnaissance",
        "not exactkv",
        "not vllm integration",
        "default runtime",
        "private",
        "throughput",
        "vericache",
    ):
        assert phrase in text, phrase


def test_doc_no_forbidden_positive_claims() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in ("exactkv supports vllm", "vllm integration works", "throughput improved"):
        assert phrase not in text
