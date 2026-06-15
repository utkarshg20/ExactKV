"""Tests for Experiment 064 vLLM KV visibility probe docs and reports."""
from __future__ import annotations

from pathlib import Path

from exactkv.integrations.vllm_kv_visibility import (
    EXPERIMENT_064_ID,
    EXP064_CLAIM_NOTE,
    KV_VISIBILITY_STATUSES,
    validate_exp064_report,
)
from exactkv.integrations.vllm_probe import FORBIDDEN_CLAIMS

_DOC = Path(__file__).resolve().parents[1] / "docs" / "EXPERIMENT_064_VLLM_KV_VISIBILITY_PROBE.md"


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment_id": EXPERIMENT_064_ID,
        "status": "blocked",
        "environment_note": "RunPod vLLM template",
        "python_executable": "/usr/bin/python3",
        "torch_version": "2.11.0+cu130",
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX A5000",
        "gpu_memory_before": "free=0.88GiB used=23.11GiB total=23.99GiB",
        "gpu_memory_after": "free=0.88GiB used=23.11GiB total=23.99GiB",
        "running_server_detected": True,
        "stopped_processes": [],
        "vllm_version": "0.23.0",
        "llm_object_initialized": False,
        "generation_smoke_attempted": False,
        "generation_smoke_passed": False,
        "generated_text_preview": "",
        "visible_llm_attrs": [],
        "visible_engine_attrs": [],
        "visible_scheduler_attrs": [],
        "visible_cache_attrs": [],
        "visible_block_attrs": [],
        "cache_config_summary": "",
        "kv_cache_visibility_status": "blocked_by_running_server",
        "raw_kv_export_status": "blocked_by_running_server",
        "possible_adapter_path": "adapter blocked pending stable hook",
        "blockers": ["GPU busy — vLLM/OpenAI server detected"],
        "claim_note": EXP064_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }
    base.update(overrides)
    return base


def test_exp064_id_and_statuses() -> None:
    assert EXPERIMENT_064_ID == "exp064_vllm_kv_visibility_probe"
    assert len(KV_VISIBILITY_STATUSES) >= 9


def test_server_blocked_report_validates() -> None:
    assert validate_exp064_report(_report()) == []


def test_doc_caveats() -> None:
    text = _DOC.read_text(encoding="utf-8").lower()
    for phrase in (
        "kv/cache visibility probe",
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
