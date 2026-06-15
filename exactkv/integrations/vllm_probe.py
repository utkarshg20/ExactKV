"""vLLM feasibility probe (Phase 15A).

Safe optional import and environment inspection only. **Does not** install vLLM,
modify torch/CUDA, or wire ExactKV runtime integration.

This is a vLLM feasibility probe, not vLLM integration.
"""
from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EXPERIMENT_059_ID = "exp059_vllm_feasibility_probe"
DEFAULT_EXP059_REPORT = Path("reports/experiment_059_vllm_feasibility_probe.json")
DEFAULT_SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_SMOKE_PROMPT = "What is 2+2? Answer in one word."
DEFAULT_SMOKE_MAX_TOKENS = 8

EXP059_CLAIM_NOTE = (
    "vLLM feasibility probe (Phase 15A). Environment and import inspection only — "
    "no vLLM runtime integration, serving, batching, or dependency installation. "
    "No speed, latency, throughput, active memory savings, or production-serving "
    "claim. vLLM import failure is an environment blocker, not an ExactKV "
    "correctness failure. Default ExactKV generation behavior unchanged."
)

FORBIDDEN_CLAIMS = (
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
)


@dataclass
class VllmProbeResult:
    """Result of a vLLM feasibility probe run."""

    status: str
    python_executable: str
    platform_info: str
    torch_version: str
    cuda_available: bool
    gpu_name: str
    vllm_importable: bool
    vllm_version: str
    import_error: str
    llm_class_importable: bool
    sampling_params_importable: bool
    generation_smoke_attempted: bool
    generation_smoke_passed: bool
    generation_smoke_error: str
    visible_integration_surfaces: dict[str, str]
    kv_cache_access_status: str
    blockers: list[str] = field(default_factory=list)
    claim_note: str = EXP059_CLAIM_NOTE
    forbidden_claims: list[str] = field(default_factory=lambda: list(FORBIDDEN_CLAIMS))

    def to_report_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_id"] = EXPERIMENT_059_ID
        data["platform"] = data.pop("platform_info")
        data["forbidden_claims"] = list(self.forbidden_claims)
        return data


def _torch_environment() -> tuple[str, bool, str]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda else ""
        return torch.__version__, cuda, gpu
    except Exception as exc:  # noqa: BLE001
        return "", False, f"torch probe error: {exc}"


def _inspect_integration_surfaces(vllm_module: Any) -> dict[str, str]:
    """Describe visible vLLM integration surfaces without implementing wiring."""
    surfaces: dict[str, str] = {
        "model_loading_surface": "unknown",
        "generation_call_surface": "unknown",
        "sampling_greedy_config_surface": "unknown",
        "kv_cache_access_surface": "unknown",
        "scheduler_cache_api_surface": "unknown",
        "restored_full_kv_verifier_path": "unknown",
    }
    if vllm_module is None:
        for key in surfaces:
            surfaces[key] = "blocked"
        return surfaces

    if hasattr(vllm_module, "LLM"):
        surfaces["model_loading_surface"] = "accessible"
    else:
        surfaces["model_loading_surface"] = "blocked"

    llm_cls = getattr(vllm_module, "LLM", None)
    if llm_cls is not None and callable(getattr(llm_cls, "generate", None)):
        surfaces["generation_call_surface"] = "accessible"
    else:
        surfaces["generation_call_surface"] = "blocked"

    if hasattr(vllm_module, "SamplingParams"):
        surfaces["sampling_greedy_config_surface"] = "accessible"
    else:
        surfaces["sampling_greedy_config_surface"] = "blocked"

    kv_hints = (
        "CacheEngine",
        "CacheConfig",
        "KVCache",
        "PagedAttention",
        "BlockManager",
    )
    kv_found = any(
        importlib.util.find_spec(f"vllm.{name}") is not None
        or hasattr(vllm_module, name)
        for name in kv_hints
    )
    surfaces["kv_cache_access_surface"] = "accessible" if kv_found else "unknown"

    scheduler_hints = ("Scheduler", "SchedulerOutputs", "EngineCore")
    sched_found = any(
        importlib.util.find_spec(f"vllm.{name}") is not None
        or hasattr(vllm_module, name)
        for name in scheduler_hints
    )
    surfaces["scheduler_cache_api_surface"] = "accessible" if sched_found else "unknown"

    if surfaces["kv_cache_access_surface"] == "accessible":
        surfaces["restored_full_kv_verifier_path"] = (
            "unknown — paged KV surfaces visible; authoritative full-KV export "
            "for restored verifier not verified in this probe"
        )
    else:
        surfaces["restored_full_kv_verifier_path"] = (
            "blocked — KV cache APIs not visible at probe time"
        )
    return surfaces


def _kv_cache_access_status(surfaces: dict[str, str]) -> str:
    kv = surfaces.get("kv_cache_access_surface", "unknown")
    if kv == "accessible":
        return "partial — vLLM cache-related symbols detected; export path not implemented"
    if kv == "blocked":
        return "blocked — vLLM not importable or cache APIs not visible"
    return "unknown — requires deeper prototype inspection"


def _attempt_generation_smoke(
    *,
    model_id: str = DEFAULT_SMOKE_MODEL,
    prompt: str = DEFAULT_SMOKE_PROMPT,
    max_tokens: int = DEFAULT_SMOKE_MAX_TOKENS,
) -> tuple[bool, bool, str]:
    """Tiny greedy generation smoke when vLLM is already importable."""
    torch_version, cuda_available, _ = _torch_environment()
    if not cuda_available:
        return False, False, "CUDA unavailable — generation smoke skipped"
    if not torch_version:
        return False, False, "torch unavailable — generation smoke skipped"

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        return False, False, f"LLM/SamplingParams import failed: {exc}"

    try:
        llm = LLM(model=model_id, dtype="float16", max_model_len=256)
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)
        outputs = llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return True, False, "generation returned empty outputs"
        _ = outputs[0].outputs[0].text
        return True, True, ""
    except Exception as exc:  # noqa: BLE001
        return True, False, f"{type(exc).__name__}: {exc}"


def probe_vllm_availability(
    *,
    run_generation_smoke: bool = True,
    smoke_model_id: str = DEFAULT_SMOKE_MODEL,
) -> VllmProbeResult:
    """Probe vLLM importability and visible integration surfaces."""
    python_executable = sys.executable
    platform_info = platform.platform()
    torch_version, cuda_available, gpu_name = _torch_environment()
    blockers: list[str] = []

    vllm_module: Any | None = None
    vllm_version = ""
    import_error = ""
    llm_importable = False
    sampling_importable = False

    if importlib.util.find_spec("vllm") is None:
        import_error = "ModuleNotFoundError: No module named 'vllm'"
        blockers.append(import_error)
        surfaces = _inspect_integration_surfaces(None)
        return VllmProbeResult(
            status="blocked",
            python_executable=python_executable,
            platform_info=platform_info,
            torch_version=torch_version,
            cuda_available=cuda_available,
            gpu_name=gpu_name if cuda_available else "",
            vllm_importable=False,
            vllm_version="",
            import_error=import_error,
            llm_class_importable=False,
            sampling_params_importable=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            generation_smoke_error="",
            visible_integration_surfaces=surfaces,
            kv_cache_access_status=_kv_cache_access_status(surfaces),
            blockers=blockers,
        )

    try:
        vllm_module = importlib.import_module("vllm")
        vllm_version = str(getattr(vllm_module, "__version__", "") or "")
    except Exception as exc:  # noqa: BLE001
        import_error = f"{type(exc).__name__}: {exc}"
        blockers.append(import_error)
        surfaces = _inspect_integration_surfaces(None)
        return VllmProbeResult(
            status="blocked",
            python_executable=python_executable,
            platform_info=platform_info,
            torch_version=torch_version,
            cuda_available=cuda_available,
            gpu_name=gpu_name if cuda_available else "",
            vllm_importable=False,
            vllm_version="",
            import_error=import_error,
            llm_class_importable=False,
            sampling_params_importable=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            generation_smoke_error="",
            visible_integration_surfaces=surfaces,
            kv_cache_access_status=_kv_cache_access_status(surfaces),
            blockers=blockers,
        )

    try:
        from vllm import LLM  # noqa: F401

        llm_importable = True
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"LLM class import failed: {exc}")

    try:
        from vllm import SamplingParams  # noqa: F401

        sampling_importable = True
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"SamplingParams import failed: {exc}")

    surfaces = _inspect_integration_surfaces(vllm_module)
    gen_attempted = False
    gen_passed = False
    gen_error = ""

    if run_generation_smoke and llm_importable and sampling_importable:
        gen_attempted, gen_passed, gen_error = _attempt_generation_smoke(
            model_id=smoke_model_id,
        )
        if gen_attempted and not gen_passed and gen_error:
            blockers.append(f"generation_smoke: {gen_error}")

    if not llm_importable or not sampling_importable:
        status = "blocked"
    elif gen_attempted and not gen_passed:
        status = "failed"
    else:
        status = "pass"

    return VllmProbeResult(
        status=status,
        python_executable=python_executable,
        platform_info=platform_info,
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name if cuda_available else "",
        vllm_importable=True,
        vllm_version=vllm_version,
        import_error=import_error,
        llm_class_importable=llm_importable,
        sampling_params_importable=sampling_importable,
        generation_smoke_attempted=gen_attempted,
        generation_smoke_passed=gen_passed,
        generation_smoke_error=gen_error,
        visible_integration_surfaces=surfaces,
        kv_cache_access_status=_kv_cache_access_status(surfaces),
        blockers=blockers,
    )


def build_vllm_blocked_report(
    *,
    import_error: str = "vLLM not importable",
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    """Build a blocked Exp 059 report without attempting vLLM import."""
    torch_version, cuda_available, gpu_name = _torch_environment()
    surfaces = _inspect_integration_surfaces(None)
    blocker_list = list(blockers or [import_error])
    result = VllmProbeResult(
        status="blocked",
        python_executable=sys.executable,
        platform_info=platform.platform(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name if cuda_available else "",
        vllm_importable=False,
        vllm_version="",
        import_error=import_error,
        llm_class_importable=False,
        sampling_params_importable=False,
        generation_smoke_attempted=False,
        generation_smoke_passed=False,
        generation_smoke_error="",
        visible_integration_surfaces=surfaces,
        kv_cache_access_status=_kv_cache_access_status(surfaces),
        blockers=blocker_list,
    )
    return result.to_report_dict()


def validate_exp059_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 059 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "python_executable",
        "platform",
        "torch_version",
        "cuda_available",
        "gpu_name",
        "vllm_importable",
        "vllm_version",
        "import_error",
        "llm_class_importable",
        "sampling_params_importable",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "generation_smoke_error",
        "visible_integration_surfaces",
        "kv_cache_access_status",
        "blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_059_ID:
        errors.append("experiment_id must be exp059_vllm_feasibility_probe")
    if report.get("status") not in ("pass", "blocked", "failed"):
        errors.append("status must be pass, blocked, or failed")
    if not isinstance(report.get("cuda_available"), bool):
        errors.append("cuda_available must be a bool")
    if not isinstance(report.get("vllm_importable"), bool):
        errors.append("vllm_importable must be a bool")
    if not isinstance(report.get("blockers"), list):
        errors.append("blockers must be a list")
    surfaces = report.get("visible_integration_surfaces")
    if not isinstance(surfaces, dict):
        errors.append("visible_integration_surfaces must be a dict")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    if report.get("status") == "pass" and not report.get("vllm_importable"):
        errors.append("status pass requires vllm_importable true")
    return errors
