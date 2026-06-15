"""vLLM API surface and KV-cache visibility reconnaissance (Phase 15C).

Safe optional import and introspection only. **Does not** wire ExactKV into vLLM,
patch vLLM, or claim integration.

This is vLLM API surface reconnaissance, not ExactKV-vLLM integration.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import re
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from exactkv.integrations.vllm_probe import (
    DEFAULT_SMOKE_MAX_TOKENS,
    DEFAULT_SMOKE_MODEL,
    DEFAULT_SMOKE_PROMPT,
    FORBIDDEN_CLAIMS,
)

EXPERIMENT_063_ID = "exp063_vllm_api_surface_recon"
DEFAULT_EXP063_REPORT = Path("reports/experiment_063_vllm_api_surface_recon.json")

EXP063_CLAIM_NOTE = (
    "vLLM API surface reconnaissance (Phase 15C). Import-level and optional "
    "object-level visibility inspection only — no ExactKV vLLM integration, serving, "
    "batching, or performance claims. Visible private vLLM attributes are not treated "
    "as stable APIs. ExactKV default runtime unchanged."
)

KV_CACHE_ACCESS_STATUSES = (
    "not_importable",
    "import_only_unknown",
    "module_names_visible",
    "object_attrs_visible_private_only",
    "public_api_visible",
    "blocked_by_oom",
    "blocked_by_running_server",
    "blocked_by_exception",
)

# Names to probe via find_spec / guarded dir() — no model loading.
_TOP_LEVEL_MODULE_HINTS = (
    "vllm",
    "vllm.config",
    "vllm.engine",
    "vllm.engine.llm_engine",
    "vllm.engine.arg_utils",
    "vllm.core",
    "vllm.core.scheduler",
    "vllm.core.block_manager",
    "vllm.worker",
    "vllm.model_executor",
    "vllm.attention",
    "vllm.v1",
    "vllm.entrypoints",
    "vllm.entrypoints.llm",
    "vllm.entrypoints.openai",
)

_CONFIG_CLASS_HINTS = (
    "CacheConfig",
    "ModelConfig",
    "SchedulerConfig",
    "EngineArgs",
    "VllmConfig",
)

_ENGINE_SURFACE_HINTS = (
    "LLMEngine",
    "AsyncLLMEngine",
    "EngineCore",
    "LLM",
)

_SCHEDULER_SURFACE_HINTS = (
    "Scheduler",
    "SchedulerOutputs",
    "BlockManager",
    "SchedulerContext",
)

_CACHE_SURFACE_HINTS = (
    "CacheEngine",
    "CacheConfig",
    "KVCache",
    "PagedAttention",
    "BlockManager",
    "CacheManager",
)

_OBJECT_ATTR_HINTS = (
    "llm_engine",
    "engine",
    "cache_engine",
    "scheduler",
    "model_executor",
    "kv_cache",
    "cache_config",
)

_MIN_FREE_GIB_FOR_LLM_INIT = float(os.environ.get("VLLM_RECON_MIN_FREE_GIB", "4.0"))


@dataclass
class GpuPreflight:
    """GPU and server preflight snapshot."""

    nvidia_smi_available: bool
    gpu_memory_summary: str
    free_gib: float | None
    used_gib: float | None
    total_gib: float | None
    running_server_detected: bool
    server_process_hints: list[str]
    stopped_processes: list[str]
    raw_nvidia_smi: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VllmSurfaceReconResult:
    """Result of Exp 063 vLLM API surface reconnaissance."""

    status: str
    environment_note: str
    python_executable: str
    python_version: str
    platform_info: str
    torch_version: str
    cuda_available: bool
    cuda_runtime: str
    gpu_name: str
    gpu_memory_summary: str
    running_server_detected: bool
    stopped_processes: list[str]
    vllm_importable: bool
    vllm_version: str
    llm_class_importable: bool
    sampling_params_importable: bool
    generation_smoke_attempted: bool
    generation_smoke_passed: bool
    generation_smoke_error: str
    generated_text_preview: str
    llm_object_initialized: bool
    visible_top_level_modules: list[str]
    visible_config_surfaces: list[str]
    visible_engine_surfaces: list[str]
    visible_scheduler_surfaces: list[str]
    visible_cache_surfaces: list[str]
    object_level_attr_names: list[str]
    kv_cache_access_status: str
    possible_adapter_path: str
    blockers: list[str] = field(default_factory=list)
    claim_note: str = EXP063_CLAIM_NOTE
    forbidden_claims: list[str] = field(default_factory=lambda: list(FORBIDDEN_CLAIMS))

    def to_report_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_id"] = EXPERIMENT_063_ID
        data["platform"] = data.pop("platform_info")
        data["forbidden_claims"] = list(self.forbidden_claims)
        return data


def _torch_environment() -> tuple[str, bool, str, str]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda else ""
        cuda_rt = str(getattr(torch.version, "cuda", "") or "")
        return torch.__version__, cuda, gpu, cuda_rt
    except Exception as exc:  # noqa: BLE001
        return "", False, "", f"torch probe error: {exc}"


def _run_command(cmd: list[str], *, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


def _parse_nvidia_smi_memory(text: str) -> tuple[float | None, float | None, float | None]:
    """Parse used/total MiB from nvidia-smi output."""
    used = total = None
    for line in text.splitlines():
        match = re.search(
            r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            used = float(match.group(1)) / 1024.0
            total = float(match.group(2)) / 1024.0
            break
    if used is None or total is None:
        return None, None, None
    free = max(total - used, 0.0)
    return free, used, total


def _detect_running_server() -> tuple[bool, list[str]]:
    code, out, _ = _run_command(["ps", "aux"])
    if code != 0:
        return False, []
    hints: list[str] = []
    patterns = (
        "vllm.entrypoints.openai",
        "openai.api_server",
        "vllm serve",
        "api_server",
        "uvicorn",
    )
    for line in out.splitlines():
        lower = line.lower()
        if "grep" in lower:
            continue
        vllm_match = "vllm" in lower and any(p in lower for p in patterns)
        uvicorn_match = "uvicorn" in lower and ("8000" in line or "vllm" in lower)
        if vllm_match or uvicorn_match:
            hints.append(line.strip()[:200])
    return bool(hints), hints


def _stop_template_server_processes() -> list[str]:
    """Stop likely vLLM/OpenAI server processes; return stopped command lines."""
    stopped: list[str] = []
    running, hints = _detect_running_server()
    if not running:
        return stopped
    for hint in hints:
        tokens = hint.split()
        if len(tokens) < 2:
            continue
        try:
            pid = int(tokens[1])
        except ValueError:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(hint[:200])
        except OSError:
            continue
    return stopped


def preflight_gpu_check(*, stop_server: bool = False) -> GpuPreflight:
    """Inspect GPU memory and running vLLM server processes."""
    stopped: list[str] = []
    if stop_server:
        stopped = _stop_template_server_processes()

    code, smi_out, smi_err = _run_command(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
    )
    raw = smi_out if code == 0 else smi_err
    free_gib = used_gib = total_gib = None
    gpu_name = ""
    if code == 0 and smi_out.strip():
        line = smi_out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if parts:
            gpu_name = parts[0]
        if len(parts) >= 3:
            used_mib = float(re.sub(r"[^0-9.]", "", parts[1]) or "0")
            total_mib = float(re.sub(r"[^0-9.]", "", parts[2]) or "0")
            used_gib = used_mib / 1024.0
            total_gib = total_mib / 1024.0
            free_gib = max(total_gib - used_gib, 0.0)

    if free_gib is None:
        _, smi_human, _ = _run_command(["nvidia-smi"])
        raw = smi_human or raw
        free_gib, used_gib, total_gib = _parse_nvidia_smi_memory(smi_human)

    running, hints = _detect_running_server()
    if free_gib is not None and used_gib is not None and total_gib is not None:
        summary = f"free={free_gib:.2f}GiB used={used_gib:.2f}GiB total={total_gib:.2f}GiB"
    else:
        summary = raw.strip()[:300] if raw else "nvidia-smi unavailable"

    return GpuPreflight(
        nvidia_smi_available=code == 0,
        gpu_memory_summary=summary,
        free_gib=free_gib,
        used_gib=used_gib,
        total_gib=total_gib,
        running_server_detected=running,
        server_process_hints=hints,
        stopped_processes=stopped,
        raw_nvidia_smi=raw[:2000],
    )


def _module_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _discover_top_level_modules() -> list[str]:
    found: list[str] = []
    for name in _TOP_LEVEL_MODULE_HINTS:
        if _module_importable(name):
            found.append(name)
    return found


def _discover_symbols(module_name: str, hints: tuple[str, ...]) -> list[str]:
    if not _module_importable(module_name):
        return []
    try:
        mod = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return []
    names = set(dir(mod))
    return [h for h in hints if h in names]


def _discover_config_surfaces() -> list[str]:
    found: list[str] = []
    for mod_name in ("vllm.config", "vllm.engine.arg_utils", "vllm"):
        found.extend(_discover_symbols(mod_name, _CONFIG_CLASS_HINTS))
    return sorted(set(found))


def _discover_engine_surfaces() -> list[str]:
    found: list[str] = []
    for mod_name in ("vllm", "vllm.engine.llm_engine", "vllm.v1.engine"):
        found.extend(_discover_symbols(mod_name, _ENGINE_SURFACE_HINTS))
    return sorted(set(found))


def _discover_scheduler_surfaces() -> list[str]:
    found: list[str] = []
    for mod_name in ("vllm.core.scheduler", "vllm.core", "vllm"):
        found.extend(_discover_symbols(mod_name, _SCHEDULER_SURFACE_HINTS))
    return sorted(set(found))


def _discover_cache_module_surfaces() -> list[str]:
    found: list[str] = []
    for mod_name in (
        "vllm.config",
        "vllm.core.block_manager",
        "vllm.attention",
        "vllm.worker.cache_engine",
        "vllm",
    ):
        found.extend(_discover_symbols(mod_name, _CACHE_SURFACE_HINTS))
    return sorted(set(found))


def _safe_attr_names(obj: Any) -> list[str]:
    try:
        names = [n for n in dir(obj) if not n.startswith("__")]
    except Exception:  # noqa: BLE001
        return []
    hits = [n for n in names if any(h in n.lower() for h in _OBJECT_ATTR_HINTS)]
    return sorted(set(hits))[:40]


def _has_enough_gpu_memory(free_gib: float | None) -> bool:
    if free_gib is None:
        return False
    return free_gib >= _MIN_FREE_GIB_FOR_LLM_INIT


def classify_kv_cache_access_status(
    *,
    vllm_importable: bool,
    llm_importable: bool,
    cache_modules: list[str],
    object_attrs: list[str],
    generation_blocked_by_oom: bool,
    generation_blocked_by_server: bool,
    recon_exception: str = "",
) -> str:
    """Conservative KV/cache visibility classification."""
    if not vllm_importable:
        return "not_importable"
    if recon_exception:
        return "blocked_by_exception"
    if generation_blocked_by_server and not object_attrs:
        return "blocked_by_running_server"
    if generation_blocked_by_oom and not object_attrs:
        return "blocked_by_oom"
    if object_attrs:
        public_cache = any(
            name in cache_modules
            for name in ("CacheEngine", "CacheConfig", "KVCache", "PagedAttention")
        )
        if public_cache:
            return "public_api_visible"
        return "object_attrs_visible_private_only"
    if cache_modules:
        return "module_names_visible"
    if llm_importable:
        return "import_only_unknown"
    return "not_importable"


def build_possible_adapter_path(
    *,
    kv_status: str,
    cache_surfaces: list[str],
    object_attrs: list[str],
    running_server: bool,
    oom_blocked: bool,
) -> str:
    """Conservative adapter-path note — not an integration claim."""
    if kv_status == "not_importable":
        return "blocked — vLLM not importable; no adapter path at recon time"
    if oom_blocked or running_server:
        parts = []
        if running_server:
            parts.append("blocked by running server")
        if oom_blocked:
            parts.append("blocked by OOM")
        return (
            "potential adapter path requires idle GPU prototype validation; "
            + "; ".join(parts)
        )
    if kv_status == "public_api_visible":
        return (
            "potential adapter path — cache-related public symbols visible; "
            "compressed-draft + restored-verifier wiring requires prototype validation"
        )
    if kv_status == "module_names_visible":
        return (
            "potential adapter path — cache module names visible at import level; "
            "authoritative full-KV export for restored verifier not verified"
        )
    if kv_status == "object_attrs_visible_private_only":
        attrs = ", ".join(object_attrs[:5]) if object_attrs else "cache-like attrs"
        return (
            f"visible private attributes ({attrs}) require prototype validation; "
            "not treated as stable APIs"
        )
    if cache_surfaces:
        return "cache access remains unknown — module hints only; prototype validation required"
    return "cache access remains unknown — import-only recon; no stable KV export path identified"


def _attempt_llm_object_recon(
    *,
    model_id: str = DEFAULT_SMOKE_MODEL,
    prompt: str = DEFAULT_SMOKE_PROMPT,
    max_tokens: int = DEFAULT_SMOKE_MAX_TOKENS,
) -> tuple[bool, bool, bool, str, str, list[str]]:
    """Optional LLM init + tiny smoke + safe attr recon. Returns init, attempted, passed, err, preview, attrs."""
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        return False, False, False, f"LLM import failed: {exc}", "", []

    try:
        llm = LLM(model=model_id, dtype="float16", max_model_len=256)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        if "memory" in err.lower() or "oom" in err.lower():
            return False, False, False, err, "", []
        return False, False, False, err, "", []

    attrs = _safe_attr_names(llm)
    for nested in ("llm_engine", "engine"):
        if hasattr(llm, nested):
            try:
                attrs.extend(_safe_attr_names(getattr(llm, nested)))
            except Exception:  # noqa: BLE001
                pass
    attrs = sorted(set(attrs))[:40]

    try:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)
        outputs = llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return True, True, False, "generation returned empty outputs", "", attrs
        text = outputs[0].outputs[0].text
        preview = text.strip()[:200]
        return True, True, True, "", preview, attrs
    except Exception as exc:  # noqa: BLE001
        return True, True, False, f"{type(exc).__name__}: {exc}", "", attrs


def run_vllm_surface_recon(
    *,
    environment_note: str = "",
    allow_llm_init: bool = False,
    stop_template_server: bool = False,
    smoke_model_id: str = DEFAULT_SMOKE_MODEL,
) -> VllmSurfaceReconResult:
    """Run Exp 063 vLLM API surface reconnaissance."""
    blockers: list[str] = []
    preflight = preflight_gpu_check(stop_server=stop_template_server)
    if preflight.stopped_processes:
        preflight = preflight_gpu_check(stop_server=False)

    torch_version, cuda_available, gpu_name, cuda_runtime = _torch_environment()
    if not gpu_name and preflight.raw_nvidia_smi:
        first = preflight.raw_nvidia_smi.splitlines()[0] if preflight.raw_nvidia_smi else ""
        gpu_name = first.split(",")[0].strip() if "," in first else gpu_name

    vllm_importable = _module_importable("vllm")
    vllm_version = ""
    llm_importable = False
    sampling_importable = False

    if not vllm_importable:
        blockers.append("ModuleNotFoundError: No module named 'vllm'")
        kv_status = classify_kv_cache_access_status(
            vllm_importable=False,
            llm_importable=False,
            cache_modules=[],
            object_attrs=[],
            generation_blocked_by_oom=False,
            generation_blocked_by_server=False,
        )
        return VllmSurfaceReconResult(
            status="blocked",
            environment_note=environment_note,
            python_executable=sys.executable,
            python_version=platform.python_version(),
            platform_info=platform.platform(),
            torch_version=torch_version,
            cuda_available=cuda_available,
            cuda_runtime=cuda_runtime,
            gpu_name=gpu_name,
            gpu_memory_summary=preflight.gpu_memory_summary,
            running_server_detected=preflight.running_server_detected,
            stopped_processes=preflight.stopped_processes,
            vllm_importable=False,
            vllm_version="",
            llm_class_importable=False,
            sampling_params_importable=False,
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            generation_smoke_error="",
            generated_text_preview="",
            llm_object_initialized=False,
            visible_top_level_modules=[],
            visible_config_surfaces=[],
            visible_engine_surfaces=[],
            visible_scheduler_surfaces=[],
            visible_cache_surfaces=[],
            object_level_attr_names=[],
            kv_cache_access_status=kv_status,
            possible_adapter_path=build_possible_adapter_path(
                kv_status=kv_status,
                cache_surfaces=[],
                object_attrs=[],
                running_server=preflight.running_server_detected,
                oom_blocked=False,
            ),
            blockers=blockers,
        )

    try:
        vllm_mod = importlib.import_module("vllm")
        vllm_version = str(getattr(vllm_mod, "__version__", "") or "")
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"vllm import failed: {exc}")

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

    top_modules = _discover_top_level_modules()
    config_surfaces = _discover_config_surfaces()
    engine_surfaces = _discover_engine_surfaces()
    scheduler_surfaces = _discover_scheduler_surfaces()
    cache_surfaces = _discover_cache_module_surfaces()

    enough_mem = _has_enough_gpu_memory(preflight.free_gib)
    should_init = allow_llm_init or (enough_mem and not preflight.running_server_detected)

    gen_attempted = False
    gen_passed = False
    gen_error = ""
    gen_preview = ""
    llm_initialized = False
    object_attrs: list[str] = []
    oom_blocked = False
    server_blocked = preflight.running_server_detected

    if not llm_importable or not sampling_importable:
        pass
    elif not should_init:
        if preflight.running_server_detected:
            blockers.append("GPU busy — vLLM/OpenAI server detected; skipped LLM object init")
        elif not enough_mem:
            oom_blocked = True
            blockers.append(
                f"GPU memory low ({preflight.gpu_memory_summary}); skipped LLM object init"
            )
    else:
        llm_initialized, gen_attempted, gen_passed, gen_error, gen_preview, object_attrs = (
            _attempt_llm_object_recon(model_id=smoke_model_id)
        )
        if gen_attempted and not gen_passed and gen_error:
            blockers.append(f"generation_smoke: {gen_error}")
            if "memory" in gen_error.lower():
                oom_blocked = True
                llm_initialized = False

    kv_status = classify_kv_cache_access_status(
        vllm_importable=vllm_importable,
        llm_importable=llm_importable,
        cache_modules=cache_surfaces,
        object_attrs=object_attrs,
        generation_blocked_by_oom=oom_blocked,
        generation_blocked_by_server=server_blocked and not llm_initialized,
    )

    adapter_path = build_possible_adapter_path(
        kv_status=kv_status,
        cache_surfaces=cache_surfaces,
        object_attrs=object_attrs,
        running_server=server_blocked,
        oom_blocked=oom_blocked,
    )

    if not vllm_importable or not llm_importable or not sampling_importable:
        status = "blocked"
    elif gen_attempted and not gen_passed:
        status = "failed"
    else:
        status = "pass"

    return VllmSurfaceReconResult(
        status=status,
        environment_note=environment_note,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        platform_info=platform.platform(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_runtime=cuda_runtime,
        gpu_name=gpu_name,
        gpu_memory_summary=preflight.gpu_memory_summary,
        running_server_detected=preflight.running_server_detected,
        stopped_processes=preflight.stopped_processes,
        vllm_importable=vllm_importable,
        vllm_version=vllm_version,
        llm_class_importable=llm_importable,
        sampling_params_importable=sampling_importable,
        generation_smoke_attempted=gen_attempted,
        generation_smoke_passed=gen_passed,
        generation_smoke_error=gen_error,
        generated_text_preview=gen_preview,
        llm_object_initialized=llm_initialized,
        visible_top_level_modules=top_modules,
        visible_config_surfaces=config_surfaces,
        visible_engine_surfaces=engine_surfaces,
        visible_scheduler_surfaces=scheduler_surfaces,
        visible_cache_surfaces=cache_surfaces,
        object_level_attr_names=object_attrs,
        kv_cache_access_status=kv_status,
        possible_adapter_path=adapter_path,
        blockers=blockers,
    )


def validate_exp063_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 063 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "environment_note",
        "python_executable",
        "python_version",
        "platform",
        "torch_version",
        "cuda_available",
        "gpu_name",
        "gpu_memory_summary",
        "running_server_detected",
        "stopped_processes",
        "vllm_importable",
        "vllm_version",
        "llm_class_importable",
        "sampling_params_importable",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "generation_smoke_error",
        "llm_object_initialized",
        "visible_top_level_modules",
        "visible_config_surfaces",
        "visible_engine_surfaces",
        "visible_scheduler_surfaces",
        "visible_cache_surfaces",
        "kv_cache_access_status",
        "possible_adapter_path",
        "blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_063_ID:
        errors.append("experiment_id must be exp063_vllm_api_surface_recon")
    if report.get("status") not in ("pass", "blocked", "failed"):
        errors.append("status must be pass, blocked, or failed")
    if report.get("kv_cache_access_status") not in KV_CACHE_ACCESS_STATUSES:
        errors.append(f"invalid kv_cache_access_status: {report.get('kv_cache_access_status')}")
    for flag in (
        "cuda_available",
        "running_server_detected",
        "vllm_importable",
        "llm_class_importable",
        "sampling_params_importable",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "llm_object_initialized",
    ):
        if not isinstance(report.get(flag), bool):
            errors.append(f"{flag} must be a bool")
    for key in (
        "visible_top_level_modules",
        "visible_config_surfaces",
        "visible_engine_surfaces",
        "visible_scheduler_surfaces",
        "visible_cache_surfaces",
        "stopped_processes",
        "blockers",
    ):
        if not isinstance(report.get(key), list):
            errors.append(f"{key} must be a list")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    path = str(report.get("possible_adapter_path", "")).lower()
    for forbidden_phrase in (
        "vllm integration works",
        "exactkv supports vllm",
        "throughput improvement",
        "memory savings",
    ):
        if forbidden_phrase in path:
            errors.append(f"possible_adapter_path must not claim: {forbidden_phrase}")
    return errors


def recommend_next_step_after_surface_recon(*, status: str, kv_status: str) -> str:
    if status == "blocked":
        return "Resolve vLLM import or GPU blockers before adapter prototype design"
    if kv_status in ("object_attrs_visible_private_only", "module_names_visible", "public_api_visible"):
        return (
            "Phase 15D: isolated KV export prototype spike in vLLM image — "
            "still no ExactKV default-runtime integration"
        )
    return "Phase 15D: deeper KV export prototype with idle GPU — still no integration claim"
