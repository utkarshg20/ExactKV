"""vLLM KV/cache visibility prototype (Phase 15D).

Metadata-only object-level inspection after tiny LLM init. **Does not** wire ExactKV
into vLLM or export raw KV tensors via private APIs.

This is a vLLM KV/cache visibility probe, not ExactKV-vLLM integration.
"""
from __future__ import annotations

import importlib
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from exactkv.integrations.vllm_probe import (
    DEFAULT_SMOKE_MAX_TOKENS,
    DEFAULT_SMOKE_MODEL,
    DEFAULT_SMOKE_PROMPT,
    FORBIDDEN_CLAIMS,
)
from exactkv.integrations.vllm_surface_recon import (
    _has_enough_gpu_memory,
    preflight_gpu_check,
)

EXPERIMENT_064_ID = "exp064_vllm_kv_visibility_probe"
DEFAULT_EXP064_REPORT = Path("reports/experiment_064_vllm_kv_visibility_probe.json")

EXP064_CLAIM_NOTE = (
    "vLLM KV/cache visibility probe (Phase 15D). Metadata-only object inspection "
    "after optional tiny LLM init — no ExactKV vLLM integration, serving, batching, "
    "or performance claims. Private vLLM attributes are not stable APIs. Raw KV export "
    "is not claimed unless a clear public API is found. ExactKV default runtime "
    "unchanged."
)

KV_VISIBILITY_STATUSES = (
    "blocked_by_running_server",
    "blocked_by_oom",
    "llm_init_only",
    "engine_visible",
    "cache_config_visible",
    "private_cache_attrs_visible",
    "public_kv_export_visible",
    "metadata_only_probe_success",
    "raw_kv_export_not_available",
)

RAW_KV_EXPORT_STATUSES = (
    "not_probed",
    "public_api_candidate_found",
    "raw_kv_export_not_available",
    "blocked_by_oom",
    "blocked_by_running_server",
)

_ATTR_KEYWORDS = (
    "cache",
    "kv",
    "block",
    "paged",
    "scheduler",
    "engine",
    "executor",
    "gpu",
)

_PUBLIC_KV_METHOD_HINTS = (
    "get_kv_cache",
    "export_kv",
    "get_cache",
    "kv_cache",
    "cache_engine",
)

_INSPECT_MAX_DEPTH = 2
_INSPECT_MAX_ATTRS = 200


@dataclass
class BoundedInspectResult:
    """Categorized attribute names from bounded recursive inspection."""

    llm_attrs: list[str] = field(default_factory=list)
    engine_attrs: list[str] = field(default_factory=list)
    executor_attrs: list[str] = field(default_factory=list)
    scheduler_attrs: list[str] = field(default_factory=list)
    cache_attrs: list[str] = field(default_factory=list)
    block_attrs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def all_cache_related(self) -> list[str]:
        return sorted(set(self.cache_attrs + self.block_attrs))


@dataclass
class VllmKvVisibilityResult:
    """Result of Exp 064 KV/cache visibility probe."""

    status: str
    environment_note: str
    python_executable: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_runtime: str
    gpu_name: str
    gpu_memory_before: str
    gpu_memory_after: str
    running_server_detected: bool
    stopped_processes: list[str]
    vllm_version: str
    llm_object_initialized: bool
    generation_smoke_attempted: bool
    generation_smoke_passed: bool
    generation_smoke_error: str
    generated_text_preview: str
    visible_llm_attrs: list[str]
    visible_engine_attrs: list[str]
    visible_scheduler_attrs: list[str]
    visible_cache_attrs: list[str]
    visible_block_attrs: list[str]
    cache_config_summary: str
    kv_cache_visibility_status: str
    raw_kv_export_status: str
    possible_adapter_path: str
    blockers: list[str] = field(default_factory=list)
    claim_note: str = EXP064_CLAIM_NOTE
    forbidden_claims: list[str] = field(default_factory=lambda: list(FORBIDDEN_CLAIMS))

    def to_report_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_id"] = EXPERIMENT_064_ID
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


def _matches_keyword(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in _ATTR_KEYWORDS)


def _categorize_attr(name: str) -> str | None:
    lower = name.lower()
    if "block" in lower:
        return "block"
    if "cache" in lower or "kv" in lower or "paged" in lower:
        return "cache"
    if "scheduler" in lower:
        return "scheduler"
    if "executor" in lower:
        return "executor"
    if "engine" in lower:
        return "engine"
    if _matches_keyword(name):
        return "llm"
    return None


def _safe_getattr(obj: Any, name: str) -> Any | None:
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001
        return None


def _is_safe_child(value: Any) -> bool:
    if value is None:
        return False
    if callable(value):
        return False
    mod = type(value).__module__ or ""
    if mod.startswith("torch") and hasattr(value, "shape"):
        return False
    return True


def bounded_inspect_surfaces(
    root: Any,
    *,
    max_depth: int = _INSPECT_MAX_DEPTH,
    max_attrs: int = _INSPECT_MAX_ATTRS,
) -> BoundedInspectResult:
    """Bounded recursive metadata-only attribute inspection."""
    result = BoundedInspectResult()
    seen: set[int] = set()
    attr_count = 0

    def walk(obj: Any, depth: int, prefix: str) -> None:
        nonlocal attr_count
        if depth > max_depth or attr_count >= max_attrs:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        try:
            names = dir(obj)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"dir failed at {prefix}: {exc}")
            return

        for name in names:
            if attr_count >= max_attrs:
                break
            if name.startswith("__"):
                continue
            if not _matches_keyword(name):
                continue
            attr_count += 1
            cat = _categorize_attr(name)
            qualified = f"{prefix}.{name}" if prefix else name
            if cat == "block":
                result.block_attrs.append(qualified)
            elif cat == "cache":
                result.cache_attrs.append(qualified)
            elif cat == "scheduler":
                result.scheduler_attrs.append(qualified)
            elif cat == "executor":
                result.executor_attrs.append(qualified)
            elif cat == "engine":
                result.engine_attrs.append(qualified)
            else:
                result.llm_attrs.append(qualified)

            if depth < max_depth:
                child = _safe_getattr(obj, name)
                if _is_safe_child(child):
                    walk(child, depth + 1, qualified)

    try:
        walk(root, 0, "")
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"inspect root failed: {exc}")

    for key in (
        "llm_attrs",
        "engine_attrs",
        "executor_attrs",
        "scheduler_attrs",
        "cache_attrs",
        "block_attrs",
    ):
        setattr(result, key, sorted(set(getattr(result, key)))[:max_attrs])
    return result


def _summarize_cache_config(llm: Any) -> str:
    """Metadata-only cache config summary without tensor access."""
    candidates = (
        ("llm_engine", "cache_config"),
        ("llm_engine", "model_config"),
        ("engine", "cache_config"),
        ("cache_config",),
    )
    for path in candidates:
        obj: Any = llm
        for part in path:
            obj = _safe_getattr(obj, part)
            if obj is None:
                break
        if obj is None:
            continue
        parts: list[str] = []
        for field_name in (
            "block_size",
            "num_gpu_blocks",
            "num_cpu_blocks",
            "cache_dtype",
            "enable_prefix_caching",
            "gpu_memory_utilization",
        ):
            val = _safe_getattr(obj, field_name)
            if val is not None and not callable(val):
                if hasattr(val, "shape"):
                    continue
                parts.append(f"{field_name}={val!r}")
        if parts:
            return "; ".join(parts[:8])
        try:
            return f"{type(obj).__name__} visible"
        except Exception:  # noqa: BLE001
            pass
    return ""


def _detect_public_kv_export(llm: Any, inspect_result: BoundedInspectResult) -> str:
    """Check for obvious public KV export hooks without calling them."""
    for name in _PUBLIC_KV_METHOD_HINTS:
        if _safe_getattr(llm, name) is not None:
            return "public_api_candidate_found"
    for path in inspect_result.engine_attrs + inspect_result.cache_attrs:
        base = path.split(".")[-1]
        if base in _PUBLIC_KV_METHOD_HINTS:
            return "public_api_candidate_found"
    return "raw_kv_export_not_available"


def classify_kv_visibility_status(
    *,
    llm_initialized: bool,
    running_server: bool,
    oom_blocked: bool,
    inspect_result: BoundedInspectResult,
    cache_config_summary: str,
    raw_kv_status: str,
) -> str:
    """Conservative KV/cache visibility classification."""
    if running_server and not llm_initialized and not oom_blocked:
        return "blocked_by_running_server"
    if oom_blocked and not llm_initialized:
        return "blocked_by_oom"
    if not llm_initialized:
        return "blocked_by_oom" if oom_blocked else "blocked_by_running_server"
    if raw_kv_status == "public_api_candidate_found":
        return "public_kv_export_visible"
    if inspect_result.cache_attrs or inspect_result.block_attrs:
        if inspect_result.engine_attrs:
            return "metadata_only_probe_success"
        return "private_cache_attrs_visible"
    if cache_config_summary:
        return "cache_config_visible"
    if inspect_result.engine_attrs:
        return "engine_visible"
    return "llm_init_only"


def build_possible_adapter_path(
    *,
    kv_status: str,
    raw_kv_status: str,
    cache_config_summary: str,
    inspect_result: BoundedInspectResult,
) -> str:
    """Conservative adapter hypothesis — not an integration claim."""
    if kv_status == "blocked_by_running_server":
        return (
            "adapter blocked pending stable hook — running server occupied GPU; "
            "stop server and re-probe on idle GPU"
        )
    if kv_status == "blocked_by_oom":
        return "adapter blocked pending idle GPU — OOM during LLM init or smoke"
    if raw_kv_status == "public_api_candidate_found":
        return (
            "potential adapter path — public KV hook name visible; "
            "prototype validation required before any ExactKV wiring"
        )
    if kv_status in ("metadata_only_probe_success", "private_cache_attrs_visible"):
        attrs = ", ".join(inspect_result.cache_attrs[:4]) or "cache-like attrs"
        return (
            f"potential adapter path — cache metadata visible ({attrs}); "
            "private attrs require validation; raw KV export not available"
        )
    if cache_config_summary:
        return (
            f"potential adapter path — cache config metadata visible ({cache_config_summary}); "
            "raw KV export not available"
        )
    if kv_status == "engine_visible":
        return (
            "potential adapter path — engine surfaces visible; "
            "cache metadata probe incomplete; adapter blocked pending stable hook"
        )
    return "raw KV export not available — adapter blocked pending stable hook"


def _init_llm_and_smoke(
    *,
    model_id: str = DEFAULT_SMOKE_MODEL,
    prompt: str = DEFAULT_SMOKE_PROMPT,
    max_tokens: int = DEFAULT_SMOKE_MAX_TOKENS,
) -> tuple[Any | None, bool, bool, bool, str, str]:
    """Initialize LLM, run tiny smoke, return llm, init, attempted, passed, error, preview."""
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        return None, False, False, False, f"LLM import failed: {exc}", ""

    try:
        llm = LLM(
            model=model_id,
            dtype="float16",
            max_model_len=256,
            gpu_memory_utilization=0.35,
        )
    except Exception as exc:  # noqa: BLE001
        return None, False, False, False, f"{type(exc).__name__}: {exc}", ""

    try:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)
        outputs = llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return llm, True, True, False, "generation returned empty outputs", ""
        text = outputs[0].outputs[0].text.strip()[:200]
        return llm, True, True, True, "", text
    except Exception as exc:  # noqa: BLE001
        return llm, True, True, False, f"{type(exc).__name__}: {exc}", ""


def run_vllm_kv_visibility_probe(
    *,
    environment_note: str = "",
    stop_template_server: bool = False,
    allow_llm_init: bool = False,
    smoke_model_id: str = DEFAULT_SMOKE_MODEL,
) -> VllmKvVisibilityResult:
    """Run Exp 064 metadata-only KV/cache visibility probe."""
    blockers: list[str] = []
    pre_before = preflight_gpu_check(stop_server=False)
    gpu_before = pre_before.gpu_memory_summary

    stopped: list[str] = []
    if stop_template_server:
        stopped_preflight = preflight_gpu_check(stop_server=True)
        stopped = stopped_preflight.stopped_processes
        if stopped:
            time.sleep(10)
        pre_before = preflight_gpu_check(stop_server=False)

    pre_after = pre_before
    if stopped:
        pre_after = preflight_gpu_check(stop_server=False)

    torch_version, cuda_available, gpu_name, cuda_runtime = _torch_environment()
    if not gpu_name and pre_after.raw_nvidia_smi:
        first = pre_after.raw_nvidia_smi.splitlines()[0] if pre_after.raw_nvidia_smi else ""
        gpu_name = first.split(",")[0].strip() if "," in first else gpu_name

    vllm_version = ""
    try:
        vllm_mod = importlib.import_module("vllm")
        vllm_version = str(getattr(vllm_mod, "__version__", "") or "")
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"vllm import failed: {exc}")

    running_server = pre_after.running_server_detected
    enough_mem = _has_enough_gpu_memory(pre_after.free_gib)
    should_init = allow_llm_init or (enough_mem and not running_server)

    llm_obj: Any | None = None
    llm_initialized = False
    gen_attempted = False
    gen_passed = False
    gen_error = ""
    gen_preview = ""
    oom_blocked = False

    inspect_result = BoundedInspectResult()
    cache_config_summary = ""

    if not should_init:
        if running_server:
            blockers.append("GPU busy — vLLM/OpenAI server detected; skipped LLM object init")
        else:
            oom_blocked = True
            blockers.append(
                f"GPU memory low ({pre_after.gpu_memory_summary}); skipped LLM object init"
            )
    else:
        llm_obj, llm_initialized, gen_attempted, gen_passed, gen_error, gen_preview = (
            _init_llm_and_smoke(model_id=smoke_model_id)
        )
        if gen_error:
            blockers.append(gen_error if not llm_initialized else f"generation_smoke: {gen_error}")
            if (
                "memory" in gen_error.lower()
                or "oom" in gen_error.lower()
                or "free memory" in gen_error.lower()
            ):
                oom_blocked = True
                llm_initialized = False
            elif not llm_initialized and running_server:
                if "free memory" in gen_error.lower():
                    oom_blocked = True
                else:
                    blockers.append("LLM init failed while vLLM server process still detected")

        if llm_obj is not None and llm_initialized:
            inspect_result = bounded_inspect_surfaces(llm_obj)
            for err in inspect_result.errors:
                blockers.append(f"inspect: {err}")
            cache_config_summary = _summarize_cache_config(llm_obj)
            for nested in ("llm_engine", "engine"):
                nested_obj = _safe_getattr(llm_obj, nested)
                if nested_obj is not None:
                    nested_inspect = bounded_inspect_surfaces(nested_obj)
                    inspect_result.engine_attrs = sorted(
                        set(inspect_result.engine_attrs + nested_inspect.engine_attrs)
                    )[:_INSPECT_MAX_ATTRS]
                    inspect_result.cache_attrs = sorted(
                        set(inspect_result.cache_attrs + nested_inspect.cache_attrs)
                    )[:_INSPECT_MAX_ATTRS]
                    inspect_result.scheduler_attrs = sorted(
                        set(inspect_result.scheduler_attrs + nested_inspect.scheduler_attrs)
                    )[:_INSPECT_MAX_ATTRS]
                    inspect_result.executor_attrs = sorted(
                        set(inspect_result.executor_attrs + nested_inspect.executor_attrs)
                    )[:_INSPECT_MAX_ATTRS]
                    inspect_result.block_attrs = sorted(
                        set(inspect_result.block_attrs + nested_inspect.block_attrs)
                    )[:_INSPECT_MAX_ATTRS]

    raw_kv_status: str = "not_probed"
    if llm_initialized and llm_obj is not None:
        raw_kv_status = _detect_public_kv_export(llm_obj, inspect_result)
    elif oom_blocked:
        raw_kv_status = "blocked_by_oom"
    elif running_server:
        raw_kv_status = "blocked_by_running_server"

    kv_status = classify_kv_visibility_status(
        llm_initialized=llm_initialized,
        running_server=running_server and not stopped,
        oom_blocked=oom_blocked,
        inspect_result=inspect_result,
        cache_config_summary=cache_config_summary,
        raw_kv_status=raw_kv_status,
    )

    adapter_path = build_possible_adapter_path(
        kv_status=kv_status,
        raw_kv_status=raw_kv_status,
        cache_config_summary=cache_config_summary,
        inspect_result=inspect_result,
    )

    if not llm_initialized:
        status = "blocked"
    elif gen_attempted and not gen_passed:
        status = "failed"
    else:
        status = "pass"

    return VllmKvVisibilityResult(
        status=status,
        environment_note=environment_note,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_runtime=cuda_runtime,
        gpu_name=gpu_name,
        gpu_memory_before=gpu_before,
        gpu_memory_after=pre_after.gpu_memory_summary,
        running_server_detected=pre_after.running_server_detected,
        stopped_processes=stopped,
        vllm_version=vllm_version,
        llm_object_initialized=llm_initialized,
        generation_smoke_attempted=gen_attempted,
        generation_smoke_passed=gen_passed,
        generation_smoke_error=gen_error,
        generated_text_preview=gen_preview,
        visible_llm_attrs=inspect_result.llm_attrs,
        visible_engine_attrs=inspect_result.engine_attrs,
        visible_scheduler_attrs=inspect_result.scheduler_attrs,
        visible_cache_attrs=inspect_result.cache_attrs,
        visible_block_attrs=inspect_result.block_attrs,
        cache_config_summary=cache_config_summary,
        kv_cache_visibility_status=kv_status,
        raw_kv_export_status=raw_kv_status,
        possible_adapter_path=adapter_path,
        blockers=blockers,
    )


def validate_exp064_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 064 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "environment_note",
        "python_executable",
        "torch_version",
        "cuda_available",
        "gpu_name",
        "gpu_memory_before",
        "gpu_memory_after",
        "running_server_detected",
        "stopped_processes",
        "vllm_version",
        "llm_object_initialized",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "generated_text_preview",
        "visible_llm_attrs",
        "visible_engine_attrs",
        "visible_scheduler_attrs",
        "visible_cache_attrs",
        "visible_block_attrs",
        "cache_config_summary",
        "kv_cache_visibility_status",
        "raw_kv_export_status",
        "possible_adapter_path",
        "blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_064_ID:
        errors.append("experiment_id must be exp064_vllm_kv_visibility_probe")
    if report.get("status") not in ("pass", "blocked", "failed"):
        errors.append("status must be pass, blocked, or failed")
    if report.get("kv_cache_visibility_status") not in KV_VISIBILITY_STATUSES:
        errors.append(f"invalid kv_cache_visibility_status: {report.get('kv_cache_visibility_status')}")
    if report.get("raw_kv_export_status") not in RAW_KV_EXPORT_STATUSES:
        errors.append(f"invalid raw_kv_export_status: {report.get('raw_kv_export_status')}")
    for flag in (
        "cuda_available",
        "running_server_detected",
        "llm_object_initialized",
        "generation_smoke_attempted",
        "generation_smoke_passed",
    ):
        if not isinstance(report.get(flag), bool):
            errors.append(f"{flag} must be a bool")
    for key in (
        "visible_llm_attrs",
        "visible_engine_attrs",
        "visible_scheduler_attrs",
        "visible_cache_attrs",
        "visible_block_attrs",
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
    for bad in ("vllm integration works", "exactkv supports vllm", "throughput improvement"):
        if bad in path:
            errors.append(f"possible_adapter_path must not claim: {bad}")
    return errors


def recommend_next_step_after_kv_probe(*, status: str, kv_status: str) -> str:
    if status == "blocked":
        return (
            "Phase 15E: idle-GPU object-level probe on vLLM CUDA-13 image without "
            "auto-started server — see EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md"
        )
    if kv_status in ("metadata_only_probe_success", "private_cache_attrs_visible", "cache_config_visible"):
        return (
            "Phase 15F: isolated KV export prototype spike with explicit private-API "
            "validation — still no ExactKV default-runtime integration"
        )
    return (
        "Phase 15E: idle-GPU object-level probe — still no integration claim; "
        "see EXPERIMENT_065_IDLE_VLLM_OBJECT_KV_PROBE.md"
    )


# --- Experiment 065: idle-GPU object-level KV/cache probe (Phase 15E) ---

EXPERIMENT_065_ID = "exp065_idle_vllm_object_kv_probe"
DEFAULT_EXP065_REPORT = Path("reports/experiment_065_idle_vllm_object_kv_probe.json")

EXP065_CLAIM_NOTE = (
    "Idle-GPU vLLM object-level probe (Phase 15E). Metadata-only inspection after "
    "tiny LLM init on idle GPU — no ExactKV vLLM integration, serving, batching, or "
    "performance claims. Private vLLM attributes are not stable APIs. Raw KV export "
    "is not claimed unless a clear public API is found. ExactKV default runtime unchanged."
)

EXP065_KV_VISIBILITY_STATUSES = (
    "blocked_by_running_server",
    "blocked_by_oom",
    "llm_init_failed",
    "llm_init_success_no_cache_surface",
    "cache_config_visible",
    "private_cache_attrs_visible",
    "engine_cache_metadata_visible",
    "public_kv_export_visible",
    "raw_kv_export_not_available",
)


@dataclass
class IdleVllmObjectKvProbeResult:
    """Result of Exp 065 idle-GPU object-level KV/cache probe."""

    status: str
    environment_note: str
    python_executable: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_runtime: str
    gpu_name: str
    gpu_memory_before: str
    gpu_memory_after: str
    running_server_detected: bool
    stopped_processes: list[str]
    vllm_version: str
    llm_object_initialized: bool
    generation_smoke_attempted: bool
    generation_smoke_passed: bool
    generation_smoke_error: str
    generated_text_preview: str
    visible_llm_attrs: list[str]
    visible_engine_attrs: list[str]
    visible_model_executor_attrs: list[str]
    visible_scheduler_attrs: list[str]
    visible_cache_attrs: list[str]
    visible_block_attrs: list[str]
    cache_config_summary: str
    kv_cache_visibility_status: str
    raw_kv_export_status: str
    possible_adapter_path: str
    blockers: list[str] = field(default_factory=list)
    claim_note: str = EXP065_CLAIM_NOTE
    forbidden_claims: list[str] = field(default_factory=lambda: list(FORBIDDEN_CLAIMS))

    def to_report_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_id"] = EXPERIMENT_065_ID
        data["forbidden_claims"] = list(self.forbidden_claims)
        return data


def _inspect_llm_surfaces(llm_obj: Any) -> tuple[BoundedInspectResult, str, list[str]]:
    """Run bounded inspection and cache-config summary; return inspect, summary, errors."""
    inspect_result = bounded_inspect_surfaces(llm_obj)
    errors = list(inspect_result.errors)
    cache_config_summary = _summarize_cache_config(llm_obj)
    for nested in ("llm_engine", "engine"):
        nested_obj = _safe_getattr(llm_obj, nested)
        if nested_obj is None:
            continue
        nested_inspect = bounded_inspect_surfaces(nested_obj)
        errors.extend(nested_inspect.errors)
        for key in (
            "engine_attrs",
            "executor_attrs",
            "cache_attrs",
            "scheduler_attrs",
            "block_attrs",
        ):
            merged = sorted(set(getattr(inspect_result, key) + getattr(nested_inspect, key)))
            setattr(inspect_result, key, merged[:_INSPECT_MAX_ATTRS])
    me = _safe_getattr(llm_obj, "model_executor")
    if me is None and _safe_getattr(llm_obj, "llm_engine") is not None:
        me = _safe_getattr(_safe_getattr(llm_obj, "llm_engine"), "model_executor")
    if me is not None:
        me_inspect = bounded_inspect_surfaces(me)
        errors.extend(me_inspect.errors)
        inspect_result.executor_attrs = sorted(
            set(inspect_result.executor_attrs + me_inspect.executor_attrs + me_inspect.cache_attrs)
        )[:_INSPECT_MAX_ATTRS]
    return inspect_result, cache_config_summary, errors


def classify_exp065_kv_visibility_status(
    *,
    llm_initialized: bool,
    init_attempted: bool,
    running_server: bool,
    oom_blocked: bool,
    init_failed: bool,
    inspect_result: BoundedInspectResult,
    cache_config_summary: str,
    raw_kv_status: str,
) -> str:
    """Conservative Exp 065 KV/cache visibility classification."""
    if running_server and not llm_initialized:
        return "blocked_by_running_server"
    if oom_blocked and not llm_initialized:
        return "blocked_by_oom"
    if init_failed or (init_attempted and not llm_initialized):
        return "llm_init_failed"
    if not llm_initialized:
        return "blocked_by_oom" if oom_blocked else "blocked_by_running_server"
    if raw_kv_status == "public_api_candidate_found":
        return "public_kv_export_visible"
    has_cache = bool(inspect_result.cache_attrs or inspect_result.block_attrs)
    has_engine = bool(inspect_result.engine_attrs or inspect_result.executor_attrs)
    if has_engine and (has_cache or cache_config_summary):
        return "engine_cache_metadata_visible"
    if has_cache:
        return "private_cache_attrs_visible"
    if cache_config_summary:
        return "cache_config_visible"
    return "llm_init_success_no_cache_surface"


def build_exp065_possible_adapter_path(
    *,
    kv_status: str,
    raw_kv_status: str,
    cache_config_summary: str,
    inspect_result: BoundedInspectResult,
) -> str:
    """Conservative adapter hypothesis for Exp 065."""
    if kv_status == "blocked_by_running_server":
        return (
            "adapter blocked pending idle GPU — running server detected; "
            "use idle vLLM CUDA-13 image without auto-started serve"
        )
    if kv_status in ("blocked_by_oom", "llm_init_failed"):
        return "adapter blocked pending idle GPU — LLM init or smoke failed"
    if raw_kv_status == "public_api_candidate_found":
        return (
            "potential adapter path — public KV hook name visible; "
            "prototype validation required before any ExactKV wiring"
        )
    if kv_status == "engine_cache_metadata_visible":
        attrs = ", ".join((inspect_result.cache_attrs + inspect_result.engine_attrs)[:4])
        return (
            f"potential adapter path — engine/cache metadata visible ({attrs}); "
            "private attrs require validation; raw KV export not available"
        )
    if kv_status == "private_cache_attrs_visible":
        attrs = ", ".join(inspect_result.cache_attrs[:4]) or "cache-like attrs"
        return (
            f"potential adapter path — cache metadata visible ({attrs}); "
            "private attrs require validation; raw KV export not available"
        )
    if cache_config_summary:
        return (
            f"potential adapter path — cache config metadata visible ({cache_config_summary}); "
            "raw KV export not available"
        )
    if kv_status == "llm_init_success_no_cache_surface":
        return (
            "LLM init succeeded but cache surfaces not visible — "
            "adapter blocked pending stable hook"
        )
    return "raw KV export not available — adapter blocked pending stable hook"


def run_idle_vllm_object_kv_probe(
    *,
    environment_note: str = "",
    stop_server: bool = False,
    smoke_model_id: str = DEFAULT_SMOKE_MODEL,
) -> IdleVllmObjectKvProbeResult:
    """Run Exp 065 idle-GPU object-level KV/cache probe."""
    blockers: list[str] = []
    pre_before = preflight_gpu_check(stop_server=False)
    gpu_before = pre_before.gpu_memory_summary
    stopped: list[str] = []

    if stop_server:
        stop_preflight = preflight_gpu_check(stop_server=True)
        stopped = stop_preflight.stopped_processes
        if stopped:
            time.sleep(10)
        pre_before = preflight_gpu_check(stop_server=False)

    running_server = pre_before.running_server_detected
    if running_server and not stopped:
        blockers.append(
            "Running vLLM/OpenAI/model server detected — idle GPU required; "
            "not stopping server by default"
        )

    torch_version, cuda_available, gpu_name, cuda_runtime = _torch_environment()
    if not gpu_name and pre_before.raw_nvidia_smi:
        first = pre_before.raw_nvidia_smi.splitlines()[0] if pre_before.raw_nvidia_smi else ""
        gpu_name = first.split(",")[0].strip() if "," in first else gpu_name

    vllm_version = ""
    try:
        vllm_mod = importlib.import_module("vllm")
        vllm_version = str(getattr(vllm_mod, "__version__", "") or "")
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"vllm import failed: {exc}")

    llm_obj: Any | None = None
    llm_initialized = False
    gen_attempted = False
    gen_passed = False
    gen_error = ""
    gen_preview = ""
    oom_blocked = False
    init_failed = False
    init_attempted = False

    inspect_result = BoundedInspectResult()
    cache_config_summary = ""
    gpu_after = pre_before.gpu_memory_summary

    if running_server and not stopped:
        pass
    elif not _has_enough_gpu_memory(pre_before.free_gib):
        oom_blocked = True
        blockers.append(
            f"GPU memory low ({pre_before.gpu_memory_summary}); need idle GPU with ~4+ GiB free"
        )
    else:
        init_attempted = True
        llm_obj, llm_initialized, gen_attempted, gen_passed, gen_error, gen_preview = (
            _init_llm_and_smoke(model_id=smoke_model_id)
        )
        post_preflight = preflight_gpu_check(stop_server=False)
        gpu_after = post_preflight.gpu_memory_summary

        if gen_error:
            blockers.append(gen_error if not llm_initialized else f"generation_smoke: {gen_error}")
            if (
                "memory" in gen_error.lower()
                or "oom" in gen_error.lower()
                or "free memory" in gen_error.lower()
            ):
                oom_blocked = True
                llm_initialized = False
            elif not llm_initialized:
                init_failed = True

        if llm_obj is not None and llm_initialized:
            inspect_result, cache_config_summary, inspect_errors = _inspect_llm_surfaces(llm_obj)
            for err in inspect_errors:
                blockers.append(f"inspect: {err}")

    raw_kv_status: str = "not_probed"
    if llm_initialized and llm_obj is not None:
        raw_kv_status = _detect_public_kv_export(llm_obj, inspect_result)
    elif oom_blocked:
        raw_kv_status = "blocked_by_oom"
    elif running_server and not stopped:
        raw_kv_status = "blocked_by_running_server"
    elif init_failed:
        raw_kv_status = "raw_kv_export_not_available"

    kv_status = classify_exp065_kv_visibility_status(
        llm_initialized=llm_initialized,
        init_attempted=init_attempted,
        running_server=running_server and not stopped,
        oom_blocked=oom_blocked,
        init_failed=init_failed,
        inspect_result=inspect_result,
        cache_config_summary=cache_config_summary,
        raw_kv_status=raw_kv_status,
    )

    adapter_path = build_exp065_possible_adapter_path(
        kv_status=kv_status,
        raw_kv_status=raw_kv_status,
        cache_config_summary=cache_config_summary,
        inspect_result=inspect_result,
    )

    if running_server and not stopped:
        status = "blocked"
    elif not llm_initialized:
        status = "blocked" if oom_blocked or init_failed or running_server else "failed"
    elif gen_attempted and not gen_passed:
        status = "failed"
    else:
        status = "pass"

    return IdleVllmObjectKvProbeResult(
        status=status,
        environment_note=environment_note,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_runtime=cuda_runtime,
        gpu_name=gpu_name,
        gpu_memory_before=gpu_before,
        gpu_memory_after=gpu_after,
        running_server_detected=running_server,
        stopped_processes=stopped,
        vllm_version=vllm_version,
        llm_object_initialized=llm_initialized,
        generation_smoke_attempted=gen_attempted,
        generation_smoke_passed=gen_passed,
        generation_smoke_error=gen_error,
        generated_text_preview=gen_preview,
        visible_llm_attrs=inspect_result.llm_attrs,
        visible_engine_attrs=inspect_result.engine_attrs,
        visible_model_executor_attrs=inspect_result.executor_attrs,
        visible_scheduler_attrs=inspect_result.scheduler_attrs,
        visible_cache_attrs=inspect_result.cache_attrs,
        visible_block_attrs=inspect_result.block_attrs,
        cache_config_summary=cache_config_summary,
        kv_cache_visibility_status=kv_status,
        raw_kv_export_status=raw_kv_status,
        possible_adapter_path=adapter_path,
        blockers=blockers,
    )


def validate_exp065_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 065 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "environment_note",
        "python_executable",
        "torch_version",
        "cuda_available",
        "gpu_name",
        "gpu_memory_before",
        "gpu_memory_after",
        "running_server_detected",
        "stopped_processes",
        "vllm_version",
        "llm_object_initialized",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "generated_text_preview",
        "visible_llm_attrs",
        "visible_engine_attrs",
        "visible_model_executor_attrs",
        "visible_scheduler_attrs",
        "visible_cache_attrs",
        "visible_block_attrs",
        "cache_config_summary",
        "kv_cache_visibility_status",
        "raw_kv_export_status",
        "possible_adapter_path",
        "blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_065_ID:
        errors.append("experiment_id must be exp065_idle_vllm_object_kv_probe")
    if report.get("status") not in ("pass", "blocked", "failed"):
        errors.append("status must be pass, blocked, or failed")
    if report.get("kv_cache_visibility_status") not in EXP065_KV_VISIBILITY_STATUSES:
        errors.append(
            f"invalid kv_cache_visibility_status: {report.get('kv_cache_visibility_status')}"
        )
    if report.get("raw_kv_export_status") not in RAW_KV_EXPORT_STATUSES:
        errors.append(f"invalid raw_kv_export_status: {report.get('raw_kv_export_status')}")
    for flag in (
        "cuda_available",
        "running_server_detected",
        "llm_object_initialized",
        "generation_smoke_attempted",
        "generation_smoke_passed",
    ):
        if not isinstance(report.get(flag), bool):
            errors.append(f"{flag} must be a bool")
    for key in (
        "visible_llm_attrs",
        "visible_engine_attrs",
        "visible_model_executor_attrs",
        "visible_scheduler_attrs",
        "visible_cache_attrs",
        "visible_block_attrs",
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
    for bad in ("vllm integration works", "exactkv supports vllm", "throughput improvement"):
        if bad in path:
            errors.append(f"possible_adapter_path must not claim: {bad}")
    return errors


def recommend_next_step_after_idle_probe(*, status: str, kv_status: str) -> str:
    if status == "blocked":
        return (
            "Use idle vLLM CUDA-13 pod without auto-started serve; re-run "
            "run_exp065_idle_vllm_object_kv_probe.py"
        )
    if kv_status in (
        "engine_cache_metadata_visible",
        "private_cache_attrs_visible",
        "cache_config_visible",
    ):
        return (
            "Phase 15F: isolated KV export prototype spike with explicit private-API "
            "validation — still no ExactKV default-runtime integration"
        )
    return (
        "Phase 15F: deeper KV metadata mapping or export spike — still no integration claim"
    )
