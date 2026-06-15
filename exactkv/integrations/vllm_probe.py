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

EXPERIMENT_060_ID = "exp060_vllm_venv_feasibility"
DEFAULT_EXP060_REPORT = Path("reports/experiment_060_vllm_venv_feasibility.json")
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VLLM_VENV_PATH = _REPO_ROOT / ".venv-vllm"
DEFAULT_VLLM_VENV_PYTHON = DEFAULT_VLLM_VENV_PATH / "bin" / "python"
SYSTEM_PYTHON = Path("/usr/bin/python3")

EXP060_CLAIM_NOTE = (
    "Isolated vLLM venv feasibility test (Phase 15B). vLLM installed only in a "
    "separate venv — not system Python. Environment availability check only; no "
    "ExactKV vLLM integration, serving, batching, or performance claims. Passing "
    "this phase means a vLLM environment is available for future integration work "
    "only. Default ExactKV generation behavior unchanged."
)


@dataclass
class PythonEnvMetadata:
    """Torch/CUDA metadata for one Python executable."""

    python_executable: str
    python_version: str
    torch_version: str
    cuda_available: bool
    gpu_name: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_python_env_metadata(python_executable: str | Path) -> PythonEnvMetadata:
    """Collect torch/CUDA metadata from an arbitrary Python via subprocess."""
    import json
    import subprocess

    code = (
        "import json,sys\n"
        "out={'python_executable':sys.executable,'python_version':sys.version.split()[0],"
        "'torch_version':'','cuda_available':False,'gpu_name':'','error':''}\n"
        "try:\n"
        " import torch\n"
        " out['torch_version']=torch.__version__\n"
        " out['cuda_available']=bool(torch.cuda.is_available())\n"
        " out['gpu_name']=torch.cuda.get_device_name(0) if out['cuda_available'] else ''\n"
        "except Exception as exc:\n"
        " out['error']=f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return PythonEnvMetadata(
            python_executable=str(python_executable),
            python_version="",
            torch_version="",
            cuda_available=False,
            gpu_name="",
            error=f"subprocess error: {exc}",
        )
    if proc.returncode != 0 or not proc.stdout.strip():
        err = proc.stderr.strip() or f"exit code {proc.returncode}"
        return PythonEnvMetadata(
            python_executable=str(python_executable),
            python_version="",
            torch_version="",
            cuda_available=False,
            gpu_name="",
            error=err,
        )
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return PythonEnvMetadata(
        python_executable=str(data.get("python_executable", python_executable)),
        python_version=str(data.get("python_version", "")),
        torch_version=str(data.get("torch_version", "")),
        cuda_available=bool(data.get("cuda_available")),
        gpu_name=str(data.get("gpu_name", "")),
        error=str(data.get("error", "")),
    )


def tail_text(text: str, *, max_lines: int = 40) -> str:
    """Return trailing lines for report capture."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


@dataclass
class VllmVenvProbeResult:
    """Result of Exp 060 isolated vLLM venv feasibility probe."""

    status: str
    system_python: str
    venv_python: str
    system_torch_version: str
    venv_torch_version: str
    system_cuda_available: bool
    venv_cuda_available: bool
    gpu_name: str
    vllm_importable: bool
    vllm_version: str
    install_attempted: bool
    install_success: bool
    import_error: str
    generation_smoke_attempted: bool
    generation_smoke_passed: bool
    generation_smoke_error: str
    stdout_tail: str
    stderr_tail: str
    blockers: list[str] = field(default_factory=list)
    claim_note: str = EXP060_CLAIM_NOTE
    forbidden_claims: list[str] = field(default_factory=lambda: list(FORBIDDEN_CLAIMS))

    def to_report_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_id"] = EXPERIMENT_060_ID
        data["forbidden_claims"] = list(self.forbidden_claims)
        return data


def probe_vllm_in_subprocess(
    python_executable: str | Path,
    *,
    run_generation_smoke: bool = True,
    smoke_model_id: str = DEFAULT_SMOKE_MODEL,
    cwd: Path | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Run vLLM probe inside a subprocess Python and return JSON + stdout/stderr."""
    import json
    import subprocess

    worker = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "root=Path.cwd()\n"
        "if str(root) not in sys.path: sys.path.insert(0,str(root))\n"
        "from exactkv.integrations.vllm_probe import probe_vllm_availability\n"
        f"result=probe_vllm_availability(run_generation_smoke={run_generation_smoke!r},"
        f" smoke_model_id={smoke_model_id!r})\n"
        "print(json.dumps(result.to_report_dict()))\n"
    )
    workdir = str(cwd or _REPO_ROOT)
    proc = subprocess.run(
        [str(python_executable), "-c", worker],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        cwd=workdir,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0 or not stdout.strip():
        return (
            {"status": "blocked", "import_error": stderr.strip() or f"exit {proc.returncode}"},
            stdout,
            stderr,
        )
    payload_line = stdout.strip().splitlines()[-1]
    return json.loads(payload_line), stdout, stderr


def run_vllm_venv_feasibility(
    *,
    venv_python: Path | None = None,
    system_python: Path | None = None,
    run_generation_smoke: bool = True,
    smoke_model_id: str = DEFAULT_SMOKE_MODEL,
) -> VllmVenvProbeResult:
    """Run Exp 060 comparing system Python baseline vs isolated vLLM venv."""
    venv_py = Path(venv_python or DEFAULT_VLLM_VENV_PYTHON)
    sys_py = Path(system_python or SYSTEM_PYTHON)
    blockers: list[str] = []

    system_meta = collect_python_env_metadata(sys_py)
    if system_meta.error:
        blockers.append(f"system_python: {system_meta.error}")

    install_attempted = venv_py.is_file()
    if not install_attempted:
        blockers.append(f"venv python not found: {venv_py}")
        return VllmVenvProbeResult(
            status="blocked",
            system_python=str(sys_py),
            venv_python=str(venv_py),
            system_torch_version=system_meta.torch_version,
            venv_torch_version="",
            system_cuda_available=system_meta.cuda_available,
            venv_cuda_available=False,
            gpu_name=system_meta.gpu_name,
            vllm_importable=False,
            vllm_version="",
            install_attempted=False,
            install_success=False,
            import_error="venv not created — run setup_vllm_venv_runpod.sh",
            generation_smoke_attempted=False,
            generation_smoke_passed=False,
            generation_smoke_error="",
            stdout_tail="",
            stderr_tail="",
            blockers=blockers,
        )

    venv_meta = collect_python_env_metadata(venv_py)
    if venv_meta.error:
        blockers.append(f"venv_python: {venv_meta.error}")
    if not venv_meta.torch_version:
        blockers.append("venv cannot import torch")
    if not venv_meta.cuda_available:
        blockers.append("CUDA unavailable inside vLLM venv")

    probe_payload, stdout, stderr = probe_vllm_in_subprocess(
        venv_py,
        run_generation_smoke=run_generation_smoke,
        smoke_model_id=smoke_model_id,
    )

    vllm_pkg_importable = bool(probe_payload.get("vllm_importable"))
    llm_importable = bool(probe_payload.get("llm_class_importable"))
    sampling_importable = bool(probe_payload.get("sampling_params_importable"))
    vllm_importable = vllm_pkg_importable and llm_importable and sampling_importable
    install_success = vllm_importable
    import_error = str(probe_payload.get("import_error", ""))
    if not vllm_importable and probe_payload.get("blockers"):
        for item in probe_payload["blockers"]:
            blocker = str(item)
            if blocker not in blockers:
                blockers.append(blocker)
            if not import_error:
                import_error = blocker
    elif not vllm_importable and import_error:
        blockers.append(import_error)

    gen_attempted = bool(probe_payload.get("generation_smoke_attempted"))
    gen_passed = bool(probe_payload.get("generation_smoke_passed"))
    gen_error = str(probe_payload.get("generation_smoke_error", ""))
    if gen_attempted and not gen_passed and gen_error:
        blockers.append(f"generation_smoke: {gen_error}")

    if not vllm_importable or not venv_meta.cuda_available:
        status = "blocked"
    elif gen_attempted and not gen_passed:
        status = "failed"
    elif vllm_importable and (not run_generation_smoke or gen_passed):
        status = "pass"
    else:
        status = "blocked"

    gpu_name = venv_meta.gpu_name or system_meta.gpu_name
    return VllmVenvProbeResult(
        status=status,
        system_python=str(sys_py),
        venv_python=str(venv_py),
        system_torch_version=system_meta.torch_version,
        venv_torch_version=venv_meta.torch_version,
        system_cuda_available=system_meta.cuda_available,
        venv_cuda_available=venv_meta.cuda_available,
        gpu_name=gpu_name,
        vllm_importable=vllm_importable,
        vllm_version=str(probe_payload.get("vllm_version", "")),
        install_attempted=True,
        install_success=install_success,
        import_error=import_error,
        generation_smoke_attempted=gen_attempted,
        generation_smoke_passed=gen_passed,
        generation_smoke_error=gen_error,
        stdout_tail=tail_text(stdout),
        stderr_tail=tail_text(stderr),
        blockers=blockers,
    )


def validate_exp060_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 060 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "system_python",
        "venv_python",
        "system_torch_version",
        "venv_torch_version",
        "system_cuda_available",
        "venv_cuda_available",
        "gpu_name",
        "vllm_importable",
        "vllm_version",
        "install_attempted",
        "install_success",
        "import_error",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "generation_smoke_error",
        "stdout_tail",
        "stderr_tail",
        "blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_060_ID:
        errors.append("experiment_id must be exp060_vllm_venv_feasibility")
    if report.get("status") not in ("pass", "blocked", "failed"):
        errors.append("status must be pass, blocked, or failed")
    for flag in ("install_attempted", "install_success", "vllm_importable"):
        if not isinstance(report.get(flag), bool):
            errors.append(f"{flag} must be a bool")
    for flag in ("system_cuda_available", "venv_cuda_available"):
        if not isinstance(report.get(flag), bool):
            errors.append(f"{flag} must be a bool")
    if not isinstance(report.get("blockers"), list):
        errors.append("blockers must be a list")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    if report.get("status") == "pass":
        if not report.get("vllm_importable"):
            errors.append("status pass requires vllm_importable true")
        if not report.get("venv_cuda_available"):
            errors.append("status pass requires venv_cuda_available true")
        if report.get("generation_smoke_attempted") and not report.get("generation_smoke_passed"):
            errors.append("status pass requires generation_smoke_passed when attempted")
    return errors


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

    try:
        has_llm = hasattr(vllm_module, "LLM")
    except Exception:  # noqa: BLE001
        has_llm = False
    if has_llm:
        surfaces["model_loading_surface"] = "accessible"
    else:
        surfaces["model_loading_surface"] = "blocked"

    llm_cls = None
    try:
        llm_cls = getattr(vllm_module, "LLM", None)
    except Exception:  # noqa: BLE001
        llm_cls = None
        surfaces["model_loading_surface"] = "blocked"
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

    surfaces = _inspect_integration_surfaces(None if not llm_importable else vllm_module)
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
