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

EXPERIMENT_061_ID = "exp061_vllm_version_sweep"
DEFAULT_EXP061_REPORT = Path("reports/experiment_061_vllm_version_sweep.json")
DEFAULT_VLLM_SWEEP_ROOT = _REPO_ROOT / ".venv-vllm-sweep"
DEFAULT_SWEEP_LOG_DIR = _REPO_ROOT / "reports" / "vllm_sweep_logs"
DEFAULT_SWEEP_MANIFEST = _REPO_ROOT / "reports" / "vllm_sweep_manifest.json"
DEFAULT_SWEEP_MAX_CANDIDATES = 5

EXP061_CLAIM_NOTE = (
    "vLLM environment compatibility sweep (Phase 15B-unblock). Versioned isolated "
    "venvs only — not system Python. Identifies a candidate vLLM wheel for future "
    "integration work only; no ExactKV vLLM integration, serving, batching, or "
    "performance claims. Default ExactKV generation behavior unchanged."
)

CANDIDATE_CLASSIFICATIONS = (
    "install_failed",
    "import_failed",
    "cuda_failed",
    "generation_failed",
    "pass",
)

EXPERIMENT_062_ID = "exp062_vllm_container_feasibility"
DEFAULT_EXP062_REPORT = Path("reports/experiment_062_vllm_container_feasibility.json")

EXP062_CLAIM_NOTE = (
    "vLLM container/CUDA-13 environment feasibility probe (Phase 15C-env). "
    "Environment and import inspection only — not ExactKV vLLM integration, serving, "
    "batching, or performance claims. Passing this phase means a vLLM-compatible "
    "environment exists for future integration design only. ExactKV default runtime "
    "unchanged."
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


def _sanitize_version_for_path(version: str) -> str:
    return version.replace(".", "_").replace("/", "_")


def sweep_venv_path(version: str, sweep_root: Path | None = None) -> Path:
    """Return isolated venv path for one sweep candidate version."""
    root = sweep_root or DEFAULT_VLLM_SWEEP_ROOT
    return root / f"vllm-{_sanitize_version_for_path(version)}"


def sweep_log_dir(version: str, log_root: Path | None = None) -> Path:
    """Return per-candidate sweep log directory."""
    root = log_root or DEFAULT_SWEEP_LOG_DIR
    return root / version


def _read_known_bad_version_from_exp060() -> str:
    report_path = _REPO_ROOT / DEFAULT_EXP060_REPORT
    if not report_path.is_file():
        return ""
    try:
        import json

        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    if data.get("status") != "blocked":
        return ""
    version = str(data.get("vllm_version", "")).strip()
    if version and not data.get("install_success", True):
        return version
    blockers = " ".join(str(b) for b in data.get("blockers", []))
    if "libcudart.so.13" in blockers or "libcudart.so.13" in str(data.get("import_error", "")):
        return version
    return ""


def query_pip_vllm_versions(*, python_executable: str | Path = SYSTEM_PYTHON) -> list[str]:
    """Return available vLLM versions from pip index (newest first)."""
    import re
    import subprocess

    proc = subprocess.run(
        [str(python_executable), "-m", "pip", "index", "versions", "vllm"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = re.search(r"Available versions:\s*(.+)", text)
    if not match:
        return []
    raw = match.group(1).strip().rstrip(".")
    versions = [v.strip() for v in raw.split(",") if v.strip()]
    return versions


def select_sweep_candidates(
    *,
    explicit_versions: list[str] | None = None,
    max_candidates: int = DEFAULT_SWEEP_MAX_CANDIDATES,
    exclude_versions: list[str] | None = None,
    include_known_bad: bool = False,
    python_executable: str | Path = SYSTEM_PYTHON,
) -> tuple[list[str], list[str]]:
    """Select up to ``max_candidates`` vLLM versions for the sweep."""
    excluded = list(exclude_versions or [])
    known_bad = _read_known_bad_version_from_exp060()
    if known_bad and known_bad not in excluded and not include_known_bad:
        excluded.append(known_bad)

    if explicit_versions:
        chosen = [v.strip() for v in explicit_versions if v.strip()][:max_candidates]
        return chosen, excluded

    available = query_pip_vllm_versions(python_executable=python_executable)
    if not available:
        return [], excluded

    latest = available[0]
    if latest and latest not in excluded and not include_known_bad:
        excluded.append(latest)

    chosen: list[str] = []
    for version in available:
        if version in excluded:
            continue
        chosen.append(version)
        if len(chosen) >= max_candidates:
            break
    return chosen, excluded


def classify_candidate_result(candidate: dict[str, Any]) -> str:
    """Classify one sweep candidate outcome."""
    if not candidate.get("install_success"):
        return "install_failed"
    if not candidate.get("venv_cuda_available"):
        return "cuda_failed"
    functional = (
        bool(candidate.get("import_success"))
        and bool(candidate.get("llm_class_importable"))
        and bool(candidate.get("sampling_params_importable"))
    )
    if not functional:
        return "import_failed"
    if candidate.get("generation_smoke_attempted") and not candidate.get("generation_smoke_passed"):
        return "generation_failed"
    if candidate.get("generation_smoke_passed"):
        return "pass"
    return "generation_failed"


@dataclass
class VllmCandidateResult:
    """One vLLM version sweep candidate outcome."""

    version: str
    venv_path: str
    python_version: str
    install_success: bool
    import_success: bool
    llm_class_importable: bool
    sampling_params_importable: bool
    venv_torch_version: str
    venv_cuda_available: bool
    vllm_version: str
    generation_smoke_attempted: bool
    generation_smoke_passed: bool
    generation_smoke_text: str
    classification: str
    error_summary: str
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_candidate_result_from_probe(
    *,
    version: str,
    venv_path: Path,
    install_success: bool,
    install_error: str = "",
    env_meta: PythonEnvMetadata | None = None,
    probe_payload: dict[str, Any] | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    """Normalize install + probe data into one candidate result dict."""
    meta = env_meta or PythonEnvMetadata(
        python_executable=str(venv_path / "bin" / "python"),
        python_version="",
        torch_version="",
        cuda_available=False,
        gpu_name="",
    )
    probe = probe_payload or {}
    import_success = bool(probe.get("vllm_importable"))
    llm_importable = bool(probe.get("llm_class_importable"))
    sampling_importable = bool(probe.get("sampling_params_importable"))
    gen_attempted = bool(probe.get("generation_smoke_attempted"))
    gen_passed = bool(probe.get("generation_smoke_passed"))
    gen_error = str(probe.get("generation_smoke_error", ""))
    import_error = str(probe.get("import_error", ""))

    error_parts: list[str] = []
    if install_error:
        error_parts.append(install_error)
    if probe.get("blockers"):
        error_parts.extend(str(b) for b in probe["blockers"])
    elif import_error:
        error_parts.append(import_error)
    if gen_error:
        error_parts.append(gen_error)
    if meta.error:
        error_parts.append(meta.error)

    candidate = {
        "version": version,
        "venv_path": str(venv_path),
        "python_version": meta.python_version,
        "install_success": install_success,
        "import_success": import_success and llm_importable and sampling_importable,
        "llm_class_importable": llm_importable,
        "sampling_params_importable": sampling_importable,
        "venv_torch_version": meta.torch_version,
        "venv_cuda_available": meta.cuda_available,
        "vllm_version": str(probe.get("vllm_version", "")),
        "generation_smoke_attempted": gen_attempted,
        "generation_smoke_passed": gen_passed,
        "generation_smoke_text": str(probe.get("generation_smoke_text", "")),
        "error_summary": "; ".join(error_parts),
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }
    candidate["classification"] = classify_candidate_result(candidate)
    return candidate


def build_install_failed_candidate(
    *,
    version: str,
    venv_path: Path,
    install_error: str,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    """Build candidate result when pip install fails."""
    candidate = {
        "version": version,
        "venv_path": str(venv_path),
        "python_version": "",
        "install_success": False,
        "import_success": False,
        "llm_class_importable": False,
        "sampling_params_importable": False,
        "venv_torch_version": "",
        "venv_cuda_available": False,
        "vllm_version": "",
        "generation_smoke_attempted": False,
        "generation_smoke_passed": False,
        "generation_smoke_text": "",
        "error_summary": install_error,
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }
    candidate["classification"] = classify_candidate_result(candidate)
    return candidate


def probe_sweep_candidate(
    *,
    version: str,
    venv_python: Path,
    log_dir: Path | None = None,
    install_success: bool = True,
    install_error: str = "",
    run_generation_smoke: bool = True,
) -> dict[str, Any]:
    """Probe one installed sweep candidate and optionally write ``candidate_result.json``."""
    import json

    venv_path = venv_python.parent.parent
    if not install_success:
        result = build_install_failed_candidate(
            version=version,
            venv_path=venv_path,
            install_error=install_error,
        )
    else:
        env_meta = collect_python_env_metadata(venv_python)
        probe_payload, stdout, stderr = probe_vllm_in_subprocess(
            venv_python,
            run_generation_smoke=run_generation_smoke,
        )
        result = build_candidate_result_from_probe(
            version=version,
            venv_path=venv_path,
            install_success=True,
            env_meta=env_meta,
            probe_payload=probe_payload,
            stdout=stdout,
            stderr=stderr,
        )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "candidate_result.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
    return result


def load_sweep_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Load sweep manifest written by the bash sweep script."""
    import json

    path = manifest_path or DEFAULT_SWEEP_MANIFEST
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_results(
    *,
    manifest: dict[str, Any] | None = None,
    log_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load per-candidate results from manifest or sweep log directories."""
    import json

    root = log_root or DEFAULT_SWEEP_LOG_DIR
    results: list[dict[str, Any]] = []
    manifest = manifest or {}
    candidates = manifest.get("candidates")
    if isinstance(candidates, list) and candidates:
        for version in candidates:
            result_path = root / str(version) / "candidate_result.json"
            if result_path.is_file():
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
        return results
    if root.is_dir():
        for child in sorted(root.iterdir()):
            result_path = child / "candidate_result.json"
            if result_path.is_file():
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
    return results


def recommend_next_step_after_sweep(*, any_candidate_passed: bool) -> str:
    if any_candidate_passed:
        return (
            "Phase 15C: vLLM API surface reconnaissance in the winning isolated venv — "
            "still no ExactKV integration"
        )
    return (
        "Separate environment phase: CUDA 13 base image, official vLLM container, "
        "or source build — then re-run version sweep"
    )


def build_exp061_report(
    *,
    candidate_results: list[dict[str, Any]],
    candidates: list[str] | None = None,
    excluded_versions: list[str] | None = None,
    system_python: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized Exp 061 report from sweep candidate results."""
    sys_py = Path(system_python or SYSTEM_PYTHON)
    system_meta = collect_python_env_metadata(sys_py)
    blockers: list[str] = []

    winning: str | None = None
    any_passed = False
    gen_passed = False
    for candidate in candidate_results:
        if candidate.get("classification") == "pass":
            winning = str(candidate.get("version", ""))
            any_passed = True
            gen_passed = bool(candidate.get("generation_smoke_passed"))
            break
        summary = str(candidate.get("error_summary", "")).strip()
        if summary:
            blockers.append(f"{candidate.get('version')}: {summary}")

    candidate_list = list(candidates or [])
    if not candidate_list:
        candidate_list = [str(c.get("version", "")) for c in candidate_results if c.get("version")]

    excluded = list(excluded_versions or [])
    if manifest:
        excluded = list(manifest.get("excluded_versions") or excluded)
        if not candidate_list:
            candidate_list = list(manifest.get("candidates") or [])
        if manifest.get("winning_candidate"):
            winning = str(manifest["winning_candidate"])

    if not candidate_results:
        blockers.append("no sweep candidate results found — run sweep_vllm_versions_runpod.sh")

    return {
        "experiment_id": EXPERIMENT_061_ID,
        "system_python": str(sys_py),
        "system_torch_version": system_meta.torch_version,
        "system_cuda_available": system_meta.cuda_available,
        "gpu_name": system_meta.gpu_name,
        "candidates": candidate_list,
        "excluded_versions": excluded,
        "candidate_results": candidate_results,
        "winning_candidate": winning,
        "any_candidate_passed": any_passed,
        "generation_smoke_passed": gen_passed,
        "blockers": blockers,
        "recommended_next_step": recommend_next_step_after_sweep(any_candidate_passed=any_passed),
        "claim_note": EXP061_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }


def validate_exp061_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 061 JSON report schema."""
    errors: list[str] = []
    required_top = (
        "experiment_id",
        "system_python",
        "system_torch_version",
        "system_cuda_available",
        "gpu_name",
        "candidates",
        "candidate_results",
        "winning_candidate",
        "any_candidate_passed",
        "generation_smoke_passed",
        "blockers",
        "recommended_next_step",
        "claim_note",
        "forbidden_claims",
    )
    for key in required_top:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_061_ID:
        errors.append("experiment_id must be exp061_vllm_version_sweep")
    if not isinstance(report.get("candidates"), list):
        errors.append("candidates must be a list")
    if not isinstance(report.get("candidate_results"), list):
        errors.append("candidate_results must be a list")
    for flag in ("system_cuda_available", "any_candidate_passed", "generation_smoke_passed"):
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

    required_candidate = (
        "version",
        "venv_path",
        "install_success",
        "import_success",
        "llm_class_importable",
        "sampling_params_importable",
        "venv_torch_version",
        "venv_cuda_available",
        "vllm_version",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "error_summary",
        "stdout_tail",
        "stderr_tail",
    )
    for idx, candidate in enumerate(report.get("candidate_results", [])):
        if not isinstance(candidate, dict):
            errors.append(f"candidate_results[{idx}] must be a dict")
            continue
        for key in required_candidate:
            if key not in candidate:
                errors.append(f"candidate_results[{idx}] missing field: {key}")
        classification = candidate.get("classification")
        if classification not in CANDIDATE_CLASSIFICATIONS:
            errors.append(f"candidate_results[{idx}] invalid classification: {classification}")
        for flag in (
            "install_success",
            "import_success",
            "llm_class_importable",
            "sampling_params_importable",
            "venv_cuda_available",
            "generation_smoke_attempted",
            "generation_smoke_passed",
        ):
            if not isinstance(candidate.get(flag), bool):
                errors.append(f"candidate_results[{idx}].{flag} must be a bool")

    if report.get("any_candidate_passed") and not report.get("winning_candidate"):
        errors.append("any_candidate_passed requires winning_candidate")
    if report.get("winning_candidate") and not report.get("any_candidate_passed"):
        errors.append("winning_candidate requires any_candidate_passed true")
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
    generation_smoke_text: str = ""
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
) -> tuple[bool, bool, str, str]:
    """Tiny greedy generation smoke when vLLM is already importable."""
    torch_version, cuda_available, _ = _torch_environment()
    if not cuda_available:
        return False, False, "CUDA unavailable — generation smoke skipped", ""
    if not torch_version:
        return False, False, "torch unavailable — generation smoke skipped", ""

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        return False, False, f"LLM/SamplingParams import failed: {exc}", ""

    try:
        llm = LLM(model=model_id, dtype="float16", max_model_len=256)
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)
        outputs = llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return True, False, "generation returned empty outputs", ""
        text = outputs[0].outputs[0].text
        return True, True, "", text
    except Exception as exc:  # noqa: BLE001
        return True, False, f"{type(exc).__name__}: {exc}", ""


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
            generation_smoke_text="",
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
            generation_smoke_text="",
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
    gen_text = ""

    if run_generation_smoke and llm_importable and sampling_importable:
        gen_attempted, gen_passed, gen_error, gen_text = _attempt_generation_smoke(
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
        generation_smoke_text=gen_text,
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
        generation_smoke_text="",
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


def _cuda_runtime_version() -> str:
    try:
        import torch

        return str(getattr(torch.version, "cuda", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def preview_generated_text(text: str, *, max_len: int = 200) -> str:
    """Return a short preview of generation smoke output for reports."""
    cleaned = text.strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def build_exp062_report_from_probe(
    result: VllmProbeResult,
    *,
    environment_label: str = "",
    python_version: str = "",
) -> dict[str, Any]:
    """Build normalized Exp 062 report from a vLLM probe result."""
    py_version = python_version or platform.python_version()
    return {
        "experiment_id": EXPERIMENT_062_ID,
        "status": result.status,
        "environment_label": environment_label,
        "python_executable": result.python_executable,
        "python_version": py_version,
        "platform": result.platform_info,
        "torch_version": result.torch_version,
        "cuda_available": result.cuda_available,
        "cuda_runtime": _cuda_runtime_version(),
        "gpu_name": result.gpu_name,
        "vllm_importable": result.vllm_importable,
        "vllm_version": result.vllm_version,
        "llm_class_importable": result.llm_class_importable,
        "sampling_params_importable": result.sampling_params_importable,
        "import_error": result.import_error,
        "generation_smoke_attempted": result.generation_smoke_attempted,
        "generation_smoke_passed": result.generation_smoke_passed,
        "generation_smoke_error": result.generation_smoke_error,
        "generated_text_preview": preview_generated_text(result.generation_smoke_text),
        "visible_integration_surfaces": dict(result.visible_integration_surfaces),
        "kv_cache_access_status": result.kv_cache_access_status,
        "blockers": list(result.blockers),
        "claim_note": EXP062_CLAIM_NOTE,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }


def probe_vllm_container_feasibility(
    *,
    run_generation_smoke: bool = True,
    smoke_model_id: str = DEFAULT_SMOKE_MODEL,
    environment_label: str = "",
) -> dict[str, Any]:
    """Run Exp 062 vLLM container/CUDA-13 environment feasibility probe."""
    result = probe_vllm_availability(
        run_generation_smoke=run_generation_smoke,
        smoke_model_id=smoke_model_id,
    )
    return build_exp062_report_from_probe(
        result,
        environment_label=environment_label,
        python_version=platform.python_version(),
    )


def recommend_next_step_after_container_probe(*, status: str) -> str:
    if status == "pass":
        return (
            "Phase 15C: vLLM API surface reconnaissance and KV cache visibility "
            "mapping — still no ExactKV integration"
        )
    return (
        "Resolve container/CUDA-13 environment blockers or choose another "
        "vLLM-compatible image before integration design work"
    )


def validate_exp062_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 062 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "status",
        "python_executable",
        "python_version",
        "platform",
        "torch_version",
        "cuda_available",
        "gpu_name",
        "vllm_importable",
        "vllm_version",
        "llm_class_importable",
        "sampling_params_importable",
        "generation_smoke_attempted",
        "generation_smoke_passed",
        "generation_smoke_error",
        "generated_text_preview",
        "visible_integration_surfaces",
        "kv_cache_access_status",
        "blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_062_ID:
        errors.append("experiment_id must be exp062_vllm_container_feasibility")
    if report.get("status") not in ("pass", "blocked", "failed"):
        errors.append("status must be pass, blocked, or failed")
    for flag in (
        "cuda_available",
        "vllm_importable",
        "llm_class_importable",
        "sampling_params_importable",
        "generation_smoke_attempted",
        "generation_smoke_passed",
    ):
        if not isinstance(report.get(flag), bool):
            errors.append(f"{flag} must be a bool")
    if not isinstance(report.get("blockers"), list):
        errors.append("blockers must be a list")
    surfaces = report.get("visible_integration_surfaces")
    if not isinstance(surfaces, dict):
        errors.append("visible_integration_surfaces must be a dict")
    else:
        for key in (
            "model_loading_surface",
            "generation_call_surface",
            "sampling_greedy_config_surface",
            "kv_cache_access_surface",
            "scheduler_cache_api_surface",
            "restored_full_kv_verifier_path",
        ):
            if key not in surfaces:
                errors.append(f"visible_integration_surfaces missing {key}")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    if report.get("status") == "pass":
        if not report.get("vllm_importable"):
            errors.append("status pass requires vllm_importable true")
        if not report.get("llm_class_importable"):
            errors.append("status pass requires llm_class_importable true")
        if not report.get("sampling_params_importable"):
            errors.append("status pass requires sampling_params_importable true")
        if report.get("generation_smoke_attempted") and not report.get("generation_smoke_passed"):
            errors.append("status pass requires generation_smoke_passed when attempted")
    return errors
