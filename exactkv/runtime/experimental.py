"""Non-default experimental restored-verifier runtime (Phase 13A).

Explicit opt-in entry point for the restored full-KV verifier runner.
**Does not** change ``ExactKVGenerator``, ``VerificationEngine``, or default CLI.

This is a non-default experimental restored-verifier runtime path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import (
    DEFAULT_MODEL,
    VERIFIER_SOURCE,
    configure_cuda_determinism,
    resolve_cuda_drift_dtype_configs,
)
from exactkv.cache.restored_verifier_runner import (
    DEFAULT_SMOKE_COMPRESSORS,
    DEFAULT_SMOKE_DRAFT_LEN,
    DEFAULT_SMOKE_MAX_NEW_TOKENS,
    DEFAULT_SMOKE_PROMPT_IDS,
    RestoredVerifierRunConfig,
    RestoredVerifierRunReport,
    run_restored_verifier,
)

EXPERIMENT_054_ID = "exp054_experimental_restored_verifier_runtime"
EXPERIMENT_056_ID = "exp056_cuda_restored_verifier_runtime_gate"
RUNTIME_PATH_EXPERIMENTAL = "run_experimental_restored_verifier"
CLI_OPT_IN_REQUIRED = True
DEFAULT_CUDA_GATE_MAX_NEW_TOKENS = 12
DEFAULT_CUDA_GATE_DRAFT_LEN = 4

EXPERIMENTAL_RUNTIME_CLAIM_NOTE = (
    "Non-default experimental restored-verifier runtime (Phase 13A). Explicit opt-in "
    "only — restored full KV used when enabled; default ExactKV generation unchanged. "
    "Not vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

EXP054_CLAIM_NOTE = EXPERIMENTAL_RUNTIME_CLAIM_NOTE

EXPERIMENTAL_CUDA_GATE_CLAIM_NOTE = (
    "CUDA exactness gate for explicit experimental restored-verifier runtime "
    "(Phase 14A). Restored full KV used only when explicitly enabled; default "
    "ExactKV generation unchanged. Not vLLM, LMCache, remote prefix runtime, or "
    "serving. No speed, latency, throughput, active memory savings, or "
    "production-serving claim. Passing this gate is CUDA exactness evidence only, "
    "not a performance result."
)

EXP056_CLAIM_NOTE = EXPERIMENTAL_CUDA_GATE_CLAIM_NOTE

_REQUIRED_CLAIM_MARKERS = (
    "experimental",
    "non-default",
    "not default",
    "default exactkv generation",
)


class ExperimentalRuntimeMode(str, Enum):
    """Explicit experimental runtime modes — never implicit default."""

    DEFAULT = "default"
    RESTORED_VERIFIER_OFFLINE = "restored_verifier_offline"


@dataclass
class ExperimentalRestoredVerifierConfig:
    """Explicit opt-in config for experimental restored-verifier runtime."""

    enabled: bool = False
    mode: ExperimentalRuntimeMode = ExperimentalRuntimeMode.DEFAULT
    model_id: str = ""
    device: str = ""
    dtype: str = ""
    prompt_ids: list[str] = field(default_factory=list)
    storage_backends: list[str] = field(default_factory=list)
    compressor_names: list[str] = field(default_factory=list)
    draft_lens: list[int] = field(default_factory=list)
    max_new_tokens: int = 0
    verifier_source: str = ""
    claim_note: str = ""
    namespace_prefix: str = "experimental_restored_verifier"
    file_storage_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentalRestoredVerifierConfig:
        mode_raw = data.get("mode", ExperimentalRuntimeMode.DEFAULT.value)
        mode = (
            ExperimentalRuntimeMode(mode_raw)
            if isinstance(mode_raw, str)
            else ExperimentalRuntimeMode.DEFAULT
        )
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "mode"}
        return cls(mode=mode, **fields)

    @classmethod
    def disabled(cls) -> ExperimentalRestoredVerifierConfig:
        """Factory for explicitly disabled experimental runtime."""
        return cls(enabled=False, mode=ExperimentalRuntimeMode.DEFAULT)


@dataclass
class ExperimentalRuntimeResult:
    """Structured result from ``run_experimental_restored_verifier``."""

    enabled: bool
    mode: str
    status: str
    runner_called: bool
    validation_errors: list[str] = field(default_factory=list)
    runner_report: RestoredVerifierRunReport | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "enabled": self.enabled,
            "mode": self.mode,
            "status": self.status,
            "runner_called": self.runner_called,
            "validation_errors": list(self.validation_errors),
            "message": self.message,
        }
        if self.runner_report is not None:
            data["runner_report"] = self.runner_report.to_dict()
        return data


def default_experimental_smoke_config(**overrides: Any) -> ExperimentalRestoredVerifierConfig:
    """Default Exp 054 experimental runtime smoke configuration."""
    cfg = ExperimentalRestoredVerifierConfig(
        enabled=True,
        mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE,
        model_id=DEFAULT_MODEL,
        device="cpu",
        dtype="float32",
        prompt_ids=list(DEFAULT_SMOKE_PROMPT_IDS),
        storage_backends=["in_memory_kv_storage"],
        compressor_names=list(DEFAULT_SMOKE_COMPRESSORS),
        draft_lens=[DEFAULT_SMOKE_DRAFT_LEN],
        max_new_tokens=DEFAULT_SMOKE_MAX_NEW_TOKENS,
        verifier_source=VERIFIER_SOURCE,
        claim_note=EXP054_CLAIM_NOTE,
        namespace_prefix="exp054",
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def validate_experimental_config(config: ExperimentalRestoredVerifierConfig) -> list[str]:
    """Validate experimental restored-verifier runtime config."""
    errors: list[str] = []
    if not config.enabled:
        return errors

    if config.mode != ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE:
        errors.append(
            "enabled=True requires mode == RESTORED_VERIFIER_OFFLINE"
        )

    if not config.model_id.strip():
        errors.append("model_id required when enabled")
    if not config.device.strip():
        errors.append("device required when enabled")
    if not config.dtype.strip():
        errors.append("dtype required when enabled")
    if not config.prompt_ids:
        errors.append("prompt_ids required when enabled")
    if not config.storage_backends:
        errors.append("storage_backends required when enabled")
    if not config.compressor_names:
        errors.append("compressor_names required when enabled")
    if not config.draft_lens:
        errors.append("draft_lens required when enabled")
    if config.max_new_tokens <= 0:
        errors.append("max_new_tokens must be positive when enabled")
    if config.verifier_source != VERIFIER_SOURCE:
        errors.append(f"verifier_source must be {VERIFIER_SOURCE!r}")
    if not config.claim_note.strip():
        errors.append("claim_note required when enabled")
    else:
        lower = config.claim_note.lower()
        if not any(marker in lower for marker in _REQUIRED_CLAIM_MARKERS):
            errors.append(
                "claim_note must state experimental/non-default caveats"
            )

    for backend in config.storage_backends:
        if backend not in ("in_memory_kv_storage", "file_kv_storage"):
            errors.append(f"unsupported storage backend: {backend}")

    for draft_len in config.draft_lens:
        if draft_len <= 0:
            errors.append("draft_lens must be positive integers")

    return errors


def _runner_config_from_experimental(
    config: ExperimentalRestoredVerifierConfig,
) -> RestoredVerifierRunConfig:
    """Map experimental config to runner config without global mutation."""
    primary_backend = config.storage_backends[0]
    draft_lens = list(config.draft_lens)
    return RestoredVerifierRunConfig(
        model_id=config.model_id,
        device=config.device,
        dtype=config.dtype,
        prompt_ids=list(config.prompt_ids),
        storage_backend_name=primary_backend,
        compressor_names=list(config.compressor_names),
        draft_len=draft_lens[0],
        draft_len_values=draft_lens,
        max_new_tokens=config.max_new_tokens,
        verifier_source=config.verifier_source,
        claim_note=config.claim_note,
        namespace_prefix=config.namespace_prefix,
        file_storage_root=config.file_storage_root,
    )


def run_experimental_restored_verifier(
    config: ExperimentalRestoredVerifierConfig,
    *,
    experiment_id: str = EXPERIMENT_054_ID,
) -> ExperimentalRuntimeResult:
    """Run experimental restored-verifier runtime when explicitly enabled.

    Disabled configs return a clean result without calling the runner.
    No environment-variable activation. No global registry mutation.
    """
    if not config.enabled:
        return ExperimentalRuntimeResult(
            enabled=False,
            mode=config.mode.value,
            status="disabled",
            runner_called=False,
            message="experimental restored-verifier runtime not enabled",
        )

    validation_errors = validate_experimental_config(config)
    if validation_errors:
        return ExperimentalRuntimeResult(
            enabled=True,
            mode=config.mode.value,
            status="invalid",
            runner_called=False,
            validation_errors=validation_errors,
            message="experimental config validation failed",
        )

    runner_cfg = _runner_config_from_experimental(config)
    extra_backends = [
        b for b in config.storage_backends[1:] if b != runner_cfg.storage_backend_name
    ]
    report = run_restored_verifier(
        runner_cfg,
        experiment_id=experiment_id,
        extra_backends=extra_backends or None,
    )
    status = report.status
    return ExperimentalRuntimeResult(
        enabled=True,
        mode=config.mode.value,
        status=status,
        runner_called=True,
        runner_report=report,
        message="experimental restored-verifier run complete",
    )


def report_to_exp054_json(result: ExperimentalRuntimeResult) -> dict[str, Any]:
    """Serialize experimental runtime result to Exp 054 JSON schema."""
    if not result.enabled or result.runner_report is None:
        return {
            "experiment_id": EXPERIMENT_054_ID,
            "status": result.status,
            "runtime_mode": result.mode,
            "enabled": result.enabled,
            "model": "",
            "device": "",
            "dtype": "",
            "prompt_count": 0,
            "storage_backends": [],
            "compressor_names": [],
            "draft_len": 0,
            "draft_lens": [],
            "max_new_tokens": 0,
            "verifier_source": VERIFIER_SOURCE,
            "total_cells": 0,
            "exactkv_failures": 0,
            "token_exact_match_count": 0,
            "mean_acceptance": 0.0,
            "draft_divergence_count": 0,
            "restore_blockers": [],
            "draft_blockers": [],
            "verification_blockers": [],
            "blockers": {
                "restore_blockers": [],
                "draft_blockers": [],
                "verification_blockers": [],
            },
            "validation_errors": list(result.validation_errors),
            "runner_called": result.runner_called,
            "claim_note": EXPERIMENTAL_RUNTIME_CLAIM_NOTE,
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
            "message": result.message,
        }

    report = result.runner_report
    cfg = report.config
    blockers = report.blockers
    draft_lens = cfg.resolved_draft_lens()
    return {
        "experiment_id": EXPERIMENT_054_ID,
        "status": result.status,
        "runtime_mode": result.mode,
        "enabled": result.enabled,
        "config": cfg.to_dict(),
        "model": cfg.model_id,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "prompt_count": len(cfg.prompt_ids),
        "storage_backends": sorted({c.storage_backend for c in report.cells}),
        "compressor_names": list(cfg.compressor_names),
        "draft_len": draft_lens[0] if draft_lens else 0,
        "draft_lens": draft_lens,
        "max_new_tokens": cfg.max_new_tokens,
        "verifier_source": cfg.verifier_source,
        "total_cells": report.total_cells,
        "exactkv_failures": report.exactkv_failures,
        "token_exact_match_count": report.token_exact_match_count,
        "mean_acceptance": report.mean_acceptance,
        "draft_divergence_count": report.draft_divergence_count,
        "semantic_divergence_count": report.semantic_divergence_count,
        "no_real_drift_observed": report.no_real_drift_observed,
        "first_divergences": report.first_divergences,
        "restore_blockers": blockers.get("restore_blockers", []),
        "draft_blockers": blockers.get("draft_blockers", []),
        "verification_blockers": blockers.get("verification_blockers", []),
        "blockers": blockers,
        "validation_errors": list(result.validation_errors),
        "runner_called": result.runner_called,
        "claim_note": report.claim_note,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "message": result.message,
    }


def validate_exp054_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 054 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "runtime_mode",
        "enabled",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends",
        "compressor_names",
        "draft_len",
        "max_new_tokens",
        "verifier_source",
        "exactkv_failures",
        "token_exact_match_count",
        "mean_acceptance",
        "draft_divergence_count",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "claim_note",
        "forbidden_claims",
        "runner_called",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_054_ID:
        errors.append("experiment_id must be exp054_experimental_restored_verifier_runtime")
    if not isinstance(report.get("enabled"), bool):
        errors.append("enabled must be a bool")
    if not isinstance(report.get("runner_called"), bool):
        errors.append("runner_called must be a bool")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    if report.get("enabled") and report.get("runner_called"):
        failures = int(report.get("exactkv_failures", -1))
        exact_count = int(report.get("token_exact_match_count", -1))
        total = int(report.get("total_cells", 0))
        if total > 0 and exact_count + failures != total:
            errors.append("token_exact_match_count + exactkv_failures must equal total_cells")
    draft_div = report.get("draft_divergence_count")
    if not isinstance(draft_div, int) or draft_div < 0:
        errors.append("draft_divergence_count must be a non-negative int")
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    return errors


@dataclass
class CudaRuntimeGateDtypeResult:
    """Per-dtype experimental runtime gate outcome."""

    dtype: str
    dtype_supported: bool
    status: str
    skip_reason: str = ""
    runtime_result: ExperimentalRuntimeResult | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "dtype": self.dtype,
            "dtype_supported": self.dtype_supported,
            "status": self.status,
            "skip_reason": self.skip_reason,
        }
        if self.runtime_result is not None:
            data["runtime_result"] = self.runtime_result.to_dict()
        return data


@dataclass
class CudaRuntimeGateResult:
    """Aggregate CUDA restored-verifier runtime gate result."""

    cuda_available: bool
    dtype_results: list[CudaRuntimeGateDtypeResult]
    status: str
    total_cells: int
    exactkv_failures: int
    token_exact_match_count: int
    mean_acceptance: float
    draft_divergence_count: int
    accepted_prefix_lengths: list[list[int]]
    first_divergences: list[dict[str, Any]]
    skipped_configs: list[dict[str, Any]]
    cuda_blockers: list[str]
    restore_blockers: list[str]
    draft_blockers: list[str]
    verification_blockers: list[str]
    dtype_supported: dict[str, bool]
    claim_note: str = EXP056_CLAIM_NOTE


def default_cuda_gate_experimental_config(
    dtype: str,
    **overrides: Any,
) -> ExperimentalRestoredVerifierConfig:
    """Explicit enabled experimental config for one CUDA dtype gate cell."""
    cfg = ExperimentalRestoredVerifierConfig(
        enabled=True,
        mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE,
        model_id=DEFAULT_MODEL,
        device="cuda",
        dtype=dtype,
        prompt_ids=list(DEFAULT_SMOKE_PROMPT_IDS),
        storage_backends=["in_memory_kv_storage"],
        compressor_names=list(DEFAULT_SMOKE_COMPRESSORS),
        draft_lens=[DEFAULT_CUDA_GATE_DRAFT_LEN],
        max_new_tokens=DEFAULT_CUDA_GATE_MAX_NEW_TOKENS,
        verifier_source=VERIFIER_SOURCE,
        claim_note=EXP056_CLAIM_NOTE,
        namespace_prefix=f"exp056/{dtype}",
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def run_cuda_restored_verifier_runtime_gate(
    *,
    model_id: str = DEFAULT_MODEL,
    prompt_ids: list[str] | None = None,
    max_new_tokens: int = DEFAULT_CUDA_GATE_MAX_NEW_TOKENS,
    draft_len: int = DEFAULT_CUDA_GATE_DRAFT_LEN,
) -> CudaRuntimeGateResult:
    """Run CUDA exactness gate via explicit experimental runtime path only."""
    cuda_available = torch.cuda.is_available()
    dtype_configs = resolve_cuda_drift_dtype_configs()
    dtype_supported = {c.dtype: c.dtype_supported for c in dtype_configs}
    skipped_configs = [c.to_dict() for c in dtype_configs if c.status == "skipped"]
    prompt_ids = prompt_ids or list(DEFAULT_SMOKE_PROMPT_IDS)

    if not cuda_available:
        return CudaRuntimeGateResult(
            cuda_available=False,
            dtype_results=[],
            status="blocked",
            total_cells=0,
            exactkv_failures=0,
            token_exact_match_count=0,
            mean_acceptance=0.0,
            draft_divergence_count=0,
            accepted_prefix_lengths=[],
            first_divergences=[],
            skipped_configs=skipped_configs,
            cuda_blockers=["CUDA unavailable"],
            restore_blockers=[],
            draft_blockers=[],
            verification_blockers=[],
            dtype_supported=dtype_supported,
        )

    configure_cuda_determinism()
    dtype_results: list[CudaRuntimeGateDtypeResult] = []
    all_accepted: list[list[int]] = []
    per_cell_mean: list[float] = []
    first_divergences: list[dict[str, Any]] = []
    restore_blockers: list[str] = []
    draft_blockers: list[str] = []
    verification_blockers: list[str] = []
    cuda_blockers: list[str] = []
    total_cells = 0
    exact_matches = 0
    failures = 0
    total_draft_div = 0
    tested_any = False

    for cfg_entry in dtype_configs:
        if cfg_entry.status == "skipped":
            dtype_results.append(
                CudaRuntimeGateDtypeResult(
                    dtype=cfg_entry.dtype,
                    dtype_supported=cfg_entry.dtype_supported,
                    status="skipped",
                    skip_reason=cfg_entry.skip_reason,
                )
            )
            continue

        exp_cfg = default_cuda_gate_experimental_config(
            cfg_entry.dtype,
            model_id=model_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            draft_lens=[draft_len],
        )
        try:
            runtime_result = run_experimental_restored_verifier(
                exp_cfg,
                experiment_id=EXPERIMENT_056_ID,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"{cfg_entry.dtype}: {type(exc).__name__}: {exc}"
            cuda_blockers.append(reason)
            cfg_entry.status = "skipped"
            cfg_entry.skip_reason = reason
            dtype_results.append(
                CudaRuntimeGateDtypeResult(
                    dtype=cfg_entry.dtype,
                    dtype_supported=cfg_entry.dtype_supported,
                    status="skipped",
                    skip_reason=reason,
                )
            )
            skipped_configs.append(cfg_entry.to_dict())
            continue

        tested_any = True
        dtype_status = runtime_result.status
        if runtime_result.runner_report is not None:
            report = runtime_result.runner_report
            total_cells += report.total_cells
            exact_matches += report.token_exact_match_count
            failures += report.exactkv_failures
            total_draft_div += report.draft_divergence_count
            for cell in report.cells:
                all_accepted.append(list(cell.accepted_prefix_lengths))
                per_cell_mean.append(cell.mean_acceptance)
            first_divergences.extend(report.first_divergences)
            blockers = report.blockers
            restore_blockers.extend(blockers.get("restore_blockers", []))
            draft_blockers.extend(blockers.get("draft_blockers", []))
            verification_blockers.extend(blockers.get("verification_blockers", []))
            if report.exactkv_failures > 0:
                cuda_blockers.append(
                    f"{cfg_entry.dtype}: exactkv_failures={report.exactkv_failures}"
                )

        dtype_results.append(
            CudaRuntimeGateDtypeResult(
                dtype=cfg_entry.dtype,
                dtype_supported=cfg_entry.dtype_supported,
                status=dtype_status,
                runtime_result=runtime_result,
            )
        )
        cfg_entry.status = "tested"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not tested_any:
        return CudaRuntimeGateResult(
            cuda_available=True,
            dtype_results=dtype_results,
            status="blocked",
            total_cells=0,
            exactkv_failures=0,
            token_exact_match_count=0,
            mean_acceptance=0.0,
            draft_divergence_count=0,
            accepted_prefix_lengths=[],
            first_divergences=[],
            skipped_configs=skipped_configs,
            cuda_blockers=cuda_blockers or ["no CUDA dtype configs could be tested"],
            restore_blockers=restore_blockers,
            draft_blockers=draft_blockers,
            verification_blockers=verification_blockers,
            dtype_supported=dtype_supported,
        )

    mean_acc = sum(per_cell_mean) / len(per_cell_mean) if per_cell_mean else 0.0
    status = "pass" if failures == 0 else "failed"
    return CudaRuntimeGateResult(
        cuda_available=True,
        dtype_results=dtype_results,
        status=status,
        total_cells=total_cells,
        exactkv_failures=failures,
        token_exact_match_count=exact_matches,
        mean_acceptance=mean_acc,
        draft_divergence_count=total_draft_div,
        accepted_prefix_lengths=all_accepted,
        first_divergences=first_divergences,
        skipped_configs=skipped_configs,
        cuda_blockers=cuda_blockers,
        restore_blockers=restore_blockers,
        draft_blockers=draft_blockers,
        verification_blockers=verification_blockers,
        dtype_supported=dtype_supported,
    )


def report_to_exp056_json(
    result: CudaRuntimeGateResult,
    *,
    model_id: str = DEFAULT_MODEL,
    prompt_count: int | None = None,
) -> dict[str, Any]:
    """Serialize CUDA runtime gate result to Exp 056 JSON schema."""
    tested_dtypes = [d.dtype for d in result.dtype_results if d.status != "skipped"]
    prompt_count = prompt_count if prompt_count is not None else len(DEFAULT_SMOKE_PROMPT_IDS)
    return {
        "experiment_id": EXPERIMENT_056_ID,
        "status": result.status,
        "cuda_available": result.cuda_available,
        "model": model_id,
        "runtime_path": RUNTIME_PATH_EXPERIMENTAL,
        "cli_opt_in_required": CLI_OPT_IN_REQUIRED,
        "device": "cuda" if result.cuda_available else "unknown",
        "dtype_configs": tested_dtypes,
        "dtype_supported": result.dtype_supported,
        "prompt_count": prompt_count,
        "storage_backend": "in_memory_kv_storage",
        "compressor_names": list(DEFAULT_SMOKE_COMPRESSORS),
        "draft_len": DEFAULT_CUDA_GATE_DRAFT_LEN,
        "max_new_tokens": DEFAULT_CUDA_GATE_MAX_NEW_TOKENS,
        "verifier_source": VERIFIER_SOURCE,
        "total_cells": result.total_cells,
        "exactkv_failures": result.exactkv_failures,
        "token_exact_match_count": result.token_exact_match_count,
        "mean_acceptance": result.mean_acceptance,
        "draft_divergence_count": result.draft_divergence_count,
        "accepted_prefix_lengths": result.accepted_prefix_lengths,
        "first_divergences": result.first_divergences,
        "skipped_configs": result.skipped_configs,
        "cuda_blockers": result.cuda_blockers,
        "restore_blockers": result.restore_blockers,
        "draft_blockers": result.draft_blockers,
        "verification_blockers": result.verification_blockers,
        "dtype_results": [d.to_dict() for d in result.dtype_results],
        "claim_note": result.claim_note,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }


def validate_exp056_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 056 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "cuda_available",
        "model",
        "runtime_path",
        "cli_opt_in_required",
        "device",
        "dtype_configs",
        "dtype_supported",
        "prompt_count",
        "storage_backend",
        "compressor_names",
        "draft_len",
        "max_new_tokens",
        "verifier_source",
        "total_cells",
        "exactkv_failures",
        "token_exact_match_count",
        "mean_acceptance",
        "draft_divergence_count",
        "accepted_prefix_lengths",
        "first_divergences",
        "skipped_configs",
        "cuda_blockers",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_056_ID:
        errors.append("experiment_id must be exp056_cuda_restored_verifier_runtime_gate")
    if not isinstance(report.get("cuda_available"), bool):
        errors.append("cuda_available must be a bool")
    if report.get("runtime_path") != RUNTIME_PATH_EXPERIMENTAL:
        errors.append("runtime_path must be run_experimental_restored_verifier")
    if report.get("cli_opt_in_required") is not True:
        errors.append("cli_opt_in_required must be true")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not isinstance(report.get("dtype_supported"), dict):
        errors.append("dtype_supported must be a dict")
    if not isinstance(report.get("skipped_configs"), list):
        errors.append("skipped_configs must be a list")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    if not report.get("cuda_available") and report.get("status") == "pass":
        errors.append("status cannot be pass when cuda_available is false")
    exact_count = int(report.get("token_exact_match_count", -1))
    failures = int(report.get("exactkv_failures", -1))
    total = int(report.get("total_cells", -1))
    if total >= 0 and exact_count >= 0 and failures >= 0 and total > 0:
        if exact_count + failures != total:
            errors.append("token_exact_match_count + exactkv_failures must equal total_cells")
    draft_div = report.get("draft_divergence_count")
    if not isinstance(draft_div, int) or draft_div < 0:
        errors.append("draft_divergence_count must be a non-negative int")
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    return errors
