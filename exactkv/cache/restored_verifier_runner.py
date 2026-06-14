"""Isolated restored-verifier runner (Phase 12G).

Reusable experimental API for ``lossy draft + stored/reloaded full-KV verifier``
without wiring into ``ExactKVGenerator`` or default runtime.

This is an isolated restored-verifier runner, not default runtime integration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from exactkv.cache.hf_kv_restore import FORBIDDEN_CLAIMS
from exactkv.cache.offline_verifier import (
    DEFAULT_DRIFT_DRAFT_LENS,
    DEFAULT_DRIFT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    OfflineDriftStressCellResult,
    VERIFIER_SOURCE,
    default_drift_stress_compressors,
    default_drift_stress_prompts,
    default_offline_prompts,
    run_offline_drift_stress_cell,
)
from exactkv.cache.storage import FileKVStorageBackend, InMemoryKVStorageBackend, KVStorageBackend
from exactkv.runtime.model_runtime import ModelRuntime

EXPERIMENT_052_ID = "exp052_restored_verifier_runner_smoke"
EXPERIMENT_053_ID = "exp053_restored_verifier_runner_panel"
DEFAULT_EXP051_REPORT = Path("reports/experiment_051_offline_verifier_cuda_drift_panel.json")
DEFAULT_SMOKE_MAX_NEW_TOKENS = 12
DEFAULT_SMOKE_DRAFT_LEN = 4
DEFAULT_PANEL_MAX_NEW_TOKENS = DEFAULT_DRIFT_MAX_NEW_TOKENS
DEFAULT_PANEL_DRAFT_LENS = list(DEFAULT_DRIFT_DRAFT_LENS)
DEFAULT_SMOKE_COMPRESSORS = ("int4_sim", "k8_v4_sim", "int8")
DEFAULT_PANEL_COMPRESSORS = tuple(default_drift_stress_compressors()) or (
    "int4_sim",
    "k8_v4_sim",
    "k8_v4_boundary4_v8_sim",
    "int8",
)
DEFAULT_SMOKE_PROMPT_IDS = (
    "offline_001",
    "offline_002",
    "offline_003",
    "offline_004",
)

RESTORED_VERIFIER_RUNNER_CLAIM_NOTE = (
    "Isolated restored-verifier runner (Phase 12G). Reloaded full-KV verifier with "
    "lossy compressor drafts in an experimental API only — not default runtime, "
    "vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

EXP052_CLAIM_NOTE = (
    "Restored-verifier runner smoke (Phase 12G). Consolidation smoke for the isolated "
    "runner API — not default runtime, vLLM, LMCache, remote prefix runtime, or serving. "
    "No speed, latency, throughput, active memory savings, or production-serving claim."
)

EXP053_CLAIM_NOTE = (
    "Runner-backed restored-verifier drift panel (Phase 12H). Drift-prone panel via "
    "run_restored_verifier() only — not default runtime, vLLM, LMCache, remote prefix "
    "runtime, or serving. No speed, latency, throughput, active memory savings, or "
    "production-serving claim."
)


@dataclass
class RestoredVerifierRunConfig:
    """Configuration for one isolated restored-verifier run."""

    model_id: str = DEFAULT_MODEL
    device: str = "cpu"
    dtype: str = "float32"
    prompt_ids: list[str] = field(default_factory=lambda: list(DEFAULT_SMOKE_PROMPT_IDS))
    storage_backend_name: str = "in_memory_kv_storage"
    compressor_names: list[str] = field(default_factory=lambda: list(DEFAULT_SMOKE_COMPRESSORS))
    draft_len: int = DEFAULT_SMOKE_DRAFT_LEN
    draft_len_values: list[int] | None = None
    max_new_tokens: int = DEFAULT_SMOKE_MAX_NEW_TOKENS
    verifier_source: str = VERIFIER_SOURCE
    claim_note: str = RESTORED_VERIFIER_RUNNER_CLAIM_NOTE
    namespace_prefix: str = "restored_verifier_runner"
    file_storage_root: str | None = None

    def resolved_draft_lens(self) -> list[int]:
        """Return draft lengths to iterate (single or panel)."""
        if self.draft_len_values:
            return list(self.draft_len_values)
        return [self.draft_len]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["draft_len_values"] = self.resolved_draft_lens()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RestoredVerifierRunConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RestoredVerifierCellResult:
    """Per prompt×compressor cell from the restored-verifier runner."""

    prompt_id: str
    compressor_name: str
    storage_backend: str
    draft_len: int
    token_exact_match: bool
    exactkv_failure: int
    accepted_prefix_lengths: list[int]
    first_divergence: int | None = None
    mean_acceptance: float = 0.0
    draft_divergence_count: int = 0
    semantic_divergence_count: int = 0
    category: str = ""
    restore_blocker: str = ""
    draft_blocker: str = ""
    verification_blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RestoredVerifierCellResult:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RestoredVerifierRunReport:
    """Aggregate report from one restored-verifier runner invocation."""

    experiment_id: str
    config: RestoredVerifierRunConfig
    cells: list[RestoredVerifierCellResult]
    total_cells: int
    token_exact_match_count: int
    exactkv_failures: int
    mean_acceptance: float
    draft_divergence_count: int
    blockers: dict[str, list[str]]
    claim_note: str
    semantic_divergence_count: int = 0
    no_real_drift_observed: bool = False
    status: str = "pass"
    first_divergences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "config": self.config.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "total_cells": self.total_cells,
            "token_exact_match_count": self.token_exact_match_count,
            "exactkv_failures": self.exactkv_failures,
            "mean_acceptance": self.mean_acceptance,
            "draft_divergence_count": self.draft_divergence_count,
            "semantic_divergence_count": self.semantic_divergence_count,
            "no_real_drift_observed": self.no_real_drift_observed,
            "blockers": self.blockers,
            "first_divergences": self.first_divergences,
            "claim_note": self.claim_note,
        }


def default_smoke_config(**overrides: Any) -> RestoredVerifierRunConfig:
    """Default Exp 052 smoke configuration."""
    cfg = RestoredVerifierRunConfig(claim_note=EXP052_CLAIM_NOTE, namespace_prefix="exp052")
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def default_panel_prompt_ids(*, full_panel: bool = True) -> list[str]:
    """Drift-targeted prompt ids for runner-backed panel (8–12 prompts)."""
    prompts = default_drift_stress_prompts()
    if full_panel:
        return [p["prompt_id"] for p in prompts]
    return [p["prompt_id"] for p in prompts[:8]]


def default_panel_config(**overrides: Any) -> RestoredVerifierRunConfig:
    """Default Exp 053 runner-backed drift panel configuration."""
    compressors = list(default_drift_stress_compressors()) or list(DEFAULT_PANEL_COMPRESSORS)
    cfg = RestoredVerifierRunConfig(
        prompt_ids=default_panel_prompt_ids(full_panel=True),
        compressor_names=compressors,
        draft_len_values=list(DEFAULT_PANEL_DRAFT_LENS),
        max_new_tokens=DEFAULT_PANEL_MAX_NEW_TOKENS,
        claim_note=EXP053_CLAIM_NOTE,
        namespace_prefix="exp053",
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def resolve_prompt_entries(prompt_ids: list[str]) -> list[dict[str, str]]:
    """Resolve prompt_id list against offline and drift prompt panels."""
    by_id: dict[str, dict[str, str]] = {}
    for entry in default_offline_prompts() + default_drift_stress_prompts():
        by_id[entry["prompt_id"]] = entry
    resolved: list[dict[str, str]] = []
    for pid in prompt_ids:
        if pid not in by_id:
            raise ValueError(f"unknown prompt_id: {pid}")
        entry = by_id[pid]
        resolved.append(
            {
                "prompt_id": entry["prompt_id"],
                "prompt": entry["prompt"],
                "category": entry.get("category", "smoke"),
            }
        )
    return resolved


def build_storage_backend(
    backend_name: str,
    *,
    file_root: Path | None = None,
) -> KVStorageBackend:
    """Construct a storage backend by canonical name."""
    if backend_name == "in_memory_kv_storage":
        return InMemoryKVStorageBackend(backend_name="in_memory_kv_storage")
    if backend_name == "file_kv_storage":
        root = file_root or Path("reports/exp052_kv_files")
        root.mkdir(parents=True, exist_ok=True)
        return FileKVStorageBackend(root, backend_name="file_kv_storage")
    raise ValueError(f"unsupported storage backend: {backend_name}")


def drift_result_to_cell(
    result: OfflineDriftStressCellResult,
    *,
    storage_backend: str,
) -> RestoredVerifierCellResult:
    """Map offline drift-stress cell result to runner cell result."""
    return RestoredVerifierCellResult(
        prompt_id=result.prompt_id,
        compressor_name=result.compressor_name,
        storage_backend=storage_backend,
        draft_len=result.draft_len,
        token_exact_match=result.token_exact_match,
        exactkv_failure=result.exactkv_failures,
        accepted_prefix_lengths=list(result.accepted_prefix_lengths),
        first_divergence=result.first_divergence_idx,
        mean_acceptance=result.mean_acceptance,
        draft_divergence_count=result.draft_divergence_count,
        semantic_divergence_count=result.semantic_divergence_count,
        category=result.category,
        restore_blocker=result.restore_blocker,
        draft_blocker=result.draft_blocker,
        verification_blocker=result.verification_blocker,
    )


def aggregate_blockers(cells: list[RestoredVerifierCellResult]) -> dict[str, list[str]]:
    """Collect restore/draft/verification blockers from cell results."""
    restore: list[str] = []
    draft: list[str] = []
    verification: list[str] = []
    for cell in cells:
        prefix = f"{cell.storage_backend}/{cell.compressor_name}/dl{cell.draft_len}/{cell.prompt_id}"
        if cell.restore_blocker:
            restore.append(f"{prefix}: {cell.restore_blocker}")
        if cell.draft_blocker:
            draft.append(f"{prefix}: {cell.draft_blocker}")
        if cell.verification_blocker:
            verification.append(f"{prefix}: {cell.verification_blocker}")
    return {
        "restore_blockers": restore,
        "draft_blockers": draft,
        "verification_blockers": verification,
    }


def validate_exactness_gate(report: RestoredVerifierRunReport) -> list[str]:
    """Validate exactness gate: all cells must match live full greedy."""
    errors: list[str] = []
    if report.exactkv_failures > 0:
        errors.append(f"exactkv_failures must be 0 for pass; got {report.exactkv_failures}")
    if report.token_exact_match_count + report.exactkv_failures != report.total_cells:
        errors.append(
            "token_exact_match_count + exactkv_failures must equal total_cells"
        )
    for cell in report.cells:
        if cell.exactkv_failure > 0 and cell.token_exact_match:
            errors.append(
                f"cell {cell.prompt_id}/{cell.compressor_name} has exactkv_failure "
                "but token_exact_match true"
            )
        if cell.exactkv_failure == 0 and not cell.token_exact_match:
            errors.append(
                f"cell {cell.prompt_id}/{cell.compressor_name} missing token_exact_match"
            )
    return errors


def check_phase12f_exactness_gate(
    report_path: Path | None = None,
) -> tuple[bool, str]:
    """Check Phase 12F report; block Phase 12G if CUDA panel had exactness failures."""
    path = report_path or DEFAULT_EXP051_REPORT
    if not path.is_file():
        return True, "Phase 12F report not found; using Phase 12E CPU evidence"
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    failures = int(data.get("exactkv_failures", 0))
    if failures > 0 and cells:
        return False, f"Phase 12F exactness failures: {failures}"
    if data.get("status") == "failed":
        return False, "Phase 12F status failed"
    if not data.get("cuda_available", False) and not cells:
        return True, "Phase 12F skipped (CUDA unavailable); using Phase 12E CPU evidence"
    if cells and failures == 0:
        return True, "Phase 12F passed; used as additional CUDA evidence"
    return True, "Phase 12F present; no exactness blockers"


def run_restored_verifier_cell(
    runtime: ModelRuntime,
    *,
    config: RestoredVerifierRunConfig,
    prompt_id: str,
    prompt: str,
    category: str,
    backend: KVStorageBackend,
    compressor_name: str,
) -> RestoredVerifierCellResult:
    """Run one restored-verifier cell via existing offline drift-stress helper."""
    storage_key = f"{prompt_id}__{compressor_name}__dl{config.draft_len}"
    drift_result = run_offline_drift_stress_cell(
        runtime,
        prompt_id=prompt_id,
        prompt=prompt,
        category=category,
        backend=backend,
        compressor_name=compressor_name,
        draft_len=config.draft_len,
        max_new_tokens=config.max_new_tokens,
        namespace_prefix=config.namespace_prefix,
        storage_key=storage_key,
    )
    return drift_result_to_cell(
        drift_result,
        storage_backend=config.storage_backend_name,
    )


def run_restored_verifier(
    config: RestoredVerifierRunConfig,
    *,
    experiment_id: str = EXPERIMENT_052_ID,
    extra_backends: list[str] | None = None,
) -> RestoredVerifierRunReport:
    """Run the full restored-verifier panel for the given config."""
    allowed, gate_reason = check_phase12f_exactness_gate()
    if not allowed:
        raise RuntimeError(f"Phase 12G blocked by Phase 12F exactness gate: {gate_reason}")

    prompts = resolve_prompt_entries(config.prompt_ids)
    backends_to_run = [config.storage_backend_name]
    if extra_backends:
        for name in extra_backends:
            if name not in backends_to_run:
                backends_to_run.append(name)

    runtime = ModelRuntime(config.model_id, device=config.device, dtype=config.dtype)
    cells: list[RestoredVerifierCellResult] = []
    per_cell_mean: list[float] = []
    total_draft_div = 0
    total_semantic_div = 0
    first_divergences: list[dict[str, Any]] = []
    draft_lens = config.resolved_draft_lens()

    try:
        for backend_name in backends_to_run:
            file_root = Path(config.file_storage_root) if config.file_storage_root else None
            backend = build_storage_backend(backend_name, file_root=file_root)
            for draft_len in draft_lens:
                run_cfg = RestoredVerifierRunConfig(
                    **{**config.to_dict(), "storage_backend_name": backend_name, "draft_len": draft_len}
                )
                for entry in prompts:
                    for compressor_name in config.compressor_names:
                        cell = run_restored_verifier_cell(
                            runtime,
                            config=run_cfg,
                            prompt_id=entry["prompt_id"],
                            prompt=entry["prompt"],
                            category=entry["category"],
                            backend=backend,
                            compressor_name=compressor_name,
                        )
                        cells.append(cell)
                        per_cell_mean.append(cell.mean_acceptance)
                        total_draft_div += cell.draft_divergence_count
                        total_semantic_div += cell.semantic_divergence_count
                        if cell.exactkv_failure > 0 and cell.first_divergence is not None:
                            first_divergences.append(
                                {
                                    "prompt_id": cell.prompt_id,
                                    "compressor_name": cell.compressor_name,
                                    "storage_backend": cell.storage_backend,
                                    "draft_len": cell.draft_len,
                                    "first_divergence_idx": cell.first_divergence,
                                }
                            )
    finally:
        del runtime

    exact_matches = sum(1 for c in cells if c.token_exact_match)
    failures = sum(c.exactkv_failure for c in cells)
    mean_acc = sum(per_cell_mean) / len(per_cell_mean) if per_cell_mean else 0.0
    blockers = aggregate_blockers(cells)
    status = "pass" if failures == 0 else "failed"

    report = RestoredVerifierRunReport(
        experiment_id=experiment_id,
        config=config,
        cells=cells,
        total_cells=len(cells),
        token_exact_match_count=exact_matches,
        exactkv_failures=failures,
        mean_acceptance=mean_acc,
        draft_divergence_count=total_draft_div,
        semantic_divergence_count=total_semantic_div,
        no_real_drift_observed=total_draft_div == 0,
        blockers=blockers,
        claim_note=config.claim_note,
        status=status,
        first_divergences=first_divergences,
    )
    gate_errors = validate_exactness_gate(report)
    if gate_errors and failures == 0:
        report.status = "failed"
    return report


def report_to_exp052_json(report: RestoredVerifierRunReport) -> dict[str, Any]:
    """Serialize runner report to Exp 052 JSON schema."""
    cfg = report.config
    blockers = report.blockers
    return {
        "experiment_id": EXPERIMENT_052_ID,
        "status": report.status,
        "model": cfg.model_id,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "prompt_count": len(cfg.prompt_ids),
        "storage_backends": sorted({c.storage_backend for c in report.cells}),
        "storage_backend": cfg.storage_backend_name,
        "compressor_names": list(cfg.compressor_names),
        "draft_len": cfg.draft_len,
        "max_new_tokens": cfg.max_new_tokens,
        "verifier_source": cfg.verifier_source,
        "cells": [c.to_dict() for c in report.cells],
        "total_cells": report.total_cells,
        "exactkv_failures": report.exactkv_failures,
        "token_exact_match_count": report.token_exact_match_count,
        "mean_acceptance": report.mean_acceptance,
        "draft_divergence_count": report.draft_divergence_count,
        "first_divergences": report.first_divergences,
        "restore_blockers": blockers.get("restore_blockers", []),
        "draft_blockers": blockers.get("draft_blockers", []),
        "verification_blockers": blockers.get("verification_blockers", []),
        "blockers": blockers,
        "claim_note": report.claim_note,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "config": cfg.to_dict(),
    }


def validate_exp052_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 052 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backend",
        "compressor_names",
        "draft_len",
        "max_new_tokens",
        "verifier_source",
        "exactkv_failures",
        "token_exact_match_count",
        "mean_acceptance",
        "draft_divergence_count",
        "first_divergences",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_052_ID:
        errors.append("experiment_id must be exp052_restored_verifier_runner_smoke")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    cells = report.get("cells", [])
    exact_count = int(report.get("token_exact_match_count", -1))
    failures = int(report.get("exactkv_failures", -1))
    if exact_count >= 0 and failures >= 0 and exact_count + failures != len(cells):
        errors.append("token_exact_match_count + exactkv_failures must equal len(cells)")
    draft_div = report.get("draft_divergence_count")
    if not isinstance(draft_div, int) or draft_div < 0:
        errors.append("draft_divergence_count must be a non-negative int")
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    for cell in cells:
        for field_name in (
            "prompt_id",
            "compressor_name",
            "storage_backend",
            "draft_len",
            "token_exact_match",
            "exactkv_failure",
            "accepted_prefix_lengths",
        ):
            if field_name not in cell:
                errors.append(f"cells missing field: {field_name}")
        div = cell.get("first_divergence")
        if div is not None and not isinstance(div, int):
            errors.append("first_divergence must be int or null")
    return errors


def report_to_exp053_json(report: RestoredVerifierRunReport) -> dict[str, Any]:
    """Serialize runner report to Exp 053 JSON schema."""
    cfg = report.config
    blockers = report.blockers
    return {
        "experiment_id": EXPERIMENT_053_ID,
        "status": report.status,
        "config": cfg.to_dict(),
        "model": cfg.model_id,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "prompt_count": len(cfg.prompt_ids),
        "storage_backends": sorted({c.storage_backend for c in report.cells}),
        "compressor_names": list(cfg.compressor_names),
        "draft_len_values": cfg.resolved_draft_lens(),
        "max_new_tokens": cfg.max_new_tokens,
        "verifier_source": cfg.verifier_source,
        "cells": [c.to_dict() for c in report.cells],
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
        "claim_note": report.claim_note,
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
    }


def validate_exp053_report(report: dict[str, Any]) -> list[str]:
    """Validate experiment 053 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "config",
        "model",
        "device",
        "dtype",
        "prompt_count",
        "storage_backends",
        "compressor_names",
        "draft_len_values",
        "max_new_tokens",
        "verifier_source",
        "total_cells",
        "exactkv_failures",
        "token_exact_match_count",
        "mean_acceptance",
        "draft_divergence_count",
        "semantic_divergence_count",
        "no_real_drift_observed",
        "first_divergences",
        "restore_blockers",
        "draft_blockers",
        "verification_blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_053_ID:
        errors.append("experiment_id must be exp053_restored_verifier_runner_panel")
    if report.get("verifier_source") != VERIFIER_SOURCE:
        errors.append("verifier_source must be reloaded_full_kv")
    if not isinstance(report.get("config"), dict):
        errors.append("config must be a dict")
    if not isinstance(report.get("no_real_drift_observed"), bool):
        errors.append("no_real_drift_observed must be a bool")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    cells = report.get("cells", [])
    exact_count = int(report.get("token_exact_match_count", -1))
    failures = int(report.get("exactkv_failures", -1))
    total = int(report.get("total_cells", -1))
    if total >= 0 and total != len(cells):
        errors.append("total_cells must equal len(cells)")
    if exact_count >= 0 and failures >= 0 and exact_count + failures != len(cells):
        errors.append("token_exact_match_count + exactkv_failures must equal len(cells)")
    draft_div = report.get("draft_divergence_count")
    if not isinstance(draft_div, int) or draft_div < 0:
        errors.append("draft_divergence_count must be a non-negative int")
    semantic_div = report.get("semantic_divergence_count")
    if not isinstance(semantic_div, int) or semantic_div < 0:
        errors.append("semantic_divergence_count must be a non-negative int")
    if bool(report.get("no_real_drift_observed")) and int(draft_div or -1) > 0:
        errors.append("no_real_drift_observed cannot be true when draft_divergence_count > 0")
    mean_acc = report.get("mean_acceptance")
    if not isinstance(mean_acc, (int, float)):
        errors.append("mean_acceptance must be numeric")
    for cell in cells:
        for field_name in (
            "prompt_id",
            "compressor_name",
            "storage_backend",
            "draft_len",
            "token_exact_match",
            "exactkv_failure",
            "accepted_prefix_lengths",
            "draft_divergence_count",
        ):
            if field_name not in cell:
                errors.append(f"cells missing field: {field_name}")
        div = cell.get("first_divergence")
        if div is not None and not isinstance(div, int):
            errors.append("first_divergence must be int or null")
    return errors
