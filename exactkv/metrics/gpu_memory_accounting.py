"""GPU memory accounting diagnostics for Experiment 057 (Phase 14B).

Records active CUDA memory observations at defined lifecycle points for the
explicit experimental restored-verifier runtime path. **Not** a memory savings
claim and **not** a performance benchmark.
"""
from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

from exactkv.cache.dual_cache import CacheResidency
from exactkv.cache.hf_kv_restore import (
    DEFAULT_MODEL,
    FORBIDDEN_CLAIMS,
    build_storage_payload_from_cache,
    capture_prefill_kv,
    payload_byte_summary,
    store_prefill_payload,
)
from exactkv.cache.offline_verifier import (
    VERIFIER_SOURCE,
    configure_cuda_determinism,
    resolve_cuda_drift_dtype_configs,
)
from exactkv.cache.restored_verifier_runner import (
    DEFAULT_SMOKE_COMPRESSORS,
    DEFAULT_SMOKE_DRAFT_LEN,
    DEFAULT_SMOKE_PROMPT_IDS,
)
from exactkv.cache.storage import InMemoryKVStorageBackend, KVStorageHandle
from exactkv.runtime.experimental import (
    EXP056_CLAIM_NOTE,
    RUNTIME_PATH_EXPERIMENTAL,
    ExperimentalRestoredVerifierConfig,
    ExperimentalRuntimeMode,
    default_cuda_gate_experimental_config,
    run_experimental_restored_verifier,
)
from exactkv.runtime.generation import generate_full_greedy
from exactkv.runtime.model_runtime import ModelRuntime

T = TypeVar("T")

EXPERIMENT_057_ID = "exp057_gpu_memory_accounting"
DEFAULT_EXP056_REPORT = Path("reports/experiment_056_cuda_restored_verifier_runtime_gate.json")
DEFAULT_EXP057_REPORT = Path("reports/experiment_057_gpu_memory_accounting.json")
DEFAULT_MEMORY_PROMPT_IDS = DEFAULT_SMOKE_PROMPT_IDS[:2]
DEFAULT_MAX_NEW_TOKENS = 12
DEFAULT_DRAFT_LEN = DEFAULT_SMOKE_DRAFT_LEN

EXP057_CLAIM_NOTE = (
    "GPU memory accounting diagnostic for explicit experimental restored-verifier "
    "runtime (Phase 14B). Diagnostic measurements only — active GPU memory savings "
    "are not claimed. Restored full KV used only when explicitly enabled; default "
    "ExactKV generation unchanged. Not vLLM, LMCache, remote prefix runtime, or "
    "serving. No speed, latency, throughput, active memory savings, or "
    "production-serving claim. Stored/offloaded payload accounting is separate from "
    "active GPU peak memory."
)

_FORBIDDEN_MEASUREMENT_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "wall_time_seconds",
})


@dataclass(frozen=True)
class CudaMemorySnapshot:
    """Point-in-time CUDA memory observation."""

    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int
    device: str
    dtype: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CudaMemoryMeasurement:
    """Memory delta for one labeled CUDA region."""

    label: str
    before: CudaMemorySnapshot
    after: CudaMemorySnapshot
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    delta_allocated_bytes: int
    delta_reserved_bytes: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "delta_allocated_bytes": self.delta_allocated_bytes,
            "delta_reserved_bytes": self.delta_reserved_bytes,
            "notes": self.notes,
        }


@dataclass
class GpuMemoryAccountingReport:
    """Aggregate Exp 057 GPU memory accounting report."""

    experiment_id: str = EXPERIMENT_057_ID
    status: str = "blocked"
    cuda_available: bool = False
    device_name: str = ""
    torch_version: str = ""
    model_id: str = DEFAULT_MODEL
    runtime_path: str = RUNTIME_PATH_EXPERIMENTAL
    dtype_configs: list[str] = field(default_factory=list)
    prompt_count: int = 0
    storage_backend: str = "in_memory_kv_storage"
    compressors: list[str] = field(default_factory=lambda: list(DEFAULT_SMOKE_COMPRESSORS))
    verifier_source: str = VERIFIER_SOURCE
    exactness_gate_passed: bool = False
    exp056_gate_passed: bool = False
    exactkv_failures: int = 0
    token_exact_match_count: int = 0
    total_cells: int = 0
    measurements: list[CudaMemoryMeasurement] = field(default_factory=list)
    full_kv_payload_bytes: int = 0
    stored_kv_payload_bytes: int = 0
    blockers: list[str] = field(default_factory=list)
    report_note: str = EXP057_CLAIM_NOTE
    forbidden_claims: list[str] = field(default_factory=lambda: list(FORBIDDEN_CLAIMS))
    active_gpu_memory_savings_claimed: bool = False
    speedup_claimed: bool = False
    throughput_claimed: bool = False
    latency_claimed: bool = False
    production_serving_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        by_label = {m.label: m.to_dict() for m in self.measurements}
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "model_id": self.model_id,
            "runtime_path": self.runtime_path,
            "dtype_configs": list(self.dtype_configs),
            "prompt_count": self.prompt_count,
            "storage_backend": self.storage_backend,
            "compressors": list(self.compressors),
            "verifier_source": self.verifier_source,
            "exactness_gate_passed": self.exactness_gate_passed,
            "exp056_gate_passed": self.exp056_gate_passed,
            "exactkv_failures": self.exactkv_failures,
            "token_exact_match_count": self.token_exact_match_count,
            "total_cells": self.total_cells,
            "measurements": [m.to_dict() for m in self.measurements],
            "model_loaded": by_label.get("model_loaded"),
            "full_greedy": by_label.get("full_greedy"),
            "kv_capture_store_reload": by_label.get("kv_capture_store_reload"),
            "restored_verifier_runtime": by_label.get("restored_verifier_runtime"),
            "full_kv_payload_bytes": self.full_kv_payload_bytes,
            "stored_kv_payload_bytes": self.stored_kv_payload_bytes,
            "active_gpu_memory_savings_claimed": self.active_gpu_memory_savings_claimed,
            "speedup_claimed": self.speedup_claimed,
            "throughput_claimed": self.throughput_claimed,
            "latency_claimed": self.latency_claimed,
            "production_serving_claimed": self.production_serving_claimed,
            "blockers": list(self.blockers),
            "claim_note": self.report_note,
            "forbidden_claims": list(self.forbidden_claims),
            "report_note": self.report_note,
        }


def _device_index(device: str | None) -> int:
    dev = device or "cuda"
    if dev == "cuda":
        return 0
    return int(dev.split(":")[-1])


def cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def synchronize_cuda(device: str | None = None) -> None:
    """Synchronize CUDA device before/after measurements."""
    if cuda_available():
        torch.cuda.synchronize(_device_index(device))


def reset_cuda_peak_memory(device: str | None = None) -> None:
    """Reset peak memory counters for a fresh measurement window."""
    if cuda_available():
        idx = _device_index(device)
        gc.collect()
        torch.cuda.empty_cache()
        synchronize_cuda(device)
        torch.cuda.reset_peak_memory_stats(idx)


def snapshot_cuda_memory(
    label: str,
    *,
    device: str | None = None,
    dtype: str = "",
) -> CudaMemorySnapshot:
    """Capture current CUDA memory stats."""
    if not cuda_available():
        raise RuntimeError("CUDA unavailable — cannot snapshot GPU memory")
    dev = device or "cuda"
    idx = _device_index(dev)
    synchronize_cuda(dev)
    return CudaMemorySnapshot(
        allocated_bytes=int(torch.cuda.memory_allocated(idx)),
        reserved_bytes=int(torch.cuda.memory_reserved(idx)),
        max_allocated_bytes=int(torch.cuda.max_memory_allocated(idx)),
        max_reserved_bytes=int(torch.cuda.max_memory_reserved(idx)),
        device=dev,
        dtype=dtype,
        label=label,
    )


def measure_cuda_memory(
    label: str,
    fn: Callable[[], T],
    *,
    device: str | None = None,
    dtype: str = "",
    notes: str = "",
) -> tuple[T, CudaMemoryMeasurement]:
    """Run ``fn`` inside a synchronized CUDA peak-memory window."""
    if not cuda_available():
        raise RuntimeError(f"CUDA unavailable — cannot measure {label!r}")
    dev = device or "cuda"
    idx = _device_index(dev)
    reset_cuda_peak_memory(dev)
    before = snapshot_cuda_memory(f"{label}_before", device=dev, dtype=dtype)
    result = fn()
    synchronize_cuda(dev)
    after = snapshot_cuda_memory(f"{label}_after", device=dev, dtype=dtype)
    peak_alloc = int(torch.cuda.max_memory_allocated(idx))
    peak_reserved = int(torch.cuda.max_memory_reserved(idx))
    measurement = CudaMemoryMeasurement(
        label=label,
        before=before,
        after=after,
        peak_allocated_bytes=peak_alloc,
        peak_reserved_bytes=peak_reserved,
        delta_allocated_bytes=after.allocated_bytes - before.allocated_bytes,
        delta_reserved_bytes=after.reserved_bytes - before.reserved_bytes,
        notes=notes,
    )
    return result, measurement


def check_exp056_exactness_gate(
    report_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Verify Exp 056 CUDA exactness gate passed before memory diagnostics."""
    path = report_path or DEFAULT_EXP056_REPORT
    blockers: list[str] = []
    if not path.is_file():
        return False, ["Exp 056 report not found — run Exp 056 CUDA gate first"]
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("experiment_id") != "exp056_cuda_restored_verifier_runtime_gate":
        blockers.append("Exp 056 report experiment_id mismatch")
    if not data.get("cuda_available"):
        blockers.append("Exp 056 cuda_available is false")
    if data.get("status") != "pass":
        blockers.append(f"Exp 056 status is {data.get('status')!r}, expected pass")
    total = int(data.get("total_cells", 0))
    if total <= 0:
        blockers.append("Exp 056 total_cells is zero")
    failures = int(data.get("exactkv_failures", 0))
    if failures > 0:
        blockers.append(f"Exp 056 exactkv_failures={failures}")
    exact = int(data.get("token_exact_match_count", 0))
    if exact != total:
        blockers.append(
            f"Exp 056 token_exact_match_count={exact} != total_cells={total}"
        )
    for key in ("cuda_blockers", "restore_blockers", "draft_blockers", "verification_blockers"):
        items = data.get(key) or []
        if items:
            blockers.append(f"Exp 056 {key}: {items}")
    return len(blockers) == 0, blockers


def _resolve_prompt_entries(prompt_ids: list[str]) -> list[dict[str, str]]:
    from exactkv.cache.restored_verifier_runner import resolve_prompt_entries

    return resolve_prompt_entries(prompt_ids)


def _measurement_summary(measurement: CudaMemoryMeasurement | None) -> dict[str, Any] | None:
    return measurement.to_dict() if measurement is not None else None


def _run_dtype_memory_panel(
    *,
    model_id: str,
    dtype: str,
    prompt_ids: list[str],
    compressor_names: list[str],
    max_new_tokens: int,
    draft_len: int,
) -> tuple[list[CudaMemoryMeasurement], int, int, int, int, int, list[str]]:
    """Run memory measurements and exactness-gated restored-verifier panel for one dtype."""
    measurements: list[CudaMemoryMeasurement] = []
    blockers: list[str] = []
    full_kv_bytes = 0
    stored_kv_bytes = 0
    configure_cuda_determinism()

    def load_runtime() -> ModelRuntime:
        return ModelRuntime(model_id, device="cuda", dtype=dtype)

    runtime, model_loaded = measure_cuda_memory(
        "model_loaded",
        load_runtime,
        dtype=dtype,
        notes="Model weights loaded on CUDA; no generation yet.",
    )
    measurements.append(model_loaded)

    prompts = _resolve_prompt_entries(prompt_ids)
    first_prompt = prompts[0]["prompt"]
    first_prompt_id = prompts[0]["prompt_id"]

    def run_full_greedy() -> None:
        generate_full_greedy(runtime, first_prompt, max_new_tokens)

    _, full_greedy = measure_cuda_memory(
        "full_greedy",
        run_full_greedy,
        dtype=dtype,
        notes="Live full greedy reference generation on CUDA.",
    )
    measurements.append(full_greedy)

    backend = InMemoryKVStorageBackend()

    def run_kv_roundtrip() -> None:
        nonlocal full_kv_bytes, stored_kv_bytes
        capture = capture_prefill_kv(runtime, first_prompt)
        payload = build_storage_payload_from_cache(capture)
        full_kv_bytes = payload_byte_summary(payload)
        handle = KVStorageHandle(
            namespace=f"exp057/{dtype}/{model_id}/in_memory_kv_storage",
            key=first_prompt_id,
            version="1",
        )
        store_prefill_payload(backend, handle, payload, residency=CacheResidency.CPU)
        stored = backend.get(handle)
        stored_kv_bytes = int(stored.metadata.total_payload_bytes)
        loaded = backend.get(handle).payload
        from exactkv.cache.hf_kv_restore import restore_cache_from_storage_payload

        restore_cache_from_storage_payload(loaded, device=runtime.device)

    _, kv_capture = measure_cuda_memory(
        "kv_capture_store_reload",
        run_kv_roundtrip,
        dtype=dtype,
        notes="Prefill capture, in-memory store, reload into FullKVState.",
    )
    measurements.append(kv_capture)

    exp_cfg = default_cuda_gate_experimental_config(
        dtype,
        model_id=model_id,
        prompt_ids=prompt_ids,
        compressor_names=compressor_names,
        max_new_tokens=max_new_tokens,
        draft_lens=[draft_len],
        claim_note=EXP057_CLAIM_NOTE,
        namespace_prefix=f"exp057/{dtype}",
    )

    def run_restored_verifier_runtime() -> Any:
        return run_experimental_restored_verifier(
            exp_cfg,
            experiment_id=EXPERIMENT_057_ID,
        )

    runtime_result, restored = measure_cuda_memory(
        "restored_verifier_runtime",
        run_restored_verifier_runtime,
        dtype=dtype,
        notes=(
            "Explicit experimental restored-verifier runtime via "
            "run_experimental_restored_verifier(). Diagnostic peak only."
        ),
    )
    measurements.append(restored)

    failures = 0
    exact = 0
    total = 0
    if runtime_result.runner_report is not None:
        report = runtime_result.runner_report
        failures = report.exactkv_failures
        exact = report.token_exact_match_count
        total = report.total_cells
        if failures > 0:
            blockers.append(f"{dtype}: exactkv_failures={failures}")

    for compressor in compressor_names:
        single_cfg = ExperimentalRestoredVerifierConfig(
            enabled=True,
            mode=ExperimentalRuntimeMode.RESTORED_VERIFIER_OFFLINE,
            model_id=model_id,
            device="cuda",
            dtype=dtype,
            prompt_ids=[prompt_ids[0]],
            storage_backends=["in_memory_kv_storage"],
            compressor_names=[compressor],
            draft_lens=[draft_len],
            max_new_tokens=max_new_tokens,
            verifier_source=VERIFIER_SOURCE,
            claim_note=EXP057_CLAIM_NOTE,
            namespace_prefix=f"exp057/{dtype}/{compressor}",
        )

        def run_single_compressor(cfg=single_cfg) -> None:
            result = run_experimental_restored_verifier(cfg, experiment_id=EXPERIMENT_057_ID)
            if result.runner_report and result.runner_report.exactkv_failures > 0:
                raise RuntimeError(
                    f"exactness failure for compressor {compressor}: "
                    f"{result.runner_report.exactkv_failures}"
                )

        label = f"restored_verifier_{compressor}"
        try:
            _, comp_meas = measure_cuda_memory(
                label,
                run_single_compressor,
                dtype=dtype,
                notes=f"Per-compressor restored-verifier memory ({compressor}).",
            )
            measurements.append(comp_meas)
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"{dtype}/{compressor}: {type(exc).__name__}: {exc}")

    del runtime
    gc.collect()
    if cuda_available():
        torch.cuda.empty_cache()
    return measurements, full_kv_bytes, stored_kv_bytes, failures, exact, total, blockers


def run_gpu_memory_accounting(
    *,
    model_id: str = DEFAULT_MODEL,
    prompt_ids: list[str] | None = None,
    compressor_names: list[str] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    draft_len: int = DEFAULT_DRAFT_LEN,
    exp056_report_path: Path | None = None,
    require_exp056_gate: bool = True,
) -> GpuMemoryAccountingReport:
    """Run Exp 057 GPU memory accounting diagnostic panel."""
    prompt_ids = prompt_ids or list(DEFAULT_MEMORY_PROMPT_IDS)
    compressor_names = compressor_names or list(DEFAULT_SMOKE_COMPRESSORS)
    exp056_ok, exp056_blockers = check_exp056_exactness_gate(exp056_report_path)
    if require_exp056_gate and not exp056_ok:
        return GpuMemoryAccountingReport(
            status="blocked",
            cuda_available=cuda_available(),
            prompt_count=len(prompt_ids),
            compressors=list(compressor_names),
            exp056_gate_passed=False,
            blockers=list(exp056_blockers),
        )

    if not cuda_available():
        return GpuMemoryAccountingReport(
            status="blocked",
            exp056_gate_passed=exp056_ok,
            prompt_count=len(prompt_ids),
            compressors=list(compressor_names),
            blockers=["CUDA unavailable"],
        )

    device_name = torch.cuda.get_device_name(0)
    dtype_entries = resolve_cuda_drift_dtype_configs()
    tested_dtypes: list[str] = []
    all_measurements: list[CudaMemoryMeasurement] = []
    all_blockers: list[str] = []
    full_kv_bytes = 0
    stored_kv_bytes = 0
    total_failures = 0
    total_exact = 0
    total_cells = 0

    for entry in dtype_entries:
        if entry.status == "skipped":
            all_blockers.append(f"skipped dtype {entry.dtype}: {entry.skip_reason}")
            continue
        try:
            meas, f_bytes, s_bytes, failures, exact, cells, blockers = _run_dtype_memory_panel(
                model_id=model_id,
                dtype=entry.dtype,
                prompt_ids=prompt_ids,
                compressor_names=compressor_names,
                max_new_tokens=max_new_tokens,
                draft_len=draft_len,
            )
        except Exception as exc:  # noqa: BLE001
            all_blockers.append(f"{entry.dtype}: {type(exc).__name__}: {exc}")
            continue
        tested_dtypes.append(entry.dtype)
        all_measurements.extend(meas)
        full_kv_bytes = max(full_kv_bytes, f_bytes)
        stored_kv_bytes = max(stored_kv_bytes, s_bytes)
        total_failures += failures
        total_exact += exact
        total_cells += cells
        all_blockers.extend(blockers)

    if not tested_dtypes:
        return GpuMemoryAccountingReport(
            status="blocked",
            cuda_available=True,
            device_name=device_name,
            torch_version=torch.__version__,
            model_id=model_id,
            prompt_count=len(prompt_ids),
            compressors=list(compressor_names),
            exp056_gate_passed=exp056_ok,
            measurements=all_measurements,
            blockers=all_blockers or ["no CUDA dtype configs could be tested"],
        )

    exactness_ok = total_failures == 0 and (total_cells == 0 or total_exact == total_cells)
    status = "pass" if exactness_ok and not all_blockers else "failed"
    if exactness_ok and all_blockers:
        status = "failed"

    return GpuMemoryAccountingReport(
        status=status,
        cuda_available=True,
        device_name=device_name,
        torch_version=torch.__version__,
        model_id=model_id,
        dtype_configs=tested_dtypes,
        prompt_count=len(prompt_ids),
        compressors=list(compressor_names),
        exactness_gate_passed=exactness_ok,
        exp056_gate_passed=exp056_ok,
        exactkv_failures=total_failures,
        token_exact_match_count=total_exact,
        total_cells=total_cells,
        measurements=all_measurements,
        full_kv_payload_bytes=full_kv_bytes,
        stored_kv_payload_bytes=stored_kv_bytes,
        blockers=all_blockers,
    )


def validate_exp057_report(report: dict[str, Any]) -> list[str]:
    """Validate Exp 057 JSON report schema."""
    errors: list[str] = []
    required = (
        "experiment_id",
        "cuda_available",
        "device_name",
        "torch_version",
        "model_id",
        "dtype_configs",
        "prompt_count",
        "storage_backend",
        "compressors",
        "verifier_source",
        "exactness_gate_passed",
        "exactkv_failures",
        "token_exact_match_count",
        "measurements",
        "full_kv_payload_bytes",
        "stored_kv_payload_bytes",
        "active_gpu_memory_savings_claimed",
        "speedup_claimed",
        "throughput_claimed",
        "latency_claimed",
        "production_serving_claimed",
        "blockers",
        "claim_note",
        "forbidden_claims",
    )
    for key in required:
        if key not in report:
            errors.append(f"missing report field: {key}")
    if report.get("experiment_id") != EXPERIMENT_057_ID:
        errors.append("experiment_id must be exp057_gpu_memory_accounting")
    for flag in (
        "active_gpu_memory_savings_claimed",
        "speedup_claimed",
        "throughput_claimed",
        "latency_claimed",
        "production_serving_claimed",
    ):
        if report.get(flag) is not False:
            errors.append(f"{flag} must be false")
    measurements = report.get("measurements", [])
    if not isinstance(measurements, list):
        errors.append("measurements must be a list")
    else:
        for i, meas in enumerate(measurements):
            for field_name in (
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "delta_allocated_bytes",
                "delta_reserved_bytes",
            ):
                val = meas.get(field_name)
                if not isinstance(val, int) or val < 0:
                    errors.append(f"measurements[{i}].{field_name} must be non-negative int")
            for forbidden_field in _FORBIDDEN_MEASUREMENT_FIELDS:
                if forbidden_field in meas:
                    errors.append(f"forbidden measurement field {forbidden_field!r} in measurements[{i}]")
    payload_fields = ("full_kv_payload_bytes", "stored_kv_payload_bytes")
    for key in payload_fields:
        val = report.get(key)
        if not isinstance(val, int) or val < 0:
            errors.append(f"{key} must be a non-negative int")
    forbidden = report.get("forbidden_claims", [])
    for term in FORBIDDEN_CLAIMS:
        if term not in forbidden:
            errors.append(f"forbidden_claims must include: {term}")
    if not report.get("claim_note", "").strip():
        errors.append("claim_note required")
    return errors


def assert_report_claim_safe(report: dict[str, Any]) -> None:
    """Reject forbidden performance claim fields in Exp 057 artifacts."""
    for key in report:
        if key in _FORBIDDEN_MEASUREMENT_FIELDS:
            raise ValueError(f"Forbidden field {key!r} in Exp 057 report")
