"""GPU memory isolation helpers for Experiment 031 (V13 Phase 4).

Records PyTorch CUDA allocation observations at defined lifecycle points.
These fields are **not** part of the standard ExactKV report schema.

Diagnostic GPU memory isolation only — not a performance benchmark.
"""
from __future__ import annotations

import gc
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
    "wall_time_seconds",
})


@dataclass(frozen=True)
class CudaMemoryReading:
    """CUDA allocation observation for one measured trial."""

    allocated_before: int
    reserved_before: int
    peak_allocated: int
    peak_reserved: int
    allocated_after: int
    reserved_after: int
    delta_peak_allocated_vs_model_loaded: int
    delta_peak_reserved_vs_model_loaded: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def require_cuda(device: str) -> None:
    if not cuda_available():
        raise RuntimeError(
            "CUDA required for Experiment 031 GPU memory isolation. "
            "Do not produce final memory conclusions from CPU."
        )
    if device != "cuda" and not device.startswith("cuda:"):
        raise RuntimeError(f"Experiment 031 requires CUDA device; got {device!r}")


def _device_index(device: str) -> int:
    if device == "cuda":
        return 0
    return int(device.split(":")[-1])


def collect_gpu_memory_environment(device: str) -> dict[str, Any]:
    """Record GPU/torch metadata for Exp 031 manifests."""
    import socket

    import torch
    import transformers

    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    if torch.cuda.is_available():
        idx = _device_index(device)
        props = torch.cuda.get_device_properties(idx)
        meta["cuda_version"] = torch.version.cuda
        meta["gpu_device_name"] = props.name
        meta["gpu_total_memory_bytes"] = int(props.total_memory)
    return meta


def _sync(device: str) -> None:
    import torch

    if torch.cuda.is_available():
        idx = _device_index(device)
        torch.cuda.synchronize(idx)


def _allocated_bytes(device: str) -> int:
    import torch

    _sync(device)
    idx = _device_index(device)
    return int(torch.cuda.memory_allocated(idx))


def _reserved_bytes(device: str) -> int:
    import torch

    _sync(device)
    idx = _device_index(device)
    return int(torch.cuda.memory_reserved(idx))


def _peak_allocated_bytes(device: str) -> int:
    import torch

    _sync(device)
    idx = _device_index(device)
    return int(torch.cuda.max_memory_allocated(idx))


def _peak_reserved_bytes(device: str) -> int:
    import torch

    _sync(device)
    idx = _device_index(device)
    return int(torch.cuda.max_memory_reserved(idx))


def prepare_measurement(device: str) -> None:
    """Conservative reset before a measured region."""
    gc.collect()
    import torch

    idx = _device_index(device)
    torch.cuda.empty_cache()
    _sync(device)
    torch.cuda.reset_peak_memory_stats(idx)


def measure_model_loaded_baseline(device: str) -> dict[str, int]:
    """Snapshot after model load, before generation."""
    prepare_measurement(device)
    return {
        "allocated_bytes": _allocated_bytes(device),
        "reserved_bytes": _reserved_bytes(device),
        "peak_allocated_bytes": _peak_allocated_bytes(device),
        "peak_reserved_bytes": _peak_reserved_bytes(device),
    }


def measure_cuda_region(
    device: str,
    fn: Callable[[], T],
    *,
    model_loaded_allocated: int,
    model_loaded_reserved: int,
) -> tuple[T, CudaMemoryReading]:
    """Run ``fn`` inside a synchronized CUDA memory measurement window."""
    prepare_measurement(device)
    before_alloc = _allocated_bytes(device)
    before_reserved = _reserved_bytes(device)

    _sync(device)
    result = fn()
    _sync(device)

    peak_alloc = _peak_allocated_bytes(device)
    peak_reserved = _peak_reserved_bytes(device)
    after_alloc = _allocated_bytes(device)
    after_reserved = _reserved_bytes(device)

    reading = CudaMemoryReading(
        allocated_before=before_alloc,
        reserved_before=before_reserved,
        peak_allocated=peak_alloc,
        peak_reserved=peak_reserved,
        allocated_after=after_alloc,
        reserved_after=after_reserved,
        delta_peak_allocated_vs_model_loaded=peak_alloc - model_loaded_allocated,
        delta_peak_reserved_vs_model_loaded=peak_reserved - model_loaded_reserved,
    )
    return result, reading


def summarize_memory_trials(
    trials: list[CudaMemoryReading],
) -> dict[str, float | int]:
    """Mean/max peak allocated/reserved across repeated trials."""
    if not trials:
        return {
            "num_trials": 0,
            "mean_peak_allocated": 0,
            "max_peak_allocated": 0,
            "mean_peak_reserved": 0,
            "max_peak_reserved": 0,
            "mean_delta_peak_allocated_vs_model_loaded": 0,
            "max_delta_peak_allocated_vs_model_loaded": 0,
            "mean_delta_peak_reserved_vs_model_loaded": 0,
            "max_delta_peak_reserved_vs_model_loaded": 0,
        }
    peaks_a = [t.peak_allocated for t in trials]
    peaks_r = [t.peak_reserved for t in trials]
    deltas_a = [t.delta_peak_allocated_vs_model_loaded for t in trials]
    deltas_r = [t.delta_peak_reserved_vs_model_loaded for t in trials]
    return {
        "num_trials": len(trials),
        "mean_peak_allocated": sum(peaks_a) / len(peaks_a),
        "max_peak_allocated": max(peaks_a),
        "mean_peak_reserved": sum(peaks_r) / len(peaks_r),
        "max_peak_reserved": max(peaks_r),
        "mean_delta_peak_allocated_vs_model_loaded": sum(deltas_a) / len(deltas_a),
        "max_delta_peak_allocated_vs_model_loaded": max(deltas_a),
        "mean_delta_peak_reserved_vs_model_loaded": sum(deltas_r) / len(deltas_r),
        "max_delta_peak_reserved_vs_model_loaded": max(deltas_r),
    }


def assert_memory_artifact_safe(obj: Any, path: str = "artifact") -> None:
    """Reject forbidden performance/serving fields in Exp 031 artifacts."""
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            assert_memory_artifact_safe(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_memory_artifact_safe(item, f"{path}[{i}]")
