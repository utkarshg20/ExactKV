"""Isolated GPU memory pilot helpers for Experiment 018.

These helpers record PyTorch CUDA allocation observations at defined lifecycle
points.  They are **not** part of the standard ExactKV report schema and do
**not** populate ``active_gpu_kv_bytes``.

This is a methodology pilot only — not a performance benchmark.
"""
from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Any

# Pilot-only field names (never use ``active_gpu_kv_bytes``).
PILOT_GPU_FIELD_NAMES = frozenset({
    "gpu_baseline_model_loaded_bytes",
    "gpu_allocated_after_prefill_bytes",
    "gpu_peak_allocated_during_run_bytes",
    "gpu_allocated_after_run_bytes",
    "gpu_allocated_after_cleanup_bytes",
})

STANDARD_REPORT_FIELD_NAMES = frozenset({
    "prompt_id",
    "compressor_name",
    "exactkv_failure",
    "memory",
    "exactkv",
})

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})


@dataclass(frozen=True)
class GpuMemorySnapshot:
    """Device-level CUDA allocation observation at one lifecycle point."""

    gpu_baseline_model_loaded_bytes: int
    gpu_allocated_after_prefill_bytes: int
    gpu_peak_allocated_during_run_bytes: int
    gpu_allocated_after_run_bytes: int
    gpu_allocated_after_cleanup_bytes: int | None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


def cuda_available() -> bool:
    """Return True when PyTorch CUDA is available."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def collect_runpod_meta() -> dict[str, Any]:
    """Collect GPU/torch metadata for pilot manifests."""
    meta: dict[str, Any] = {"cuda_available": cuda_available()}
    if not meta["cuda_available"]:
        return meta
    try:
        import socket
        import torch

        meta["hostname"] = socket.gethostname()
        meta["gpu_device_name"] = torch.cuda.get_device_name(0)
        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return meta


def _sync_and_reset_peak() -> None:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _allocated_bytes() -> int:
    import torch

    torch.cuda.synchronize()
    return int(torch.cuda.memory_allocated())


def _peak_allocated_bytes() -> int:
    import torch

    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def measure_exactkv_cell_gpu_memory(
    runtime: Any,
    prompt: str,
    compressor: Any,
    *,
    draft_len: int = 4,
    max_new_tokens: int = 16,
) -> tuple[Any, GpuMemorySnapshot, bool]:
    """Run one ExactKV cell and record CUDA allocation observations.

    Returns ``(ExactKVResult, snapshot, exactkv_token_match)``.

    Requires CUDA.  Does not modify generation or verification logic.
    """
    if not cuda_available():
        raise RuntimeError("CUDA required for GPU memory pilot measurements")

    from exactkv.metrics.exactness import token_exact_match
    from exactkv.runtime.exactkv_generator import ExactKVGenerator
    from exactkv.runtime.generation import generate_full_greedy
    from exactkv.runtime.prefill import prefill_to_full_state

    _sync_and_reset_peak()
    baseline = _allocated_bytes()

    full_state = prefill_to_full_state(runtime, prompt)
    _sync_and_reset_peak()
    after_prefill = _allocated_bytes()

    gen = ExactKVGenerator(runtime, compressor, draft_len=draft_len)
    result = gen.generate(prompt, max_new_tokens)

    peak = _peak_allocated_bytes()
    after_run = _allocated_bytes()

    full_out = generate_full_greedy(runtime, prompt, max_new_tokens)
    exact = token_exact_match(full_out.generated_ids, result.output_ids)

    # Best-effort cleanup of cell-local references.
    after_cleanup: int | None = None
    try:
        del full_state, gen, full_out
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_cleanup = _allocated_bytes()
    except Exception:
        after_cleanup = None

    snapshot = GpuMemorySnapshot(
        gpu_baseline_model_loaded_bytes=baseline,
        gpu_allocated_after_prefill_bytes=after_prefill,
        gpu_peak_allocated_during_run_bytes=peak,
        gpu_allocated_after_run_bytes=after_run,
        gpu_allocated_after_cleanup_bytes=after_cleanup,
    )
    return result, snapshot, exact


def assert_pilot_artifact_safe(obj: Any, path: str = "pilot") -> None:
    """Reject forbidden fields and ``active_gpu_kv_bytes`` anywhere in artifact."""
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            assert_pilot_artifact_safe(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_pilot_artifact_safe(item, f"{path}[{i}]")


def pilot_fields_not_in_standard_schema() -> bool:
    """Return True — pilot GPU fields are intentionally outside standard reports."""
    return True
