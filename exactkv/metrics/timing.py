"""Diagnostic timing helpers for Experiment 030 (V13 Phase 3).

These utilities are **not** part of the standard ExactKV report schema.
Timing fields are permitted only in approved diagnostic artifacts (Exp 030).
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def sync_device(device: str) -> None:
    """Synchronize CUDA before/after timed sections when applicable."""
    if not _is_cuda_device(device):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def timed_call(device: str, fn: Callable[[], T]) -> tuple[T, float]:
    """Run ``fn`` with CUDA sync + monotonic wall-clock timing."""
    sync_device(device)
    t0 = time.perf_counter()
    result = fn()
    sync_device(device)
    return result, time.perf_counter() - t0


def tokens_per_second(generated_tokens: int, wall_time_seconds: float) -> float:
    """Diagnostic tokens/sec for Exp 030 only."""
    if wall_time_seconds <= 0 or generated_tokens <= 0:
        return 0.0
    return generated_tokens / wall_time_seconds


def summarize_trials(
    wall_times: Sequence[float],
    token_counts: Sequence[int],
) -> dict[str, float]:
    """Mean/median wall time and tokens/sec across repeated trials."""
    if not wall_times:
        return {
            "mean_wall_time_seconds": 0.0,
            "median_wall_time_seconds": 0.0,
            "stdev_wall_time_seconds": 0.0,
            "mean_tokens_per_second": 0.0,
            "median_tokens_per_second": 0.0,
        }
    tps = [
        tokens_per_second(tok, wt)
        for tok, wt in zip(token_counts, wall_times, strict=True)
    ]
    stdev = statistics.pstdev(wall_times) if len(wall_times) > 1 else 0.0
    return {
        "mean_wall_time_seconds": statistics.mean(wall_times),
        "median_wall_time_seconds": statistics.median(wall_times),
        "stdev_wall_time_seconds": stdev,
        "mean_tokens_per_second": statistics.mean(tps),
        "median_tokens_per_second": statistics.median(tps),
    }


def estimate_sequential_verifier_forwards(traces: Sequence[Any]) -> int:
    """Estimate full-model forward passes during sequential verification."""
    total = 0
    for trace in traces:
        acc = trace.acceptance
        n = len(trace.draft_tokens)
        if acc.all_matched:
            total += max(0, n - 1)
        else:
            total += acc.num_accepted
    return total


def estimate_span_verifier_forwards(traces: Sequence[Any]) -> int:
    """Estimate full-model forward passes during span verification."""
    return sum(1 for trace in traces if len(trace.draft_tokens) >= 2)


def collect_timing_environment(device: str) -> dict[str, Any]:
    """Record hardware/software metadata for diagnostic timing manifests."""
    import socket

    import torch
    import transformers

    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta["cuda_version"] = torch.version.cuda
        idx = 0 if device == "cuda" else int(device.split(":")[-1])
        meta["gpu_device_name"] = torch.cuda.get_device_name(idx)
    return meta
