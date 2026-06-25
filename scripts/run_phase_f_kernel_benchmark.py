#!/usr/bin/env python3
"""Phase F kernel benchmark — torch vs Triton latency and throughput."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from exactkv.kernel.kv_compression_kernel import KERNEL_MODES, PHASE_E_ID  # noqa: E402
from exactkv.kernel.triton_kv_compression_kernel import (  # noqa: E402
    PHASE_F_ID,
    TritonKVCompressionKernel,
    is_triton_available,
)
from exactkv.runtime.kv_kernel_backend_selector import backend_info, compress_kv  # noqa: E402

DEFAULT_REPORT = Path("reports/phaseF_kernel_benchmark.json")


def _sample_kv(device: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k = torch.randn(1, 8, 512, 64, generator=gen, device=device)
    v = torch.randn(1, 8, 512, 64, generator=gen, device=device)
    return k, v


def _bench_mode(
    k: torch.Tensor,
    v: torch.Tensor,
    mode: str,
    *,
    backend: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict:
    if backend == "triton" and not (is_triton_available() and k.is_cuda):
        return {"backend": backend, "mode": mode, "status": "skipped"}

    kernel = TritonKVCompressionKernel(force_torch=(backend == "torch"))
    for _ in range(warmup):
        kernel.compress_kv(k, v, mode, seed=seed)
    if k.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = None
    for _ in range(iters):
        result = kernel.compress_kv(k, v, mode, seed=seed)
    if k.is_cuda:
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    tokens = k.shape[-2]
    throughput = tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
    meta = result.metadata if result else {}
    bytes_before = meta.get("memory_before", 0)
    bytes_after = meta.get("memory_after", 0)
    bandwidth_gb_s = 0.0
    if elapsed_ms > 0 and bytes_before:
        bandwidth_gb_s = (bytes_before * 2) / (elapsed_ms / 1000.0) / 1e9

    return {
        "backend": backend,
        "mode": mode,
        "status": "ok",
        "latency_ms": round(elapsed_ms, 4),
        "compression_ratio": meta.get("compression_ratio"),
        "memory_before": bytes_before,
        "memory_after": bytes_after,
        "throughput_tokens_per_sec": round(throughput, 2),
        "memory_bandwidth_gb_per_sec": round(bandwidth_gb_s, 4),
        "execution_backend": meta.get("execution_backend", backend),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase F KV kernel benchmark")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU torch-only benchmark")
        device = "cpu"

    k, v = _sample_kv(device, args.seed)
    modes = [m for m in KERNEL_MODES if m != "noop"]
    rows: list[dict] = []

    for mode in modes:
        rows.append(
            _bench_mode(
                k, v, mode, backend="torch", warmup=args.warmup, iters=args.iters, seed=args.seed,
            ),
        )
        if device == "cuda":
            rows.append(
                _bench_mode(
                    k, v, mode, backend="triton", warmup=args.warmup, iters=args.iters, seed=args.seed,
                ),
            )

    speedups: list[dict] = []
    for mode in modes:
        torch_row = next((r for r in rows if r.get("mode") == mode and r.get("backend") == "torch" and r.get("status") == "ok"), None)
        triton_row = next((r for r in rows if r.get("mode") == mode and r.get("backend") == "triton" and r.get("status") == "ok"), None)
        if torch_row and triton_row and triton_row["latency_ms"] > 0:
            speedups.append(
                {
                    "mode": mode,
                    "torch_latency_ms": torch_row["latency_ms"],
                    "triton_latency_ms": triton_row["latency_ms"],
                    "speedup_x": round(torch_row["latency_ms"] / triton_row["latency_ms"], 2),
                },
            )

    report = {
        "phase_f_id": PHASE_F_ID,
        "phase_e_id": PHASE_E_ID,
        "backend_info": backend_info(),
        "device": device,
        "kv_shape": list(k.shape),
        "benchmarks": rows,
        "speedups": speedups,
        "note": "Measured latencies only — no hardcoded performance claims.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"phase_id={PHASE_F_ID}")
    print(f"device={device}")
    print(f"wrote={args.output}")
    for s in speedups:
        print(f"speedup {s['mode']}: {s['speedup_x']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
