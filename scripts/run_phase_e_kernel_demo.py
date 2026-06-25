#!/usr/bin/env python3
"""Phase E kernel demo — baseline vs compressed KV memory comparison."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from exactkv.kernel.kv_compression_kernel import (  # noqa: E402
    KERNEL_MODES,
    KVCompressionKernel,
    PHASE_E_ID,
)


def _sample_kv(
    *,
    device: str,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k = torch.randn(batch, heads, seq_len, head_dim, generator=gen, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, generator=gen, device=device)
    return k, v


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase E KV compression kernel demo")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"

    k, v = _sample_kv(
        device=device,
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        seed=args.seed,
    )
    kernel = KVCompressionKernel()
    baseline_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()

    print(f"phase_id={PHASE_E_ID}")
    print(f"device={device}")
    print(f"kv_shape=[{args.batch}, {args.heads}, {args.seq_len}, {args.head_dim}]")
    print(f"baseline_kv_bytes={baseline_bytes}")
    print()
    print("| mode | memory_before | memory_after | ratio | seq_len_after |")
    print("|------|--------------:|-------------:|------:|--------------:|")

    for mode in KERNEL_MODES:
        result = kernel.compress_kv(k, v, mode, seed=args.seed)
        meta = result.metadata
        if result.k_compressed.dim() >= 2:
            seq_after = result.k_compressed.shape[-2]
        else:
            seq_after = meta.get("compressed_seq_len", meta.get("original_seq_len", "packed"))
        print(
            f"| {mode} | {meta['memory_before']} | {meta['memory_after']} | "
            f"{meta['compression_ratio']:.4f} | {seq_after} |",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
