"""Shard real adapter (probe-first) for Phase H+."""
from __future__ import annotations

from typing import Any

import torch

from exactkv.adapters.kernel_backed_adapter import KernelBackedKVCompressor
from exactkv.core.compressor_interface import CompressedKV, KVCompressor


def _estimate_shard_probe_scores(k: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    """Heuristic probe scores from tensor statistics (no new metrics — proxies only)."""
    kv = torch.cat([k.reshape(-1), v.reshape(-1)])
    norm = float(kv.norm().item())
    var = float(kv.var().item()) if kv.numel() > 1 else 0.0
    stability = max(0.0, min(1.0, 1.0 / (1.0 + var * 0.01)))
    divergence_risk = max(0.0, min(1.0, norm * 1e-4))
    acceptance_proxy = max(0.0, min(1.0, 1.0 - divergence_risk * 0.5))
    return {
        "stability_score_estimate": round(stability, 4),
        "divergence_risk_estimate": round(divergence_risk, 4),
        "acceptance_proxy_score": round(acceptance_proxy, 4),
    }


class ShardRealKVCompressor(KVCompressor):
    """Probe-first Shard adapter with block-sparse heuristic backend."""

    def __init__(self) -> None:
        self._backend = KernelBackedKVCompressor("shard_real", "block_sparse")

    def name(self) -> str:
        return "shard_real"

    def supports_gpu(self) -> bool:
        return torch.cuda.is_available()

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        out = self._backend.compress(k, v, **kwargs)
        scores = _estimate_shard_probe_scores(k, v)
        out.metadata.update(
            {
                "adapter": "shard_real",
                "backend": "block_sparse_probe_heuristic",
                "probe_only": True,
                **scores,
            },
        )
        return out

    def decompress(self, compressed_kv: CompressedKV, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self._backend.decompress(compressed_kv, **kwargs)
