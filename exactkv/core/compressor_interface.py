"""Universal KV compressor interface (Phase H).

Tensor-level plug-in contract for external and simulated compressors.
Distinct from ``exactkv.compressors.base.KVCompressor`` (FullKVState lifecycle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CompressedKV:
    """Compressed KV payload returned by Phase H compressors."""

    k: torch.Tensor
    v: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float | None:
        before = self.metadata.get("memory_before")
        after = self.metadata.get("memory_after")
        if before and after and before > 0:
            return float(after) / float(before)
        return self.metadata.get("compression_ratio")


class KVCompressor(ABC):
    """Universal plug-in interface for KV compression backends."""

    @abstractmethod
    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        """Return compressed KV representation."""

    @abstractmethod
    def decompress(self, compressed_kv: CompressedKV, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed K/V tensors suitable for attention."""

    @abstractmethod
    def name(self) -> str:
        """Unique compressor identifier."""

    def supports_gpu(self) -> bool:
        """Whether backend uses CUDA/Triton."""
        return False
