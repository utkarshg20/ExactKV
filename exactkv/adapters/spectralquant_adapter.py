"""SpectralQuant Phase H adapter (mock fallback via int4 kernel path)."""
from __future__ import annotations

from typing import Any

import torch

from exactkv.adapters.kernel_backed_adapter import KernelBackedKVCompressor
from exactkv.core.compressor_interface import CompressedKV


class SpectralQuantKVCompressor(KernelBackedKVCompressor):
    """Conforms to ``KVCompressor``; uses int4 kernel simulation when real SQ unavailable."""

    def __init__(self) -> None:
        super().__init__("spectralquant", "int4", supports_gpu=False)

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        out = super().compress(k, v, **kwargs)
        out.metadata["adapter"] = "spectralquant"
        out.metadata["backend"] = "mock_int4_kernel_fallback"
        return out
