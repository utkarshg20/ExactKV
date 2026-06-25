"""KVQuant Phase H adapter (mock fallback via int4 kernel path)."""
from __future__ import annotations

from typing import Any

from exactkv.adapters.kernel_backed_adapter import KernelBackedKVCompressor
from exactkv.core.compressor_interface import CompressedKV


class KVQuantKVCompressor(KernelBackedKVCompressor):
    """Conforms to ``KVCompressor``; int4 kernel proxy when KVQuant env unavailable."""

    def __init__(self) -> None:
        super().__init__("kvquant", "int4", supports_gpu=False)

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        out = super().compress(k, v, **kwargs)
        out.metadata["adapter"] = "kvquant"
        out.metadata["backend"] = "mock_int4_kernel_fallback"
        return out
