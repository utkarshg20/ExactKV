"""TurboQuant Phase H adapter (mock when upstream package unavailable)."""
from __future__ import annotations

from typing import Any

import torch

from exactkv.adapters.kernel_backed_adapter import KernelBackedKVCompressor
from exactkv.core.compressor_interface import CompressedKV, KVCompressor


class TurboQuantKVCompressor(KernelBackedKVCompressor):
    """Conforms to ``KVCompressor``; int8 kernel proxy for TurboQuant slot."""

    def __init__(self) -> None:
        super().__init__("turboquant", "int8", supports_gpu=False)

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        out = super().compress(k, v, **kwargs)
        out.metadata["adapter"] = "turboquant"
        out.metadata["backend"] = "mock_int8_kernel_fallback"
        try:
            from exactkv.compressors import turboquant_adapter  # noqa: F401, PLC0415

            out.metadata["upstream_module"] = turboquant_adapter.__name__
        except ImportError:
            out.metadata["upstream_module"] = None
        return out

    @staticmethod
    def try_real_backend(runtime: Any = None) -> KVCompressor | None:
        """Return real TurboQuant adapter if importable; else None."""
        _ = runtime
        try:
            from exactkv.compressors.turboquant_adapter import TurboQuantPythonAdapter  # noqa: PLC0415

            return _RuntimeBackedWrapper(TurboQuantPythonAdapter, "turboquant")
        except ImportError:
            return None


class _RuntimeBackedWrapper(KVCompressor):
    """Placeholder for future runtime-backed TurboQuant tensor bridge."""

    def __init__(self, cls: type, name: str) -> None:
        self._cls = cls
        self._name = name

    def name(self) -> str:
        return self._name

    def supports_gpu(self) -> bool:
        return False

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        _ = kwargs
        return CompressedKV(k=k, v=v, metadata={"backend": "runtime_adapter_unwired"})

    def decompress(self, compressed_kv: CompressedKV, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        _ = kwargs
        return compressed_kv.k, compressed_kv.v
