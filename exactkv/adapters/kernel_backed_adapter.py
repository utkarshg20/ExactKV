"""Kernel-backed Phase H compressor adapter (delegates to Phase E kernel)."""
from __future__ import annotations

from typing import Any

import torch

from exactkv.core.compressor_interface import CompressedKV, KVCompressor
from exactkv.kernel.kv_compression_kernel import KVCompressionKernel


class KernelBackedKVCompressor(KVCompressor):
    """Wraps Phase E ``KVCompressionKernel`` for the universal interface."""

    def __init__(self, name: str, kernel_mode: str, *, supports_gpu: bool = False) -> None:
        self._name = name
        self._kernel_mode = kernel_mode
        self._supports_gpu = supports_gpu
        self._kernel = KVCompressionKernel()

    def name(self) -> str:
        return self._name

    def supports_gpu(self) -> bool:
        return self._supports_gpu

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        seed = int(kwargs.get("seed", 0))
        result = self._kernel.compress_kv(k, v, self._kernel_mode, seed=seed)
        meta = dict(result.metadata)
        meta["kernel_mode"] = self._kernel_mode
        return CompressedKV(k=result.k_compressed, v=result.v_compressed, metadata=meta)

    def decompress(self, compressed_kv: CompressedKV, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        _ = kwargs
        return compressed_kv.k, compressed_kv.v
