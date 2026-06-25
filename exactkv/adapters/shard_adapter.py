"""Shard Phase H adapter (probe simulation via block_sparse kernel)."""
from __future__ import annotations

from typing import Any

from exactkv.adapters.kernel_backed_adapter import KernelBackedKVCompressor
from exactkv.core.compressor_interface import CompressedKV


class ShardKVCompressor(KernelBackedKVCompressor):
    """Conforms to ``KVCompressor``; block-sparse compaction proxy for shard probe."""

    def __init__(self) -> None:
        super().__init__("shard", "block_sparse", supports_gpu=False)

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        out = super().compress(k, v, **kwargs)
        out.metadata["adapter"] = "shard"
        out.metadata["backend"] = "block_sparse_probe_proxy"
        out.metadata["probe_only"] = True
        return out
