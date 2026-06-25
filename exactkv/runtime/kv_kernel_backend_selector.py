"""KV compression backend selector (Phase F).

Chooses triton / cuda / torch execution path without modifying Phases A–E.
"""
from __future__ import annotations

from typing import Any, Literal

import torch

from exactkv.kernel.kv_compression_kernel import (
    KERNEL_MODES,
    CompressedKVResult,
    KVCompressionKernel,
)
from exactkv.kernel.triton_kv_compression_kernel import (
    TritonKVCompressionKernel,
    block_sparse_with_backend,
    is_triton_available,
)

BackendName = Literal["triton", "cuda", "torch"]

_ACTIVE_BACKEND: BackendName | None = None
_FORCE_TORCH = False


def select_kv_backend(preferred: str | None = None) -> BackendName:
    """Select compression backend: triton (CUDA+Triton), cuda alias, or torch."""
    global _ACTIVE_BACKEND
    if preferred is not None:
        pref = preferred.lower()
        if pref == "cuda":
            pref = "triton" if is_triton_available() else "torch"
        if pref not in ("triton", "torch"):
            msg = f"unknown backend preference: {preferred}"
            raise ValueError(msg)
        _ACTIVE_BACKEND = pref  # type: ignore[assignment]
        return _ACTIVE_BACKEND

    if _ACTIVE_BACKEND is not None:
        return _ACTIVE_BACKEND

    if _FORCE_TORCH:
        _ACTIVE_BACKEND = "torch"
    elif is_triton_available():
        _ACTIVE_BACKEND = "triton"
    else:
        _ACTIVE_BACKEND = "torch"
    return _ACTIVE_BACKEND


def set_force_torch_backend(force: bool) -> None:
    """Force Phase E torch kernels (for tests / CPU CI)."""
    global _FORCE_TORCH, _ACTIVE_BACKEND
    _FORCE_TORCH = force
    _ACTIVE_BACKEND = None


def _resolve_kernel(backend: BackendName) -> KVCompressionKernel | TritonKVCompressionKernel:
    if backend == "triton":
        return TritonKVCompressionKernel(force_torch=False)
    return KVCompressionKernel()


def compress_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mode: str,
    *,
    seed: int = 0,
    backend: str | None = None,
) -> CompressedKVResult:
    """Compress KV tensors using selected backend."""
    if mode not in KERNEL_MODES:
        msg = f"unsupported mode {mode!r}"
        raise ValueError(msg)

    selected = select_kv_backend(backend)
    if _FORCE_TORCH:
        selected = "torch"

    if mode == "block_sparse":
        from exactkv.kernel.kv_compression_kernel import (
            DEFAULT_BLOCK_SPARSE_BLOCK_SIZE,
            DEFAULT_BLOCK_SPARSE_DROP_RATE,
        )

        kernel = _resolve_kernel(selected)
        block_size = getattr(kernel, "block_size", DEFAULT_BLOCK_SPARSE_BLOCK_SIZE)
        drop_rate = getattr(kernel, "block_drop_rate", DEFAULT_BLOCK_SPARSE_DROP_RATE)
        return block_sparse_with_backend(
            k_cache,
            v_cache,
            block_size=block_size,
            drop_rate=drop_rate,
            seed=seed,
        )

    kernel = _resolve_kernel(selected)
    result = kernel.compress_kv(k_cache, v_cache, mode, seed=seed)
    result.metadata["selected_backend"] = selected
    result.metadata.setdefault("execution_backend", selected if selected == "triton" else "torch")
    return result


def backend_info() -> dict[str, Any]:
    """Return runtime backend capability snapshot."""
    return {
        "triton_available": is_triton_available(),
        "cuda_available": torch.cuda.is_available(),
        "active_backend": select_kv_backend(),
        "force_torch": _FORCE_TORCH,
        "modes": list(KERNEL_MODES),
    }
