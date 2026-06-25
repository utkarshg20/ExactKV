"""Bridge Phase D probe compression to Phase E kernel execution.

Preserves Phase D API shape; Phases A–D modules remain unchanged.
"""
from __future__ import annotations

from typing import Any

import torch

from exactkv.kernel.kv_compression_kernel import (
    KERNEL_MODES,
    CompressedKVResult,
    KVCompressionKernel,
    compress_from_phase_d_output,
)

# Phase D probe mode → Phase E kernel mode
PHASE_D_TO_KERNEL_MODE: dict[str, str] = {
    "noop": "noop",
    "int8_sim": "int8",
    "int4_sim": "int4",
    "kv_dropout_sim": "block_sparse",
}

# Kernel mode aliases for direct calls
KERNEL_MODE_ALIASES: dict[str, str] = {
    **PHASE_D_TO_KERNEL_MODE,
    **{m: m for m in KERNEL_MODES},
}


def resolve_kernel_mode(mode: str) -> str:
    """Map Phase D or kernel mode string to canonical kernel mode."""
    if mode in KERNEL_MODE_ALIASES:
        return KERNEL_MODE_ALIASES[mode]
    msg = f"unknown compression mode: {mode}"
    raise ValueError(msg)


def compress_kv_via_kernel(
    k_tensors: list[torch.Tensor],
    v_tensors: list[torch.Tensor],
    mode: str,
    *,
    seed: int = 0,
    kernel: KVCompressionKernel | None = None,
    **kwargs: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Drop-in replacement for ``simulate_compression_on_kv`` using Phase E kernel.

    Returns dequantized K/V lists suitable for ``rebuild_cache`` forward paths.
    """
    _ = kwargs  # dropout_rate etc. ignored; block_sparse uses kernel seed
    k_kernel = kernel or KVCompressionKernel()
    kernel_mode = resolve_kernel_mode(mode)
    k_out: list[torch.Tensor] = []
    v_out: list[torch.Tensor] = []
    for layer_idx, (k, v) in enumerate(zip(k_tensors, v_tensors)):
        result = k_kernel.compress_kv(
            k,
            v,
            kernel_mode,
            seed=seed + layer_idx,
        )
        k_out.append(result.k_dequant)
        v_out.append(result.v_dequant)
    return k_out, v_out


def compress_kv_via_kernel_with_metadata(
    k_tensors: list[torch.Tensor],
    v_tensors: list[torch.Tensor],
    mode: str,
    *,
    seed: int = 0,
    kernel: KVCompressionKernel | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[CompressedKVResult]]:
    """Like ``compress_kv_via_kernel`` but also return per-layer ``CompressedKVResult``."""
    k_kernel = kernel or KVCompressionKernel()
    kernel_mode = resolve_kernel_mode(mode)
    k_out: list[torch.Tensor] = []
    v_out: list[torch.Tensor] = []
    results: list[CompressedKVResult] = []
    for layer_idx, (k, v) in enumerate(zip(k_tensors, v_tensors)):
        result = k_kernel.compress_kv(k, v, kernel_mode, seed=seed + layer_idx)
        k_out.append(result.k_dequant)
        v_out.append(result.v_dequant)
        results.append(result)
    return k_out, v_out, results


def use_kernel_for_phase_d(enabled: bool = True) -> None:
    """Opt-in monkey-patch hook: route Phase D simulation through kernel.

    Call ``use_kernel_for_phase_d(False)`` to restore original simulation.
    Phases A–D source files are not modified; this patches at runtime only.
    """
    import exactkv.runtime.kv_probe_layer as probe_mod

    if not enabled:
        if hasattr(probe_mod, "_ORIGINAL_SIMULATE_COMPRESSION"):
            probe_mod.simulate_compression_on_kv = probe_mod._ORIGINAL_SIMULATE_COMPRESSION
        return

    if not hasattr(probe_mod, "_ORIGINAL_SIMULATE_COMPRESSION"):
        probe_mod._ORIGINAL_SIMULATE_COMPRESSION = probe_mod.simulate_compression_on_kv
    probe_mod.simulate_compression_on_kv = compress_kv_via_kernel


__all__ = [
    "PHASE_D_TO_KERNEL_MODE",
    "compress_from_phase_d_output",
    "compress_kv_via_kernel",
    "compress_kv_via_kernel_with_metadata",
    "resolve_kernel_mode",
    "use_kernel_for_phase_d",
]
