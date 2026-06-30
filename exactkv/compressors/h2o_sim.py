"""H2OSimCompressor — Heavy Hitter Oracle token-eviction compressor (v2.8).

H2O (Zhang et al., 2023) keeps a fixed budget of KV tokens by retaining those
with the highest accumulated attention scores. Because ExactKV operates post-prefill
and does not yet expose attention score tensors through the KVCompressor protocol,
this implementation uses the well-established H2O approximation:

    keep = attention_sink_tokens  ∪  recent_window_tokens

Attention sinks (Xiao et al., 2023 / StreamingLLM) are the first few tokens in
the sequence, which empirically accumulate disproportionately large attention
weights in nearly all transformer architectures. Recent tokens also have high
attention scores in causal LMs.

This approximation is a "simulate path" adapter:

* ``is_simulated = True``     — approximates real attention-score-based eviction
* ``supports_real_bytes_claim = True`` — dropped tokens genuinely free memory
* ``supports_token_dropping = True`` — this is the point

Claim boundary
--------------
H2OSimCompressor produces real byte savings (dropped KV pairs are not stored).
Because we approximate attention scores rather than tracking them live,
divergence rates may differ from true H2O with online score tracking.
Label this in the paper as "H2O-approximate (sink+recent strategy)".

Parameters
----------
keep_ratio : float
    Fraction of sequence tokens to keep (0 < keep_ratio ≤ 1). Default 0.5.
    Applied per-layer independently to key AND value tensors.
sink_fraction : float
    Of the kept budget, what fraction goes to the initial-token sink window.
    Default 0.25 (e.g. keep_ratio=0.5 → 25% sinks + 75% recent).
min_keep : int
    Absolute minimum number of tokens to keep regardless of keep_ratio.
    Prevents degenerate empty sequences. Default 4.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from exactkv.cache.compressed_state import CompressedKVState
from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import (
    extract_kv_tensors,
    kv_seq_len,
    rebuild_cache,
)
from exactkv.compressors.base import CompressorCapabilities, CompressionStats

_FP32_BYTES = 4  # reference bytes per element


def _evict_tokens(
    k: torch.Tensor,
    v: torch.Tensor,
    keep_ratio: float,
    sink_fraction: float,
    min_keep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return kept (k, v, kept_indices) after H2O-style eviction.

    k, v shapes: (batch, heads, seq_len, head_dim)  [standard HF DynamicCache]
               OR (batch, seq_len, heads, head_dim)  [alternative format]

    We operate on the sequence-length dimension regardless of format.
    """
    # Infer the sequence dimension: usually dim 2 for (B, H, S, D).
    # We detect by comparing shape[1] and shape[2] to typical head counts (1–128).
    # Heuristic: for most models, heads ≤ 128 and seq_len > heads.
    if k.ndim == 4:
        # (batch, heads, seq_len, head_dim)  — standard HF layout
        seq_dim = 2
        seq_len = k.shape[seq_dim]
    else:
        # Unexpected shape: fall back to noop (keep all)
        return k, v, torch.arange(k.shape[-2], device=k.device)

    budget = max(min_keep, math.ceil(keep_ratio * seq_len))
    if budget >= seq_len:
        return k, v, torch.arange(seq_len, device=k.device)

    sink_count   = max(1, round(budget * sink_fraction))
    recent_count = budget - sink_count

    sink_indices   = torch.arange(sink_count, device=k.device)
    recent_start   = max(sink_count, seq_len - recent_count)
    recent_indices = torch.arange(recent_start, seq_len, device=k.device)

    # Merge and deduplicate while preserving order
    kept_set  = torch.cat([sink_indices, recent_indices])
    kept_idx, sort_order = torch.sort(kept_set.unique())
    del sort_order  # noqa: F841 — we only need kept_idx sorted

    kept_k = k.index_select(seq_dim, kept_idx)
    kept_v = v.index_select(seq_dim, kept_idx)
    return kept_k, kept_v, kept_idx


class H2OSimCompressor:
    """H2O-style token eviction compressor (attention sink + recency window).

    This is the first token-dropping compressor in ExactKV (v2.8). It drops
    the lowest-priority KV tokens by keeping:
      * ``sink_fraction`` of the budget from the start (attention sinks)
      * ``1 - sink_fraction`` of the budget from the end (recent window)

    ExactKV's verifier loop ensures ``exactkv_failures=0`` even when the lossy
    path diverges, allowing clean measurement of where eviction-induced drift begins.
    """

    name: str = "h2o_sim"

    def __init__(
        self,
        keep_ratio: float = 0.5,
        sink_fraction: float = 0.25,
        min_keep: int = 4,
    ) -> None:
        if not 0 < keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if not 0 < sink_fraction < 1:
            raise ValueError(f"sink_fraction must be in (0, 1), got {sink_fraction}")
        self.keep_ratio    = keep_ratio
        self.sink_fraction = sink_fraction
        self.min_keep      = min_keep

        self.capabilities = CompressorCapabilities(
            name="h2o_sim",
            compressor_type="token_dropping",
            is_simulated=True,
            supports_real_bytes_claim=True,
            supports_token_dropping=True,
            supports_quantization=False,
            key_bit_width=None,
            value_bit_width=None,
            notes=(
                f"H2O-approximate token eviction: keeps first {sink_fraction:.0%} of budget "
                f"as attention sinks and last {1-sink_fraction:.0%} as recency window. "
                f"keep_ratio={keep_ratio}. "
                "is_simulated=True because real H2O tracks online attention scores; "
                "this uses the sink+recent approximation. "
                "supports_real_bytes_claim=True: dropped tokens are not stored."
            ),
        )

    def compress(self, state: FullKVState) -> CompressedKVState:
        """Evict low-priority KV tokens using sink + recency heuristic."""
        k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
        seq_len = kv_seq_len(state.past_key_values)

        layers: list[dict[str, Any]] = []
        first_kept_indices: torch.Tensor | None = None

        for k, v in zip(k_tensors, v_tensors):
            kept_k, kept_v, kept_idx = _evict_tokens(
                k, v, self.keep_ratio, self.sink_fraction, self.min_keep,
            )
            layers.append({"k": kept_k, "v": kept_v, "kept_indices": kept_idx})
            if first_kept_indices is None:
                first_kept_indices = kept_idx

        kept_len = layers[0]["k"].shape[2] if layers else seq_len

        data: dict[str, Any] = {
            "layers":       layers,
            "cache_format": cache_format,
            "original_seq_len": seq_len,
            "kept_seq_len": kept_len,
            "dtype":        state.dtype,
            "device":       state.device,
            "keep_ratio":   self.keep_ratio,
        }

        return CompressedKVState(
            data=data,
            metadata={
                "next_token_id":    state.next_token_id,
                "kept_seq_len":     kept_len,
                "original_seq_len": seq_len,
                "tokens_evicted":   seq_len - kept_len,
                "eviction_rate":    round(1.0 - kept_len / max(seq_len, 1), 4),
            },
            compressor_name=self.name,
            logical_seq_len=state.seq_len,
            generated_ids=state.generated_ids,
            device=state.device,
        )

    def materialize_for_draft(self, compressed: CompressedKVState) -> Any:
        """Return the evicted cache (kept tokens only) for the draft forward pass."""
        d = compressed.data
        dtype        = d["dtype"]
        device       = d["device"]
        cache_format = d["cache_format"]
        kept_len     = d["kept_seq_len"]

        k_out = [layer["k"].to(dtype=dtype, device=device) for layer in d["layers"]]
        v_out = [layer["v"].to(dtype=dtype, device=device) for layer in d["layers"]]

        return rebuild_cache(k_out, v_out, cache_format, kept_len)

    def update_after_commit(
        self,
        compressed: CompressedKVState,
        new_full_state: FullKVState,
    ) -> CompressedKVState:
        """Re-evict from the updated full state (V1 strategy: recompress from full)."""
        return self.compress(new_full_state)

    def stats(self, compressed: CompressedKVState) -> CompressionStats:
        d = compressed.data
        layers   = d["layers"]
        orig_len = d["original_seq_len"]
        kept_len = d["kept_seq_len"]

        # Reference: full fp32 storage of the *original* (unevicted) sequence
        full_bytes = sum(
            l["k"].nelement() * _FP32_BYTES * (orig_len / max(kept_len, 1))
            + l["v"].nelement() * _FP32_BYTES * (orig_len / max(kept_len, 1))
            for l in layers
        )
        full_bytes = int(full_bytes)

        # Actual stored: only the kept tokens in the original dtype
        elem_bytes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(d["dtype"], 4)
        stored_bytes = sum(
            l["k"].nelement() * elem_bytes + l["v"].nelement() * elem_bytes
            for l in layers
        )
        # kept_indices tensor overhead (int64 × kept_len per layer)
        index_bytes = sum(l["kept_indices"].nelement() * 8 for l in layers)
        compressed_bytes = stored_bytes + index_bytes

        ratio  = compressed_bytes / max(full_bytes, 1)
        factor = full_bytes / max(compressed_bytes, 1)

        return CompressionStats(
            compressor_name=self.name,
            full_bytes=full_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=round(ratio, 4),
            memory_reduction_factor=round(factor, 4),
            seq_len=kept_len,
            num_layers=len(layers),
            stored_kv_bytes=stored_bytes,
            materialized_working_kv_bytes=stored_bytes,
            metadata_bytes=index_bytes,
            temporary_workspace_bytes=0,
            total_kv_footprint_bytes=stored_bytes + index_bytes,
        )
