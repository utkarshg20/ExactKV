"""Restricted KIVI offline adapter (isolated PYTHONPATH / .venv-kivi only).

NOT registered in the default compressor registry.  Upstream ``models.utils_quant``
from jy-yuan/KIVI is imported lazily when ``KIVIOfflineAdapter`` is constructed.

Scope: offline simulate quant/dequant on post-RoPE HF KV tensors only.  No
LlamaForCausalLM_KIVI, no CUDA/Triton pack kernels, no flash-attn, no kivi_gemv.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn.functional as F

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import extract_kv_tensors, kv_total_bytes, rebuild_cache
from exactkv.compressors.backend_adapter import BackendAdapter
from exactkv.compressors.base import CompressorCapabilities
from exactkv.runtime.model_runtime import ModelRuntime


def _ensure_kivi_on_path() -> None:
    """Prepend KIVI repo root from ``EXACTKV_KIVI_ROOT`` without global PYTHONPATH."""
    root = os.environ.get("EXACTKV_KIVI_ROOT", "").strip()
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _import_kivi_utils() -> tuple[Any, Any, Any]:
    """Lazy import of upstream KIVI quant helpers (jy-yuan/KIVI models/utils_quant.py)."""
    _ensure_kivi_on_path()
    try:
        from models.utils_quant import (  # noqa: PLC0415
            dequantize_by_channel_and_unpack_cache,
            process_input,
            quantize_by_channel_and_pack_cache,
        )
    except ImportError as exc:
        raise ImportError(
            "KIVI models.utils_quant is not importable. Clone https://github.com/jy-yuan/KIVI "
            "and set EXACTKV_KIVI_ROOT to the repo root (e.g. EXACTKV_KIVI_ROOT=/tmp/kivi_research). "
            "Full KIVI pip install is not required for the offline simulate path."
        ) from exc
    return (
        quantize_by_channel_and_pack_cache,
        dequantize_by_channel_and_unpack_cache,
        process_input,
    )


def _discover_backend_version() -> str:
    try:
        import models.utils_quant as uq  # noqa: PLC0415

        root = os.path.dirname(os.path.dirname(os.path.abspath(uq.__file__)))
        git_head = os.path.join(root, ".git", "HEAD")
        if os.path.isfile(git_head):
            with open(git_head, encoding="utf-8") as fh:
                ref = fh.read().strip()
            if ref.startswith("ref: "):
                ref_path = os.path.join(root, ".git", ref[5:])
                if os.path.isfile(ref_path):
                    with open(ref_path, encoding="utf-8") as fh:
                        return fh.read().strip()[:12]
            return ref[:12]
        return f"dev:{os.path.basename(root)}"
    except Exception:
        return "unknown"


def _infer_head_dim(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Cannot infer head_dim: model has no config")
    hidden = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if hidden is None or num_heads is None:
        raise ValueError(
            f"Cannot infer head_dim from config: hidden_size={hidden}, "
            f"num_attention_heads={num_heads}"
        )
    if num_heads <= 0 or hidden % num_heads != 0:
        raise ValueError(
            f"Invalid attention head configuration: hidden_size={hidden}, "
            f"num_attention_heads={num_heads}"
        )
    return hidden // num_heads


def _kivi_simulate_quantize_per_token(
    data: torch.Tensor,
    group_size: int,
    num_bits: int,
    *,
    process_input_fn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU-safe port of KIVI ``quantize_and_pack(..., simulate=True)`` per-token path.

    Upstream ``quantize_and_pack`` hardcodes ``device='cuda'`` for the simulate
    branch; this wrapper preserves the same math on ``data.device``.
    """
    flat = data.reshape(-1, data.shape[-1])
    input_groups, mn, mx = process_input_fn(flat, group_size)
    input_groups = input_groups.transpose(0, 1)
    mn_t = mn.t()
    mx_t = mx.t()
    mn_exp = mn_t.unsqueeze(-1)
    mx_exp = mx_t.unsqueeze(-1)
    n_groups = input_groups.shape[0]
    bits_row = torch.full(
        (n_groups,),
        num_bits,
        dtype=torch.int32,
        device=data.device,
    )
    b_levels = (2 ** bits_row - 1).view(n_groups, 1, 1).to(dtype=input_groups.dtype)
    mn_adj = mn_exp - 1e-6
    mx_adj = mx_exp + 1e-6
    scale = b_levels / (mx_adj - mn_adj)
    output = (input_groups - mn_adj) * scale
    output = F.relu(output)
    output = torch.min(output, b_levels).round_()
    return output, scale.squeeze(-1), mn_t


def _kivi_simulate_dequantize_per_token(
    data: torch.Tensor,
    group_size: int,
    shape: torch.Size,
    scale: torch.Tensor,
    mn: torch.Tensor,
) -> torch.Tensor:
    """CPU-safe port of KIVI ``dequantize_and_unpack(..., simulate=True)``."""
    scale_exp = scale.unsqueeze(-1)
    mn_exp = mn.unsqueeze(-1)
    restored = data / scale_exp + mn_exp
    return restored.reshape(shape)


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.element_size() * obj.nelement())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    return 0


def _layer_payload_bytes(layer: dict) -> int:
    total = 0
    for key in (
        "k_quant",
        "k_scale",
        "k_mn",
        "v_quant",
        "v_scale",
        "v_mn",
        "k_residual",
        "v_residual",
    ):
        if key in layer:
            total += _tensor_bytes(layer[key])
    return total


def _split_residual(
    tensor: torch.Tensor,
    residual_length: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Split (B, H, S, D) into quantizable prefix and fp16 residual suffix."""
    if residual_length <= 0:
        return tensor, None
    seq_len = int(tensor.shape[2])
    if seq_len <= residual_length:
        return None, tensor
    split = seq_len - residual_length
    return tensor[:, :, :split, :], tensor[:, :, split:, :]


def _merge_prefix_residual(
    prefix: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    if prefix is None:
        if residual is None:
            raise ValueError("Both prefix and residual are None")
        return residual
    if residual is None:
        return prefix
    return torch.cat([prefix, residual], dim=2)


def _compress_layer_k(
    k_tensor: torch.Tensor,
    *,
    group_size: int,
    k_bits: int,
    quantize_k_cache: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    shape = k_tensor.shape
    k_q, k_scale, k_mn = quantize_k_cache(
        k_tensor,
        group_size,
        k_bits,
        simulate=True,
    )
    return k_q, k_scale, k_mn, shape


def _decompress_layer_k(
    k_q: torch.Tensor,
    k_scale: torch.Tensor,
    k_mn: torch.Tensor,
    shape: torch.Size,
    *,
    group_size: int,
    k_bits: int,
    dequantize_k_cache: Any,
) -> torch.Tensor:
    return dequantize_k_cache(
        k_q,
        group_size,
        shape,
        k_bits,
        k_scale,
        k_mn,
        simulate=True,
    )


def _compress_layer_v(
    v_tensor: torch.Tensor,
    *,
    group_size: int,
    v_bits: int,
    process_input_fn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    shape = v_tensor.shape
    if shape[0] != 1:
        raise ValueError(f"V tensor batch size must be 1, got {shape[0]}")
    # Per-token: quantize head_dim groups for each (head, seq) position.
    v_work = v_tensor.squeeze(0)
    v_flat = v_work.reshape(-1, v_work.shape[-1])
    v_q, v_scale, v_mn = _kivi_simulate_quantize_per_token(
        v_flat,
        group_size,
        v_bits,
        process_input_fn=process_input_fn,
    )
    return v_q, v_scale, v_mn, shape


def _decompress_layer_v(
    v_q: torch.Tensor,
    v_scale: torch.Tensor,
    v_mn: torch.Tensor,
    shape: torch.Size,
    *,
    group_size: int,
) -> torch.Tensor:
    if shape[0] != 1:
        raise ValueError(f"V tensor batch size must be 1, got {shape[0]}")
    heads, seq_len, head_dim = shape[1], shape[2], shape[3]
    flat_shape = torch.Size((heads * seq_len, head_dim))
    v_flat = _kivi_simulate_dequantize_per_token(
        v_q,
        group_size,
        flat_shape,
        v_scale,
        v_mn,
    )
    v_work = v_flat.reshape(heads, seq_len, head_dim)
    return v_work.unsqueeze(0)


class KIVIOfflineAdapter(BackendAdapter):
    """Offline KIVI adapter using upstream ``models.utils_quant`` simulate paths.

    Compresses cloned post-RoPE HF KV tensors, stores quant codes + scales,
    and materialises dequantised torch tensors for the draft path only.
    Verification always uses the authoritative full-precision ``FullKVState``.
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        *,
        head_dim: int | None = None,
        k_bits: int = 2,
        v_bits: int = 2,
        group_size: int = 32,
        residual_length: int = 0,
    ) -> None:
        (
            self._quantize_k_cache,
            self._dequantize_k_cache,
            self._process_input,
        ) = _import_kivi_utils()

        self._runtime = runtime
        self._head_dim = head_dim if head_dim is not None else _infer_head_dim(runtime.model)
        self._k_bits = k_bits
        self._v_bits = v_bits
        self._group_size = group_size
        self._residual_length = max(int(residual_length), 0)

        k_label = f"kivi_k{k_bits}_offline"
        v_label = f"kivi_v{v_bits}_offline"
        asymmetric = k_bits != v_bits
        if self._residual_length > 0:
            self.name = f"kivi_offline_k{k_bits}_v{v_bits}_r{self._residual_length}"
        else:
            self.name = f"kivi_offline_k{k_bits}_v{v_bits}"

        self.capabilities = CompressorCapabilities(
            name=self.name,
            compressor_type="quantization",
            is_simulated=False,
            supports_real_bytes_claim=False,
            supports_token_dropping=False,
            supports_quantization=True,
            key_bit_width=k_bits,
            value_bit_width=v_bits,
            key_bit_width_label=k_label,
            value_bit_width_label=v_label,
            asymmetric=asymmetric,
            backend_name="kivi",
            backend_version=_discover_backend_version(),
            adapter_name="KIVIOfflineAdapter",
            adapter_version="0.1.0",
            notes=(
                "Restricted V9 Phase D2 offline KIVI adapter (PYTHONPATH to jy-yuan/KIVI only). "
                "Uses upstream models.utils_quant simulate quant/dequant on post-RoPE HF cache "
                "tensors. V per-token path uses CPU-safe port of KIVI quantize_and_pack simulate "
                "math (upstream simulate branch hardcodes CUDA). "
                "Not KIVI production CUDA/Triton path. Not packed-bit storage. "
                "Not LlamaForCausalLM_KIVI or MistralForCausalLM_KIVI. "
                + (
                    f"Residual fp16 window: last {self._residual_length} tokens "
                    f"kept in fp16 (KIVI streaming policy). "
                    if self._residual_length > 0
                    else "Residual fp16 window not enabled (uniform quant over full seq). "
                )
                + "Not in the default compressor registry. "
                "supports_real_bytes_claim=False: stores unpacked quant codes and scale tensors. "
                "Byte accounting counts actual stored torch payloads honestly. "
                "Draft may diverge from full KV; verification uses authoritative full state only. "
                "Not KVQuant. No throughput, latency, or production-readiness claims."
            ),
        )

    def _backend_compress(
        self,
        k_tensors: list[torch.Tensor],
        v_tensors: list[torch.Tensor],
        cache_format: str,
    ) -> dict:
        if len(k_tensors) != len(v_tensors):
            raise ValueError(
                f"K/V layer count mismatch: {len(k_tensors)} vs {len(v_tensors)}"
            )
        if not k_tensors:
            raise ValueError("Empty KV tensor list")

        layers: list[dict] = []
        stored_bytes = 0

        for layer_idx, (k, v) in enumerate(zip(k_tensors, v_tensors)):
            if k.shape != v.shape:
                raise ValueError(
                    f"Layer {layer_idx}: K shape {k.shape} != V shape {v.shape}"
                )
            if k.ndim != 4:
                raise ValueError(
                    f"Layer {layer_idx}: expected 4D KV tensor (B, H, S, D), got {k.shape}"
                )
            batch, heads, seq_len, head_dim = k.shape
            if batch != 1:
                raise ValueError(
                    f"Layer {layer_idx}: batch size must be 1, got {batch}"
                )
            if head_dim != self._head_dim:
                raise ValueError(
                    f"Layer {layer_idx}: head_dim={head_dim} != expected {self._head_dim}"
                )

            k_prefix, k_residual = _split_residual(k, self._residual_length)
            v_prefix, v_residual = _split_residual(v, self._residual_length)

            layer_data: dict[str, Any] = {
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
            }

            if k_prefix is not None:
                k_q, k_scale, k_mn, k_shape = _compress_layer_k(
                    k_prefix,
                    group_size=self._group_size,
                    k_bits=self._k_bits,
                    quantize_k_cache=self._quantize_k_cache,
                )
                layer_data.update(
                    {
                        "k_quant": k_q.detach().cpu(),
                        "k_scale": k_scale.detach().cpu(),
                        "k_mn": k_mn.detach().cpu(),
                        "k_quant_shape": tuple(k_shape),
                    },
                )
            if k_residual is not None:
                layer_data["k_residual"] = k_residual.detach().cpu()

            if v_prefix is not None:
                v_q, v_scale, v_mn, v_shape = _compress_layer_v(
                    v_prefix,
                    group_size=self._group_size,
                    v_bits=self._v_bits,
                    process_input_fn=self._process_input,
                )
                layer_data.update(
                    {
                        "v_quant": v_q.detach().cpu(),
                        "v_scale": v_scale.detach().cpu(),
                        "v_mn": v_mn.detach().cpu(),
                        "v_quant_shape": tuple(v_shape),
                    },
                )
            if v_residual is not None:
                layer_data["v_residual"] = v_residual.detach().cpu()
            stored_bytes += _layer_payload_bytes(layer_data)
            layers.append(layer_data)

        return {
            "cache_format": cache_format,
            "layers": layers,
            "head_dim": self._head_dim,
            "k_bits": self._k_bits,
            "v_bits": self._v_bits,
            "group_size": self._group_size,
            "residual_length": self._residual_length,
            "dtype": k_tensors[0].dtype,
            "device": k_tensors[0].device,
            "__stored_kv_bytes__": stored_bytes,
            "__metadata_bytes_fixed__": 0,
        }

    def _backend_materialize(self, backend_data: dict, cache_format: str) -> Any:
        k_tensors: list[torch.Tensor] = []
        v_tensors: list[torch.Tensor] = []
        dtype = backend_data["dtype"]
        device = backend_data["device"]
        group_size = backend_data["group_size"]
        k_bits = backend_data["k_bits"]
        v_bits = backend_data["v_bits"]

        for layer in backend_data["layers"]:
            k_shape = torch.Size(layer["k_shape"])
            v_shape = torch.Size(layer["v_shape"])

            if "k_quant" in layer:
                k_quant_shape = torch.Size(layer["k_quant_shape"])
                k_prefix_hat = _decompress_layer_k(
                    layer["k_quant"].to(device=device, dtype=dtype),
                    layer["k_scale"].to(device=device, dtype=dtype),
                    layer["k_mn"].to(device=device, dtype=dtype),
                    k_quant_shape,
                    group_size=group_size,
                    k_bits=k_bits,
                    dequantize_k_cache=self._dequantize_k_cache,
                )
            else:
                k_prefix_hat = None

            k_residual = (
                layer["k_residual"].to(device=device, dtype=dtype)
                if "k_residual" in layer
                else None
            )
            k_hat = _merge_prefix_residual(k_prefix_hat, k_residual)
            if k_hat.shape != k_shape:
                raise ValueError(
                    f"Materialized K shape {k_hat.shape} != expected {k_shape}"
                )

            if "v_quant" in layer:
                v_quant_shape = torch.Size(layer["v_quant_shape"])
                v_prefix_hat = _decompress_layer_v(
                    layer["v_quant"].to(device=device, dtype=dtype),
                    layer["v_scale"].to(device=device, dtype=dtype),
                    layer["v_mn"].to(device=device, dtype=dtype),
                    v_quant_shape,
                    group_size=group_size,
                )
            else:
                v_prefix_hat = None

            v_residual = (
                layer["v_residual"].to(device=device, dtype=dtype)
                if "v_residual" in layer
                else None
            )
            v_hat = _merge_prefix_residual(v_prefix_hat, v_residual)
            if v_hat.shape != v_shape:
                raise ValueError(
                    f"Materialized V shape {v_hat.shape} != expected {v_shape}"
                )

            k_tensors.append(k_hat)
            v_tensors.append(v_hat)

        seq_len = int(k_tensors[0].shape[2])
        return rebuild_cache(k_tensors, v_tensors, cache_format, seq_len)

    def _backend_workspace_bytes(
        self,
        full_kv_bytes: int,
        backend_data: dict,
    ) -> dict:
        stored = int(backend_data.get("__stored_kv_bytes__", 0))
        metadata = int(backend_data.get("__metadata_bytes_fixed__", 0))
        materialized = full_kv_bytes
        temporary = max(full_kv_bytes // 4, 0)
        total = stored + materialized + metadata + temporary

        return {
            "stored_kv_bytes": stored,
            "materialized_working_kv_bytes": materialized,
            "metadata_bytes": metadata,
            "temporary_workspace_bytes": temporary,
            "total_kv_footprint_bytes": total,
        }

    def _cache_for_forward(self, cache: Any, cache_format: str) -> Any:
        if cache_format != "tuple":
            return cache
        try:
            from transformers import DynamicCache  # noqa: PLC0415

            if hasattr(DynamicCache, "from_legacy_cache"):
                return DynamicCache.from_legacy_cache(cache)
        except Exception:
            pass
        return cache

    @torch.no_grad()
    def _get_next_token_id(self, state: FullKVState, backend_data: dict) -> int:
        cache_format = backend_data["cache_format"]
        cache = self._backend_materialize(backend_data, cache_format)
        seq_len = state.seq_len

        if seq_len <= 1:
            last_tok = state.full_sequence_ids[:, -1:].to(self._runtime.device)
            out = self._runtime.forward(last_tok, past_key_values=None)
            return int(out.logits[:, -1, :].argmax(dim=-1).item())

        k_tensors, v_tensors, fmt = extract_kv_tensors(cache)
        prefix_len = seq_len - 1
        k_prefix = [t[..., :prefix_len, :].clone() for t in k_tensors]
        v_prefix = [t[..., :prefix_len, :].clone() for t in v_tensors]
        partial_cache = rebuild_cache(k_prefix, v_prefix, fmt, prefix_len)
        partial_cache = self._cache_for_forward(partial_cache, fmt)

        last_tok = state.full_sequence_ids[:, -1:].to(self._runtime.device)
        out = self._runtime.forward(last_tok, past_key_values=partial_cache)
        return int(out.logits[:, -1, :].argmax(dim=-1).item())


def create_kivi_offline_adapter(
    runtime: ModelRuntime,
    *,
    head_dim: int | None = None,
    k_bits: int = 2,
    v_bits: int = 2,
    group_size: int = 32,
    residual_length: int = 0,
) -> KIVIOfflineAdapter:
    """Factory for the restricted KIVI offline adapter (not in default registry)."""
    return KIVIOfflineAdapter(
        runtime=runtime,
        head_dim=head_dim,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length,
    )
