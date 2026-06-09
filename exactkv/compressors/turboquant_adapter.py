"""Restricted TurboQuant Python adapter (isolated [turboquant] env only).

NOT registered in the default compressor registry.  The upstream ``turboquant``
package is dev-only inside TheTom/turboquant_plus and is imported lazily when
``TurboQuantPythonAdapter`` is constructed.

Scope: offline NumPy ``KVCacheCompressor`` bridge only.  No llama.cpp, MLX,
GGUF, or production TurboQuant serving runtime.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from exactkv.cache.utils import extract_kv_tensors, kv_total_bytes, rebuild_cache
from exactkv.compressors.backend_adapter import BackendAdapter
from exactkv.compressors.base import CompressorCapabilities
from exactkv.cache.full_state import FullKVState
from exactkv.runtime.model_runtime import ModelRuntime


def _import_turboquant() -> Any:
    """Lazy import of the dev-only upstream turboquant package."""
    try:
        import turboquant  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "turboquant is not importable. Clone https://github.com/TheTom/turboquant_plus "
            "and set PYTHONPATH to the repo root (or install numpy+scipy in .venv-turboquant). "
            "The turboquant Python package is not published on PyPI."
        ) from exc
    return turboquant


def _discover_backend_version() -> str:
    import importlib.metadata
    import os

    try:
        import turboquant  # noqa: PLC0415

        root = os.path.dirname(os.path.dirname(os.path.abspath(turboquant.__file__)))
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
        pass
    try:
        return importlib.metadata.version("refract-llm")
    except importlib.metadata.PackageNotFoundError:
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


def _tensors_to_numpy_stack(
    k_tensors: list[torch.Tensor],
    v_tensors: list[torch.Tensor],
    expected_head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-layer HF tensors to TurboQuant layout (L, H, S, D)."""
    if len(k_tensors) != len(v_tensors):
        raise ValueError(
            f"K/V layer count mismatch: {len(k_tensors)} vs {len(v_tensors)}"
        )
    if not k_tensors:
        raise ValueError("Empty KV tensor list")

    k_layers: list[np.ndarray] = []
    v_layers: list[np.ndarray] = []
    ref_shape: tuple[int, ...] | None = None

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
        if head_dim != expected_head_dim:
            raise ValueError(
                f"Layer {layer_idx}: head_dim={head_dim} != expected {expected_head_dim}"
            )
        if ref_shape is None:
            ref_shape = (heads, seq_len, head_dim)
        elif (heads, seq_len, head_dim) != ref_shape:
            raise ValueError(
                f"Layer {layer_idx}: shape {(heads, seq_len, head_dim)} != "
                f"reference {ref_shape}"
            )
        k_layers.append(k.squeeze(0).detach().cpu().float().numpy())
        v_layers.append(v.squeeze(0).detach().cpu().float().numpy())

    return np.stack(k_layers), np.stack(v_layers)


def _numpy_stack_to_tensors(
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert TurboQuant layout back to per-layer HF tensors."""
    if k_cache.shape != v_cache.shape:
        raise ValueError(f"K/V numpy shape mismatch: {k_cache.shape} vs {v_cache.shape}")
    if k_cache.ndim != 4:
        raise ValueError(f"Expected 4D numpy cache, got {k_cache.shape}")

    k_tensors = [
        torch.from_numpy(k_cache[layer]).to(dtype=dtype, device=device).unsqueeze(0)
        for layer in range(k_cache.shape[0])
    ]
    v_tensors = [
        torch.from_numpy(v_cache[layer]).to(dtype=dtype, device=device).unsqueeze(0)
        for layer in range(v_cache.shape[0])
    ]
    return k_tensors, v_tensors


def _numpy_payload_bytes(obj: Any) -> int:
    """Recursively sum nbytes of numpy arrays in a compressed payload."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sum(
            _numpy_payload_bytes(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.name != "bit_width"
        )
    if isinstance(obj, (list, tuple)):
        return sum(_numpy_payload_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_numpy_payload_bytes(v) for v in obj.values())
    return 0


def _polar_quant_metadata_bytes(polar_quant: Any) -> int:
    total = 0
    rotation = getattr(polar_quant, "rotation", None)
    centroids = getattr(polar_quant, "centroids", None)
    if isinstance(rotation, np.ndarray):
        total += int(rotation.nbytes)
    if isinstance(centroids, np.ndarray):
        total += int(centroids.nbytes)
    return total


def _compressor_metadata_bytes(compressor: Any) -> int:
    total = _polar_quant_metadata_bytes(compressor.k_quantizer.polar_quant)
    total += _polar_quant_metadata_bytes(compressor.v_quantizer.polar_quant)
    qjl = getattr(compressor.k_quantizer, "qjl", None)
    if qjl is not None:
        proj = getattr(qjl, "projection", None)
        if isinstance(proj, np.ndarray):
            total += int(proj.nbytes)
    return total


def _make_compressor(
    *,
    head_dim: int,
    k_bits: int,
    v_bits: int,
    seed: int,
    norm_correction: bool,
) -> Any:
    turboquant = _import_turboquant()
    from turboquant.kv_cache import KVCacheCompressor  # noqa: PLC0415

    return KVCacheCompressor(
        head_dim=head_dim,
        k_bits=k_bits,
        v_bits=v_bits,
        seed=seed,
        norm_correction=norm_correction,
    )


class TurboQuantPythonAdapter(BackendAdapter):
    """Offline TurboQuant Python adapter using ``KVCacheCompressor``.

    Compresses cloned HF KV tensors via NumPy, stores ``CompressedKVCache``,
    and materialises dequantised torch tensors for the draft path only.
    Verification always uses the authoritative full-precision ``FullKVState``.
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        *,
        head_dim: int | None = None,
        k_bits: int = 3,
        v_bits: int = 3,
        seed: int = 42,
        norm_correction: bool = True,
    ) -> None:
        _import_turboquant()  # fail fast with clear message

        self._runtime = runtime
        self._head_dim = head_dim if head_dim is not None else _infer_head_dim(runtime.model)
        self._k_bits = k_bits
        self._v_bits = v_bits
        self._seed = seed
        self._norm_correction = norm_correction

        self._compressor = _make_compressor(
            head_dim=self._head_dim,
            k_bits=k_bits,
            v_bits=v_bits,
            seed=seed,
            norm_correction=norm_correction,
        )
        self._metadata_bytes_fixed = _compressor_metadata_bytes(self._compressor)

        k_label = f"turboquant_k{k_bits}"
        v_label = f"turboquant_v{v_bits}"
        asymmetric = k_bits != v_bits

        self.name = "turboquant_python"
        self.capabilities = CompressorCapabilities(
            name="turboquant_python",
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
            backend_name="turboquant_plus_python",
            backend_version=_discover_backend_version(),
            adapter_name="TurboQuantPythonAdapter",
            adapter_version="0.1.0",
            notes=(
                "Restricted V9 Phase B Python TurboQuant adapter ([turboquant] env only). "
                "Wraps upstream dev-only KVCacheCompressor (NumPy PolarQuant path). "
                "Not llama.cpp, not MLX, not GGUF, not production TurboQuant serving runtime. "
                "Not in the default compressor registry. "
                "supports_real_bytes_claim=False: Python prototype stores int64 indices and "
                "float metadata arrays, not packed-bit llama.cpp turbo formats. "
                "Byte accounting counts actual numpy payloads honestly. "
                "Draft may diverge from full KV; verification uses authoritative full state only. "
                "No throughput, latency, or production-readiness claims."
            ),
        )

    def _backend_compress(
        self,
        k_tensors: list[torch.Tensor],
        v_tensors: list[torch.Tensor],
        cache_format: str,
    ) -> dict:
        k_np, v_np = _tensors_to_numpy_stack(k_tensors, v_tensors, self._head_dim)
        compressed = self._compressor.compress(k_np, v_np)
        stored_bytes = _numpy_payload_bytes(compressed)

        return {
            "cache_format": cache_format,
            "compressed": compressed,
            "head_dim": self._head_dim,
            "k_bits": self._k_bits,
            "v_bits": self._v_bits,
            "seed": self._seed,
            "norm_correction": self._norm_correction,
            "dtype": k_tensors[0].dtype,
            "device": k_tensors[0].device,
            "__stored_kv_bytes__": stored_bytes,
            "__metadata_bytes_fixed__": self._metadata_bytes_fixed,
        }

    def _backend_materialize(self, backend_data: dict, cache_format: str) -> Any:
        compressor = _make_compressor(
            head_dim=backend_data["head_dim"],
            k_bits=backend_data["k_bits"],
            v_bits=backend_data["v_bits"],
            seed=backend_data["seed"],
            norm_correction=backend_data["norm_correction"],
        )
        k_np, v_np = compressor.decompress(backend_data["compressed"])
        k_tensors, v_tensors = _numpy_stack_to_tensors(
            k_np,
            v_np,
            dtype=backend_data["dtype"],
            device=backend_data["device"],
        )
        seq_len = int(k_np.shape[2])
        return rebuild_cache(k_tensors, v_tensors, cache_format, seq_len)

    def _backend_workspace_bytes(
        self,
        full_kv_bytes: int,
        backend_data: dict,
    ) -> dict:
        stored = int(backend_data.get("__stored_kv_bytes__", 0))
        metadata = int(backend_data.get("__metadata_bytes_fixed__", self._metadata_bytes_fixed))

        # Materialize always dequantises to full-precision working tensors.
        materialized = full_kv_bytes
        # Conservative scratch for numpy decompress + torch conversion.
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
        """Ensure past_key_values are forward-compatible (DynamicCache on HF >=5)."""
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
        """Draft-path prediction from materialised compressed KV (may differ from full KV)."""
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


def create_turboquant_python_adapter(
    runtime: ModelRuntime,
    *,
    head_dim: int | None = None,
    k_bits: int = 3,
    v_bits: int = 3,
    seed: int = 42,
    norm_correction: bool = True,
) -> TurboQuantPythonAdapter:
    """Factory for the restricted TurboQuant Python adapter (not in default registry)."""
    return TurboQuantPythonAdapter(
        runtime=runtime,
        head_dim=head_dim,
        k_bits=k_bits,
        v_bits=v_bits,
        seed=seed,
        norm_correction=norm_correction,
    )
