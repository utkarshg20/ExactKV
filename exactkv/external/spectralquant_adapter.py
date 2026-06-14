"""Restricted SpectralQuant experimental BackendAdapter (Experiment 044).

Factory-only — NOT in default compressor registry. Compresses cloned post-RoPE
HF K/V via external SpectralQuantEngine, materialises dequant tensors for draft.
Verification uses authoritative full-precision FullKVState unchanged.
"""
from __future__ import annotations

from typing import Any

import torch

from exactkv.cache.full_state import FullKVState
from exactkv.cache.utils import extract_kv_tensors, rebuild_cache
from exactkv.compressors.backend_adapter import BackendAdapter
from exactkv.compressors.base import CompressorCapabilities
from exactkv.external.spectralquant_real_kv import (
    CalibrationConfig,
    decompress_keys_layer,
    default_calibration_prompts,
    ensure_spectralquant_path,
    run_minimal_calibration,
)
from exactkv.runtime.model_runtime import ModelRuntime

MEMORY_CLAIM_NOTE = (
    "SpectralQuantExperimentalAdapter compresses K/V then immediately materialises "
    "full dequant tensors for the draft forward. stored_kv_bytes counts compressed "
    "quant indices on CPU; materialized_working_kv_bytes equals full-precision KV "
    "layout during draft. total_kv_footprint_bytes is conservative accounting — "
    "not measured peak GPU memory. No active memory savings claim."
)


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.element_size() * obj.nelement())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    return 0


def _cv_to_storage(cv: Any) -> dict[str, Any]:
    return {
        "semantic_indices": cv.semantic_indices.detach().cpu(),
        "tail_indices": cv.tail_indices.detach().cpu(),
        "d_eff": int(cv.d_eff),
        "head_dim": int(cv.head_dim),
        "b_high": int(cv.b_high),
        "b_low": int(cv.b_low),
        "original_shape": tuple(cv.original_shape),
        "actual_bits_used": int(cv.actual_bits_used),
        "mse": float(cv.mse) if cv.mse is not None else None,
    }


def _cv_from_storage(d: dict[str, Any], device: torch.device) -> Any:
    from spectralquant.nonuniform_quantization import CompressedVector

    return CompressedVector(
        semantic_indices=d["semantic_indices"].to(device=device),
        tail_indices=d["tail_indices"].to(device=device),
        d_eff=d["d_eff"],
        head_dim=d["head_dim"],
        b_high=d["b_high"],
        b_low=d["b_low"],
        original_shape=torch.Size(d["original_shape"]),
        actual_bits_used=d["actual_bits_used"],
        mse=d["mse"],
    )


def _heads_dict_to_compressed(
    stored: dict[int, dict[str, Any]],
    device: torch.device,
) -> dict[int, Any]:
    return {int(h): _cv_from_storage(d, device) for h, d in stored.items()}


class SpectralQuantExperimentalAdapter(BackendAdapter):
    """Factory-only SpectralQuant tensor adapter using SpectralQuantEngine."""

    name: str = "spectralquant_experimental"

    def __init__(
        self,
        runtime: ModelRuntime,
        engine: Any,
        *,
        calibration_config: CalibrationConfig | None = None,
    ) -> None:
        self._runtime = runtime
        self._engine = engine
        self._calibration_config = calibration_config or CalibrationConfig()
        self._device = next(runtime.model.parameters()).device
        self._dtype = next(runtime.model.parameters()).dtype

        self.capabilities = CompressorCapabilities(
            name=self.name,
            compressor_type="quantization",
            is_simulated=False,
            supports_real_bytes_claim=False,
            supports_token_dropping=False,
            supports_quantization=True,
            key_bit_width=None,
            value_bit_width=None,
            asymmetric=False,
            backend_name="spectralquant",
            backend_version="external",
            adapter_name="SpectralQuantExperimentalAdapter",
            adapter_version="0.1.0",
            notes=(
                "Restricted experimental SpectralQuant adapter (factory-only). "
                "Requires external SPECTRALQUANT_REPO_PATH clone and minimal "
                "EigenspectralCalibrator calibration. NOT in default registry. "
                "Compress/decompress per-layer K/V tensors; materialises full "
                "dequant cache for draft only. Verification uses full KV. "
                "supports_real_bytes_claim=False. No speed/memory/serving claims."
            ),
        )

    def _backend_compress(
        self,
        k_tensors: list[torch.Tensor],
        v_tensors: list[torch.Tensor],
        cache_format: str,
    ) -> dict:
        if len(k_tensors) != len(v_tensors):
            raise ValueError(f"K/V layer mismatch: {len(k_tensors)} vs {len(v_tensors)}")
        layers: list[dict[str, Any]] = []
        stored_bytes = 0
        for layer_idx, (k, v) in enumerate(zip(k_tensors, v_tensors)):
            ck = self._engine.compress_keys(k, layer_idx=layer_idx)
            cv = self._engine.compress_values(v, layer_idx=layer_idx)
            ck_stored = {h: _cv_to_storage(x) for h, x in ck.items()}
            cv_stored = {h: _cv_to_storage(x) for h, x in cv.items()}
            layer_payload = {
                "compressed_keys": ck_stored,
                "compressed_values": cv_stored,
            }
            stored_bytes += _tensor_bytes(ck_stored) + _tensor_bytes(cv_stored)
            layers.append(layer_payload)
        return {
            "cache_format": cache_format,
            "layers": layers,
            "dtype": str(k_tensors[0].dtype),
            "device": str(k_tensors[0].device),
            "__stored_kv_bytes__": stored_bytes,
            "__metadata_bytes_fixed__": 0,
        }

    def _backend_materialize(self, backend_data: dict, cache_format: str) -> Any:
        k_tensors: list[torch.Tensor] = []
        v_tensors: list[torch.Tensor] = []
        device = torch.device(backend_data["device"])
        for layer_idx, layer in enumerate(backend_data["layers"]):
            ck = _heads_dict_to_compressed(layer["compressed_keys"], device)
            cv = _heads_dict_to_compressed(layer["compressed_values"], device)
            k_hat = decompress_keys_layer(self._engine, ck, layer_idx)
            v_hat = self._engine.decompress_values(cv, layer_idx=layer_idx)
            k_tensors.append(k_hat.to(dtype=self._dtype))
            v_tensors.append(v_hat.to(dtype=self._dtype))
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
        partial = rebuild_cache(k_prefix, v_prefix, fmt, prefix_len)
        partial = self._cache_for_forward(partial, fmt)
        last_tok = state.full_sequence_ids[:, -1:].to(self._runtime.device)
        out = self._runtime.forward(last_tok, past_key_values=partial)
        return int(out.logits[:, -1, :].argmax(dim=-1).item())


def create_spectralquant_experimental_adapter(
    runtime: ModelRuntime,
    *,
    repo_path: Any | None = None,
    calibration_prompts: list[str] | None = None,
    calibration_config: CalibrationConfig | None = None,
) -> SpectralQuantExperimentalAdapter:
    """Factory for restricted SpectralQuant adapter (not in default registry)."""
    ensure_spectralquant_path(repo_path)
    prompts = calibration_prompts or default_calibration_prompts()
    cfg = calibration_config or CalibrationConfig()
    _calibrator, engine = run_minimal_calibration(
        runtime.model,
        runtime.tokenizer,
        prompts,
        config=cfg,
    )
    return SpectralQuantExperimentalAdapter(
        runtime,
        engine,
        calibration_config=cfg,
    )
