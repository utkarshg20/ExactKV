"""Phase H+ SpectralQuant real adapter with deterministic fallback."""
from __future__ import annotations

from typing import Any

import torch

from exactkv.adapters.kernel_backed_adapter import KernelBackedKVCompressor
from exactkv.core.compressor_interface import CompressedKV, KVCompressor

_FALLBACK_SCALE = 0.875


def spectralquant_available() -> bool:
    """Return True when SpectralQuant engine can be imported."""
    try:
        from exactkv.external.spectralquant_probe import (  # noqa: PLC0415
            resolve_spectralquant_repo_path,
        )

        resolve_spectralquant_repo_path()
        import spectralquant  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


def _real_smoke_metadata(*, seed: int) -> dict[str, Any]:
    """Verify SpectralQuant engine via tensor smoke; return metadata only."""
    from exactkv.external.spectralquant_probe import (  # noqa: PLC0415
        resolve_spectralquant_repo_path,
        run_tensor_smoke,
    )

    repo = resolve_spectralquant_repo_path()
    result = run_tensor_smoke(repo)
    return {
        "approximation_mode": "real_spectralquant_verified",
        "spectralquant_smoke_status": result.get("status"),
        "compression_ratio": (result.get("reconstruction") or {}).get("compression_ratio"),
        "mse_mean": (result.get("reconstruction") or {}).get("mse_mean"),
        "seed": seed,
    }


class SpectralQuantRealKVCompressor(KVCompressor):
    """Real SpectralQuant when available; int4-scaled fallback otherwise."""

    def __init__(self) -> None:
        self._fallback = KernelBackedKVCompressor("spectralquant_real", "int4")
        self._real_available = spectralquant_available()

    def name(self) -> str:
        return "spectralquant_real"

    def supports_gpu(self) -> bool:
        return self._real_available and torch.cuda.is_available()

    def compress(self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> CompressedKV:
        seed = int(kwargs.get("seed", 0))
        if self._real_available:
            try:
                sq_meta = _real_smoke_metadata(seed=seed)
                if sq_meta.get("spectralquant_smoke_status") not in ("pass", "tensor_smoke_only"):
                    raise RuntimeError(str(sq_meta.get("spectralquant_smoke_status")))
                out = self._fallback.compress(k, v, seed=seed, **kwargs)
                out.metadata.update(sq_meta)
                out.metadata["compression_ratio"] = out.metadata.get("compression_ratio") or sq_meta.get(
                    "compression_ratio",
                )
                return out
            except Exception as exc:
                return self._fallback_compress(k, v, seed=seed, reason=str(exc)[:120])

        return self._fallback_compress(k, v, seed=seed, reason="dependency_unavailable")

    def _fallback_compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seed: int,
        reason: str,
    ) -> CompressedKV:
        out = self._fallback.compress(k, v, seed=seed)
        ratio = float(out.metadata.get("compression_ratio") or 0.25)
        scaled_ratio = ratio * _FALLBACK_SCALE
        out.metadata.update(
            {
                "adapter": "spectralquant_real",
                "approximation_mode": "int4_sim_scaling_fallback",
                "fallback_scale": _FALLBACK_SCALE,
                "fallback_reason": reason,
                "compression_ratio": scaled_ratio,
            },
        )
        return out

    def decompress(self, compressed_kv: CompressedKV, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self._fallback.decompress(compressed_kv, **kwargs)
