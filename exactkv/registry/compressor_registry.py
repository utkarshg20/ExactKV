"""Phase H dynamic compressor registry."""
from __future__ import annotations

from typing import Any, Callable

from exactkv.core.compressor_interface import KVCompressor

_REGISTRY: dict[str, Callable[[], KVCompressor]] = {}


def register_compressor(name: str, cls: type[KVCompressor] | Callable[[], KVCompressor]) -> None:
    """Register a ``KVCompressor`` class or factory under ``name``."""
    if not name:
        raise ValueError("compressor name must be non-empty")
    if isinstance(cls, type):
        _REGISTRY[name] = cls
    else:
        _REGISTRY[name] = cls  # type: ignore[assignment]


def list_compressors() -> list[str]:
    """Return sorted registered compressor names."""
    return sorted(_REGISTRY)


def get_compressor(name: str, **kwargs: Any) -> KVCompressor:
    """Instantiate a registered Phase H compressor."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown compressor {name!r}. Available: {list_compressors()}")
    factory = _REGISTRY[name]
    if isinstance(factory, type):
        return factory(**kwargs)
    inst = factory()
    return inst


def _register_builtin_compressors() -> None:
    from exactkv.adapters.kernel_backed_adapter import KernelBackedKVCompressor  # noqa: PLC0415
    from exactkv.adapters.kvquant_adapter import KVQuantKVCompressor  # noqa: PLC0415
    from exactkv.adapters.shard_adapter import ShardKVCompressor  # noqa: PLC0415
    from exactkv.adapters.spectralquant_adapter import SpectralQuantKVCompressor  # noqa: PLC0415
    from exactkv.adapters.shard_real_adapter import ShardRealKVCompressor  # noqa: PLC0415
    from exactkv.adapters.spectralquant_real_adapter import SpectralQuantRealKVCompressor  # noqa: PLC0415
    from exactkv.adapters.turboquant_adapter import TurboQuantKVCompressor  # noqa: PLC0415

    register_compressor("noop", lambda: KernelBackedKVCompressor("noop", "noop"))
    register_compressor("int8", lambda: KernelBackedKVCompressor("int8", "int8"))
    register_compressor("int4_sim", lambda: KernelBackedKVCompressor("int4_sim", "int4"))
    register_compressor("k8_v4_sim", lambda: KernelBackedKVCompressor("k8_v4_sim", "int4"))
    register_compressor("spectralquant", SpectralQuantKVCompressor)
    register_compressor("spectralquant_real", SpectralQuantRealKVCompressor)
    register_compressor("kvquant", KVQuantKVCompressor)
    register_compressor("shard", ShardKVCompressor)
    register_compressor("shard_real", ShardRealKVCompressor)
    register_compressor("turboquant", TurboQuantKVCompressor)


_register_builtin_compressors()
