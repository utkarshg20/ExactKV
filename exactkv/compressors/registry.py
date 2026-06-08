"""Compressor registry for ExactKV.

Provides a lightweight name → class mapping so that compressors can be
looked up by string name instead of being imported and instantiated by hand.

Built-in registrations are performed in ``exactkv/compressors/__init__.py``,
so importing this module alone gives an empty registry.  Import the package
(``import exactkv.compressors``) to ensure the built-ins are registered.

Usage::

    from exactkv.compressors import get_compressor, list_compressors

    compressor = get_compressor("int8")       # → Int8Compressor()
    names = list_compressors()                # → ["debug_noise", "int8", "noop"]
    get_compressor("unknown")                 # → ValueError

Custom compressors::

    from exactkv.compressors.registry import register_compressor
    register_compressor("my_compressor", MyCompressor)
    compressor = get_compressor("my_compressor")
"""
from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, type] = {}


def register_compressor(name: str, cls: type) -> None:
    """Register a compressor class under ``name``.

    Args:
        name: Identifier string (lower-case, underscore-separated by convention).
        cls:  Class (not instance) that satisfies the ``KVCompressor`` protocol.
              The class must be constructable with no required arguments.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Compressor name must be a non-empty string, got {name!r}")
    _REGISTRY[name] = cls


def get_compressor(name: str, **kwargs: Any) -> Any:
    """Instantiate and return a compressor by registered name.

    Args:
        name:   Registered compressor name.
        **kwargs: Optional keyword arguments forwarded to the class constructor.

    Returns:
        A new instance of the registered compressor class.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        available = list_compressors()
        raise ValueError(
            f"Unknown compressor {name!r}. "
            f"Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_compressors() -> list[str]:
    """Return a sorted list of all registered compressor names."""
    return sorted(_REGISTRY)
