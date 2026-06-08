"""ExactKV compressors package.

Importing this package registers the three built-in compressors so that
``get_compressor("noop")`` etc. work without any extra imports.

Public API::

    from exactkv.compressors import get_compressor, list_compressors, register_compressor
    from exactkv.compressors import NoOpCompressor, Int8Compressor, DebugNoiseCompressor
"""
from exactkv.compressors.debug_noise import DebugNoiseCompressor
from exactkv.compressors.int4_sim import Int4SimCompressor
from exactkv.compressors.int8 import Int8Compressor
from exactkv.compressors.noop import NoOpCompressor
from exactkv.compressors.registry import (
    get_compressor,
    list_compressors,
    register_compressor,
)

# Register built-in compressors (idempotent on re-import)
register_compressor("noop", NoOpCompressor)
register_compressor("int8", Int8Compressor)
register_compressor("int4_sim", Int4SimCompressor)
register_compressor("debug_noise", DebugNoiseCompressor)

__all__ = [
    "NoOpCompressor",
    "Int8Compressor",
    "Int4SimCompressor",
    "DebugNoiseCompressor",
    "register_compressor",
    "get_compressor",
    "list_compressors",
]
