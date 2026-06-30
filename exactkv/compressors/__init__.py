"""ExactKV compressors package.

Importing this package registers all built-in compressors so that
``get_compressor("noop")`` etc. work without any extra imports.

Public API::

    from exactkv.compressors import get_compressor, list_compressors, register_compressor
    from exactkv.compressors import NoOpCompressor, Int8Compressor, DebugNoiseCompressor

V4 asymmetric compressors (all simulated or real-only; no real bit-packing)::

    from exactkv.compressors import (
        AsymmetricQuantSimCompressor,  # base class
        K8V4SimCompressor,    # k8_v4_sim
        K8V2SimCompressor,    # k8_v2_sim
        K4V8SimCompressor,    # k4_v8_sim
        KFullV4SimCompressor, # k_full_v4_sim
        K4VFullSimCompressor, # k4_v_full_sim
        K8VFullCompressor,    # k8_v_full  (no _sim: real storage only)
        KFullV8Compressor,    # k_full_v8  (no _sim: real storage only)
    )

V6 backend adapter (PoC — no external dependencies)::

    from exactkv.compressors import BackendAdapter, PassThroughBackendAdapter
    compressor = get_compressor("backend_passthrough")
"""
from exactkv.compressors.asymmetric_sim import (
    AsymmetricQuantSimCompressor,
    K4V8SimCompressor,
    K4VFullSimCompressor,
    K8V2SimCompressor,
    K8V4SimCompressor,
    K8VFullCompressor,
    KFullV4SimCompressor,
    KFullV8Compressor,
)
from exactkv.compressors.backend_adapter import BackendAdapter, PassThroughBackendAdapter
from exactkv.compressors.debug_noise import DebugNoiseCompressor
from exactkv.compressors.h2o_sim import H2OSimCompressor
from exactkv.compressors.int4_per_vec_sim import Int4PerVecSimCompressor
from exactkv.compressors.int4_sim import Int4SimCompressor
from exactkv.compressors.int6_sim import Int6SimCompressor
from exactkv.compressors.int8 import Int8Compressor
from exactkv.compressors.layer_aware_sim import (
    K8V4Boundary2V8SimCompressor,
    K8V4Boundary4V8SimCompressor,
    K8V4BoundaryV8SimCompressor,
    LayerAwareVSimCompressor,
)
from exactkv.compressors.noop import NoOpCompressor
from exactkv.compressors.registry import (
    get_compressor,
    list_compressors,
    register_compressor,
)

# V1–V3 symmetric compressors (idempotent on re-import)
register_compressor("noop", NoOpCompressor)
register_compressor("int8", Int8Compressor)
register_compressor("int6_sim", Int6SimCompressor)
register_compressor("int4_per_vec_sim", Int4PerVecSimCompressor)
register_compressor("int4_sim", Int4SimCompressor)
register_compressor("debug_noise", DebugNoiseCompressor)

# V4 asymmetric compressors
# _sim suffix = includes at least one simulated sub-INT8 side (4-bit or 2-bit)
# no _sim     = only real INT8 and/or full precision; is_simulated=False
register_compressor("k8_v4_sim", K8V4SimCompressor)
register_compressor("k8_v2_sim", K8V2SimCompressor)
register_compressor("k4_v8_sim", K4V8SimCompressor)
register_compressor("k_full_v4_sim", KFullV4SimCompressor)
register_compressor("k4_v_full_sim", K4VFullSimCompressor)
register_compressor("k8_v_full", K8VFullCompressor)
register_compressor("k_full_v8", KFullV8Compressor)

# V7 layer-aware simulated V policy (boundary-depth variants)
register_compressor("k8_v4_boundary_v8_sim", K8V4BoundaryV8SimCompressor)
register_compressor("k8_v4_boundary2_v8_sim", K8V4Boundary2V8SimCompressor)
register_compressor("k8_v4_boundary4_v8_sim", K8V4Boundary4V8SimCompressor)

# V6 backend adapter PoC
register_compressor("backend_passthrough", PassThroughBackendAdapter)

# V8 token-dropping compressors (H2O-style sink + recency eviction)
register_compressor("h2o_sim",        H2OSimCompressor)
register_compressor("h2o_sim_75",     lambda: H2OSimCompressor(keep_ratio=0.75))
register_compressor("h2o_sim_25",     lambda: H2OSimCompressor(keep_ratio=0.25))

__all__ = [
    # V1–V3
    "NoOpCompressor",
    "Int8Compressor",
    "Int6SimCompressor",
    "Int4PerVecSimCompressor",
    "Int4SimCompressor",
    "DebugNoiseCompressor",
    # V4
    "AsymmetricQuantSimCompressor",
    "K8V4SimCompressor",
    "K8V2SimCompressor",
    "K4V8SimCompressor",
    "KFullV4SimCompressor",
    "K4VFullSimCompressor",
    "K8VFullCompressor",
    "KFullV8Compressor",
    # V7 layer-aware sim
    "LayerAwareVSimCompressor",
    "K8V4BoundaryV8SimCompressor",
    "K8V4Boundary2V8SimCompressor",
    "K8V4Boundary4V8SimCompressor",
    # V6 backend adapter
    "BackendAdapter",
    "PassThroughBackendAdapter",
    # V8 token-dropping
    "H2OSimCompressor",
    # registry
    "register_compressor",
    "get_compressor",
    "list_compressors",
]
