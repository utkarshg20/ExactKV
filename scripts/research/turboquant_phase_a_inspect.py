#!/usr/bin/env python3
"""V9 Phase A scratch inspection for TurboQuant / TurboQuant+.

Temporary research script — NOT production adapter code. Does not import ExactKV
compressors or register any backend.

Prerequisites (isolated venv, outside ExactKV default deps):
    git clone https://github.com/TheTom/turboquant_plus.git /tmp/turboquant_plus_research
    cd /tmp/turboquant_plus_research
    python3 -m venv .venv && source .venv/bin/activate
    pip install numpy scipy
    # optional: pip install -e ".[dev]"  # installs refract-llm only, not turboquant wheel

Usage:
    PYTHONPATH=/tmp/turboquant_plus_research python3 scripts/research/turboquant_phase_a_inspect.py

Does not download HF models unless --with-hf is passed.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path


def _repo_on_path() -> bool:
    try:
        import turboquant  # noqa: F401

        return True
    except ImportError:
        return False


def inspect_exports() -> dict:
    import turboquant
    from turboquant.kv_cache import CompressedKVCache, KVCacheCompressor

    exports = [x for x in dir(turboquant) if not x.startswith("_")]
    sig_compress = str(inspect.signature(KVCacheCompressor.compress))
    sig_decompress = str(inspect.signature(KVCacheCompressor.decompress))
    return {
        "module_path": str(Path(turboquant.__file__).resolve()),
        "exports": exports,
        "KVCacheCompressor.compress": sig_compress,
        "KVCacheCompressor.decompress": sig_decompress,
        "CompressedKVCache_fields": [
            f.name for f in CompressedKVCache.__dataclass_fields__.values()
        ],
    }


def smoke_roundtrip(head_dim: int = 64) -> dict:
    import numpy as np
    from turboquant.kv_cache import KVCacheCompressor

    compressor = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3)
    k = np.random.randn(2, 2, 4, head_dim).astype(np.float32)
    v = np.random.randn(2, 2, 4, head_dim).astype(np.float32)
    compressed = compressor.compress(k, v)
    k_hat, v_hat = compressor.decompress(compressed)
    return {
        "head_dim": head_dim,
        "input_shape": list(k.shape),
        "output_shape": list(k_hat.shape),
        "k_mse": float(np.mean((k - k_hat) ** 2)),
        "v_mse": float(np.mean((v - v_hat) ** 2)),
        "compressed_layers": compressed.num_layers,
        "k_bit_width": compressed.k_bit_width,
        "v_bit_width": compressed.v_bit_width,
    }


def check_qwen_head_dims() -> dict:
    """Document Qwen2.5 head dims without downloading weights."""
    # From public HF configs (inspect only).
    return {
        "Qwen/Qwen2.5-0.5B": {
            "hidden_size": 896,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "note": "Used in turboquant_plus benchmarks/benchmark_ppl_tq_vs_rq.py",
        },
        "Qwen/Qwen3-1.7B": {
            "head_dim": 128,
            "note": "Used in benchmarks/validate_real_model.py",
        },
    }


def optional_hf_import() -> dict:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "hf_bridge_available": True,
        }
    except ImportError as exc:
        return {"hf_bridge_available": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="TurboQuant Phase A API inspection")
    parser.add_argument(
        "--with-hf",
        action="store_true",
        help="Also report torch/transformers versions if installed",
    )
    args = parser.parse_args()

    print("=== TurboQuant Phase A inspection ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"PYTHONPATH turboquant visible: {_repo_on_path()}")

    if not _repo_on_path():
        print(
            "\nERROR: turboquant not importable. Clone TheTom/turboquant_plus and set "
            "PYTHONPATH to the repo root (turboquant is dev-only, not a pip wheel)."
        )
        return 1

    info = inspect_exports()
    print("\n--- API surface ---")
    for key, val in info.items():
        print(f"{key}: {val}")

    print("\n--- Smoke roundtrip (head_dim=64, Qwen2.5-0.5B shape) ---")
    print(smoke_roundtrip(64))

    print("\n--- Smoke roundtrip (head_dim=128) ---")
    print(smoke_roundtrip(128))

    print("\n--- Qwen head_dim reference ---")
    print(check_qwen_head_dims())

    if args.with_hf:
        print("\n--- Optional HF bridge ---")
        print(optional_hf_import())

    print("\nDone. See docs/TURBOQUANT_INTEGRATION_RESEARCH.md for Phase A findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
