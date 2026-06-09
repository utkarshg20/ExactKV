#!/usr/bin/env python3
"""V9 Phase D1 scratch inspection for KIVI and KVQuant.

Temporary research script — NOT production adapter code. Does not import ExactKV
compressors or register any backend.

Prerequisites (isolated venvs, outside ExactKV default deps):

KIVI (upstream jy-yuan/KIVI):
    git clone https://github.com/jy-yuan/KIVI.git /tmp/kivi_research
    # Full install needs Python 3.10, torch==2.4.1, flash-attn, CUDA quant ext.
    # Partial inspect without install:
    PYTHONPATH=/tmp/kivi_research python3 scripts/research/kivi_kvquant_phase_d1_inspect.py --backend kivi

KVQuant (upstream SqueezeAILab/KVQuant):
    git clone https://github.com/SqueezeAILab/KVQuant.git /tmp/kvquant_research
    python3 -m venv /tmp/kvquant_venv_test
    /tmp/kvquant_venv_test/bin/pip install -e /tmp/kvquant_research/quant
    python3 scripts/research/kivi_kvquant_phase_d1_inspect.py --backend kvquant

Usage:
    python3 scripts/research/kivi_kvquant_phase_d1_inspect.py [--backend kivi|kvquant|all]
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path


KIVI_REPO = Path("/tmp/kivi_research")
KVQUANT_REPO = Path("/tmp/kvquant_research")


def _qwen_head_dims() -> dict:
    """Public HF config fields — no weight download."""
    return {
        "Qwen/Qwen2.5-0.5B": {
            "hidden_size": 896,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "head_dim": 64,
        },
        "Qwen/Qwen2.5-1.5B": {
            "hidden_size": 1536,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "head_dim": 128,
        },
        "Qwen/Qwen2.5-3B": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
        },
        "Qwen/Qwen2.5-7B": {
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "head_dim": 128,
        },
    }


def inspect_kivi_repo() -> dict:
    result: dict = {"repo_exists": KIVI_REPO.is_dir()}
    if not result["repo_exists"]:
        result["error"] = f"Clone missing: {KIVI_REPO}"
        return result

    model_files = sorted((KIVI_REPO / "models").glob("*_kivi.py"))
    result["model_files"] = [p.name for p in model_files]
    result["qwen_model_file"] = any("qwen" in n.lower() for n in result["model_files"])

    utils_path = KIVI_REPO / "models" / "utils_quant.py"
    result["utils_quant_exists"] = utils_path.is_file()
    if utils_path.is_file():
        text = utils_path.read_text()
        result["simulate_helpers"] = [
            name
            for name in (
                "quantize_by_channel_and_pack_cache",
                "dequantize_by_channel_and_unpack_cache",
                "quantize_and_pack",
                "dequantize_and_unpack",
            )
            if f"def {name}" in text
        ]

    if str(KIVI_REPO) not in sys.path:
        sys.path.insert(0, str(KIVI_REPO))
    try:
        from models.utils_quant import (  # type: ignore[import-untyped]
            dequantize_by_channel_and_unpack_cache,
            quantize_by_channel_and_pack_cache,
        )

        import torch

        k = torch.randn(1, 2, 8, 64)
        qk, ks, kmn = quantize_by_channel_and_pack_cache(k, 32, 2, simulate=True)
        k_hat = dequantize_by_channel_and_unpack_cache(
            qk, 32, k.shape, 2, ks, kmn, simulate=True
        )
        result["k_simulate_roundtrip"] = {
            "ok": True,
            "shape": list(k_hat.shape),
            "mae": float((k - k_hat).abs().mean()),
        }
    except Exception as exc:
        result["k_simulate_roundtrip"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        import kivi  # noqa: F401

        result["kivi_pkg_import"] = True
    except ImportError:
        result["kivi_pkg_import"] = False

    try:
        from quant.new_pack import triton_quantize_and_pack_along_last_dim  # type: ignore

        result["cuda_pack_import"] = True
    except Exception as exc:
        result["cuda_pack_import"] = False
        result["cuda_pack_error"] = f"{type(exc).__name__}: {exc}"

    return result


def inspect_kvquant_pkg() -> dict:
    result: dict = {}
    try:
        from kvquant.simquant_module_quantizer import QuantLinearSim, make_quant_sim

        result["simquant_import"] = True
        result["QuantLinearSim"] = str(inspect.signature(QuantLinearSim.__init__))
    except Exception as exc:
        result["simquant_import"] = False
        result["simquant_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["repo_exists"] = KVQUANT_REPO.is_dir()
    if result["repo_exists"]:
        quant_readme = KVQUANT_REPO / "quant" / "README.md"
        deploy_readme = KVQUANT_REPO / "deployment" / "README.md"
        result["subprojects"] = {
            "gradients": (KVQUANT_REPO / "gradients").is_dir(),
            "quant": (KVQUANT_REPO / "quant").is_dir(),
            "deployment": (KVQUANT_REPO / "deployment").is_dir(),
            "lwm": (KVQUANT_REPO / "lwm").is_dir(),
        }
        result["llama_scripts"] = [
            p.name
            for p in (KVQUANT_REPO / "quant").glob("*.py")
            if "llama" in p.name or "mistral" in p.name or "dbrx" in p.name
        ]
        result["has_qwen_entry_script"] = any(
            "qwen" in n.lower() for n in result["llama_scripts"]
        )
        if quant_readme.is_file():
            result["quant_readme_first_line"] = quant_readme.read_text().splitlines()[0][:80]
        if deploy_readme.is_file():
            result["deployment_requires_cuda_ext"] = "setup_cuda.py" in deploy_readme.read_text()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="KIVI / KVQuant Phase D1 API inspection")
    parser.add_argument(
        "--backend",
        choices=("kivi", "kvquant", "all"),
        default="all",
    )
    args = parser.parse_args()

    print("=== Qwen2.5 head dims (config reference, no download) ===")
    for model, cfg in _qwen_head_dims().items():
        print(f"  {model}: {cfg}")

    if args.backend in ("kivi", "all"):
        print("\n=== KIVI repo inspection ===")
        for k, v in inspect_kivi_repo().items():
            print(f"  {k}: {v}")

    if args.backend in ("kvquant", "all"):
        print("\n=== KVQuant package inspection ===")
        for k, v in inspect_kvquant_pkg().items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
