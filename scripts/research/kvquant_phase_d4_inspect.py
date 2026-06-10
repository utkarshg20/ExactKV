#!/usr/bin/env python3
"""V9 Phase D4 scratch inspection for KVQuant RunPod validation.

Temporary research script — NOT production adapter code. Does not import ExactKV
compressors or register any backend.

Usage (local CPU module walk):
    python3 scripts/research/kvquant_phase_d4_inspect.py

Usage (with KVQuant quant package):
    pip install -e /path/to/KVQuant/quant
    python3 scripts/research/kvquant_phase_d4_inspect.py --with-kvquant

Usage (Qwen2.5 structure + optional KVQuant imports):
    python3 scripts/research/kvquant_phase_d4_inspect.py --with-qwen
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path


KVQUANT_REPO = Path("/tmp/kvquant_research")
MODEL = "Qwen/Qwen2.5-0.5B"


def _kvquant_available() -> bool:
    try:
        return importlib.util.find_spec("kvquant") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _qwen_module_walk() -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL)
    result = {
        "model": MODEL,
        "config_model_type": getattr(cfg, "model_type", None),
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "num_layers": cfg.num_hidden_layers,
    }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu"
    )
    result["model_class"] = type(model).__name__

    mt = str(type(model)).lower()
    if "opt" in mt:
        mtype = "opt"
    elif "dbrx" in mt:
        mtype = "dbrx"
    else:
        mtype = "llama"
    result["kvquant_parse_model"] = mtype

    if KVQUANT_REPO.is_dir():
        sys.path.insert(0, str(KVQUANT_REPO / "quant"))
    try:
        from kvquant.modelutils import find_layers

        layer0 = model.model.layers[0]
        full = find_layers(layer0)
        result["layer0_linear_modules"] = sorted(full.keys())
        result["layer0_k_proj"] = "self_attn.k_proj" in full
        result["layer0_v_proj"] = "self_attn.v_proj" in full
    except Exception as exc:
        result["find_layers_error"] = f"{type(exc).__name__}: {exc}"

    quantizer_keys = []
    for i in range(min(3, len(model.model.layers))):
        quantizer_keys.append(f"model.layers.{i}.self_attn.k_proj")
        quantizer_keys.append(f"model.layers.{i}.self_attn.v_proj")
    result["expected_quantizer_key_sample"] = quantizer_keys

    return result


def inspect_kvquant_imports() -> dict:
    result: dict = {"kvquant_pkg": _kvquant_available()}
    if not result["kvquant_pkg"]:
        result["error"] = "pip install -e KVQuant/quant required"
        return result

    from kvquant.simquant_module_quantizer import QuantLinearSim, SimQuant, make_quant_sim

    result["QuantLinearSim"] = True
    result["SimQuant"] = True
    result["make_quant_sim"] = str(inspect.signature(make_quant_sim))
    result["QuantLinearSim_init"] = str(inspect.signature(QuantLinearSim.__init__))

    if KVQUANT_REPO.is_dir():
        result["repo_exists"] = True
        result["llama_simquant"] = (KVQUANT_REPO / "quant" / "llama_simquant.py").is_file()
        result["run_fisher"] = (KVQUANT_REPO / "gradients" / "run-fisher.py").is_file()
        result["deployment_setup_cuda"] = (
            KVQUANT_REPO / "deployment" / "kvquant" / "setup_cuda.py"
        ).is_file()
        try:
            import subprocess

            sha = subprocess.check_output(
                ["git", "-C", str(KVQUANT_REPO), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            result["repo_sha"] = sha[:12] if sha else None
        except Exception:
            result["repo_sha"] = None
    return result


def inspect_blockers() -> dict:
    """Static code-path blockers for Qwen2.5 on RunPod."""
    blockers = []
    notes = []

    if KVQUANT_REPO.is_dir():
        simquant_text = (KVQUANT_REPO / "quant" / "llama_simquant.py").read_text()
        if "use_flash_attention_2=True" in simquant_text:
            blockers.append(
                "llama_simquant.get_model hardcodes use_flash_attention_2=True — "
                "patch to sdpa/eager if flash-attn build fails"
            )
        if "DEV = torch.device('cuda:0')" in simquant_text:
            notes.append("Calibration requires CUDA (cuda:0 hardcoded)")

        fisher_text = (KVQUANT_REPO / "gradients" / "run-fisher.py").read_text()
        if "k_proj.act.grad" in fisher_text:
            blockers.append(
                "run-fisher.py expects Llama-patched k_proj.act hooks — "
                "not present on stock Qwen2 Linear; Fisher optional (fisher=None works)"
            )
        if "set_devices()" in fisher_text:
            blockers.append(
                "run-fisher.py calls model.model.set_devices() — "
                "Llama/Mistral vendored only; use calibration without Fisher first"
            )

        qsim_text = (KVQUANT_REPO / "quant" / "kvquant" / "simquant_module_quantizer.py").read_text()
        if ".cuda()" in qsim_text:
            notes.append("QuantLinearSim hardcodes .cuda() — GPU required for forward")

    return {"blockers": blockers, "notes": notes}


def main() -> int:
    parser = argparse.ArgumentParser(description="KVQuant Phase D4 inspection")
    parser.add_argument("--with-kvquant", action="store_true")
    parser.add_argument("--with-qwen", action="store_true")
    args = parser.parse_args()

    print("=== KVQuant static blockers ===")
    for k, v in inspect_blockers().items():
        print(f"  {k}: {v}")

    if args.with_kvquant or _kvquant_available():
        print("\n=== KVQuant import inspection ===")
        for k, v in inspect_kvquant_imports().items():
            print(f"  {k}: {v}")

    if args.with_qwen:
        print(f"\n=== Qwen module walk ({MODEL}) ===")
        for k, v in _qwen_module_walk().items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
