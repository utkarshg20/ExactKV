#!/usr/bin/env python3
"""V12 Phase 1 / 1b — TurboQuant production-fidelity toolchain inspector.

Research-only. Does NOT import ExactKV compressors or modify runtime.

Usage:
    python3 scripts/research/turboquant_production_phase1_inspect.py
    python3 scripts/research/turboquant_production_phase1_inspect.py \
        --workdir /workspace/turboquant_prod_prep --with-refract
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _turboquant_visible() -> bool:
    return importlib.util.find_spec("turboquant") is not None


def inspect_python_path() -> dict:
    import turboquant

    return {
        "importable": True,
        "module": str(Path(turboquant.__file__).resolve()),
        "exports": [x for x in dir(turboquant) if not x.startswith("_")],
        "compressor": "KVCacheCompressor",
    }


def _find_llama_cli(workdir: Path, llama_repo: Path) -> Path | None:
    candidates = [
        workdir / "llama-cpp-turboquant/build-cpu/bin/llama-cli",
        workdir / "llama-cpp-turboquant/build/bin/llama-cli",
        llama_repo / "build-cpu/bin/llama-cli",
        llama_repo / "build/bin/llama-cli",
        Path("/tmp/llama-cpp-turboquant/build/bin/llama-cli"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def inspect_toolchain(workdir: Path, llama_repo: Path) -> dict:
    out: dict = {
        "cmake_installed": shutil.which("cmake") is not None,
        "cmake_version": subprocess.getoutput("cmake --version 2>/dev/null").splitlines()[:1],
        "llama_repo_present": llama_repo.is_dir(),
        "workdir_present": workdir.is_dir(),
        "prep_manifest": str(workdir / "prep_manifest.txt") if (workdir / "prep_manifest.txt").is_file() else None,
    }
    cli = _find_llama_cli(workdir, llama_repo)
    out["llama_cli_exists"] = cli is not None
    out["llama_cli_path"] = str(cli) if cli else None
    if cli:
        help_text = subprocess.getoutput(f"{cli} --help 2>&1")
        for flag in ("ctk", "ctv", "turbo", "cache-type-k", "cache-type-v", "q8_0", "turbo3"):
            out[f"help_flag_{flag}"] = flag.lower() in help_text.lower()
    gguf = workdir / "models/qwen2.5-0.5b-auto.gguf"
    out["gguf_model_exists"] = gguf.is_file()
    out["gguf_model_path"] = str(gguf) if gguf.is_file() else None
    converter = llama_repo / "convert_hf_to_gguf.py"
    out["gguf_converter_exists"] = converter.is_file()
    if (workdir / "prep_manifest.txt").is_file():
        manifest = (workdir / "prep_manifest.txt").read_text()
        out["phase2_proceed"] = "phase2_proceed: YES" in manifest
        out["refract_selftest_pass"] = "refract_selftest: PASS" in manifest
        out["gguf_convert_success"] = "gguf_convert: success" in manifest or gguf.is_file()
    return out


def inspect_refract(llama_bin_dir: str | None) -> dict:
    refract = shutil.which("refract")
    if not refract:
        venv_refract = Path("/workspace/turboquant_prod_prep/venv_refract/bin/refract")
        if venv_refract.is_file():
            refract = str(venv_refract)
        else:
            return {"installed": False}
    env = os.environ.copy()
    if llama_bin_dir:
        env["LLAMA_CPP_BIN_DIR"] = llama_bin_dir
    try:
        proc = subprocess.run(
            [refract, "selftest"],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        return {
            "installed": True,
            "binary": refract,
            "selftest_exit": proc.returncode,
            "selftest_pass": proc.returncode == 0,
            "selftest_tail": "\n".join((proc.stdout + proc.stderr).splitlines()[-6:]),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"installed": True, "binary": refract, "selftest_error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="TurboQuant toolchain inspector")
    parser.add_argument("--workdir", type=Path, default=Path("/workspace/turboquant_prod_prep"))
    parser.add_argument("--llama-repo", type=Path, default=None)
    parser.add_argument("--with-refract", action="store_true")
    args = parser.parse_args()

    llama_repo = args.llama_repo or (args.workdir / "llama-cpp-turboquant")
    if not llama_repo.is_dir():
        llama_repo = Path("/tmp/llama-cpp-turboquant")

    print("=== TurboQuant production toolchain inspection ===")
    print(f"python: {sys.version.split()[0]}")

    if _turboquant_visible():
        print("\n--- Python path ---")
        for k, v in inspect_python_path().items():
            print(f"{k}: {v}")
    else:
        print("\nWARN: turboquant not importable (optional for toolchain prep)")

    print("\n--- Toolchain ---")
    tc = inspect_toolchain(args.workdir, llama_repo)
    for k, v in tc.items():
        print(f"{k}: {v}")

    if args.with_refract:
        bin_dir = None
        if tc.get("llama_cli_path"):
            bin_dir = str(Path(tc["llama_cli_path"]).parent)
        print("\n--- REFRACT ---")
        print(inspect_refract(bin_dir))

    print("\nSee docs/TURBOQUANT_PRODUCTION_TOOLCHAIN_PREP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
