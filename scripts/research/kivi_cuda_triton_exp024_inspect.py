#!/usr/bin/env python3
"""V12 Phase 4 — KIVI CUDA/Triton packed-path feasibility inspector (Exp 024).

Research-only. Does not modify ExactKV runtime or register compressors.
No timing, throughput, latency, speedup, runtime_seconds, or active_gpu_kv_bytes.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_FORBIDDEN_FIELDS = frozenset({
    "tokens_per_second",
    "throughput",
    "latency",
    "speedup",
    "runtime_seconds",
    "active_gpu_kv_bytes",
})

QWEN05B_KV_SHAPE = (1, 2, 32, 64)  # batch, num_kv_heads, seq, head_dim
GROUP_SIZE = 32
NUM_BITS = 2


def _assert_no_forbidden(obj: Any, path: str = "report") -> None:
    if isinstance(obj, dict):
        hits = _FORBIDDEN_FIELDS & obj.keys()
        if hits:
            raise ValueError(f"Forbidden fields {hits} in {path}")
        for k, v in obj.items():
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden(item, f"{path}[{i}]")


def _cosine_sim(a, b) -> float:
    import torch

    fa = a.float().reshape(-1)
    fb = b.float().reshape(-1)
    if fa.numel() == 0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(fa, fb, dim=0).item())


def _recon_stats(orig, recon) -> dict[str, Any]:
    import torch

    diff = (orig.float() - recon.float()).abs()
    return {
        "shape": list(recon.shape),
        "dtype": str(recon.dtype),
        "device": str(recon.device),
        "mae": float(diff.mean().item()),
        "max_abs_error": float(diff.max().item()),
        "cosine_similarity": _cosine_sim(orig, recon),
    }


@dataclass
class SmokeResult:
    name: str
    ok: bool
    path: str = ""
    error: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


def _try_import(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {"ok": True, "module": name, "file": getattr(mod, "__file__", "")}
    except Exception as exc:
        return {"ok": False, "module": name, "error": f"{type(exc).__name__}: {exc}"}


def _inspect_utils_quant(k_tensor, v_tensor) -> list[SmokeResult]:
    import torch

    results: list[SmokeResult] = []
    from models.utils_quant import (  # type: ignore[import-untyped]
        dequantize_by_channel_and_unpack_cache,
        dequantize_and_unpack,
        quantize_and_pack,
        quantize_by_channel_and_pack_cache,
    )

    # K simulate=True
    try:
        qk, ks, kmn = quantize_by_channel_and_pack_cache(
            k_tensor, GROUP_SIZE, NUM_BITS, simulate=True
        )
        k_hat = dequantize_by_channel_and_unpack_cache(
            qk, GROUP_SIZE, k_tensor.shape, NUM_BITS, ks, kmn, simulate=True
        )
        results.append(
            SmokeResult(
                name="k_utils_quant_simulate_true",
                ok=True,
                path="utils_quant",
                stats=_recon_stats(k_tensor, k_hat),
            )
        )
    except Exception as exc:
        results.append(
            SmokeResult(
                name="k_utils_quant_simulate_true",
                ok=False,
                path="utils_quant",
                error=f"{type(exc).__name__}: {exc}",
            )
        )

    # K simulate=False (needs dequant_cuda)
    try:
        qk, ks, kmn = quantize_by_channel_and_pack_cache(
            k_tensor, GROUP_SIZE, NUM_BITS, simulate=False
        )
        k_hat = dequantize_by_channel_and_unpack_cache(
            qk, GROUP_SIZE, k_tensor.shape, NUM_BITS, ks, kmn, simulate=False
        )
        results.append(
            SmokeResult(
                name="k_utils_quant_simulate_false",
                ok=True,
                path="utils_quant+dequant_cuda",
                stats=_recon_stats(k_tensor, k_hat),
            )
        )
    except Exception as exc:
        results.append(
            SmokeResult(
                name="k_utils_quant_simulate_false",
                ok=False,
                path="utils_quant+dequant_cuda",
                error=f"{type(exc).__name__}: {exc}",
            )
        )

    # V simulate=True (upstream may require cuda)
    try:
        qv, vs, vmn = quantize_and_pack(v_tensor, GROUP_SIZE, NUM_BITS, simulate=True)
        v_hat = dequantize_and_unpack(
            qv, GROUP_SIZE, v_tensor.shape, NUM_BITS, vs, vmn, simulate=True
        )
        results.append(
            SmokeResult(
                name="v_utils_quant_simulate_true",
                ok=True,
                path="utils_quant",
                stats=_recon_stats(v_tensor, v_hat),
            )
        )
    except Exception as exc:
        results.append(
            SmokeResult(
                name="v_utils_quant_simulate_true",
                ok=False,
                path="utils_quant",
                error=f"{type(exc).__name__}: {exc}",
            )
        )

    return results


def _inspect_new_pack(k_tensor, v_tensor) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    try:
        import quant.new_pack as npk  # type: ignore[import-untyped]
    except Exception as exc:
        return [
            SmokeResult(
                name="new_pack_import",
                ok=False,
                path="quant.new_pack",
                error=f"{type(exc).__name__}: {exc}",
            )
        ]

    # Discover callable pack/unpack pairs
    candidates = [
        ("k_quant_and_pack_kcache", "k_unpack_and_dequant_kcache", k_tensor),
        ("quant_and_pack_kcache", "unpack_and_dequant_kcache", k_tensor),
        ("quant_and_pack_vcache", "unpack_and_dequant_vcache", v_tensor),
    ]
    for pack_name, unpack_name, tensor in candidates:
        if not hasattr(npk, pack_name) or not hasattr(npk, unpack_name):
            results.append(
                SmokeResult(
                    name=f"{pack_name}",
                    ok=False,
                    path="new_pack",
                    error=f"missing {pack_name} or {unpack_name}",
                )
            )
            continue
        pack_fn = getattr(npk, pack_name)
        unpack_fn = getattr(npk, unpack_name)
        try:
            sig = inspect.signature(pack_fn)
            kwargs: dict[str, Any] = {}
            if "num_bits" in sig.parameters:
                kwargs["num_bits"] = NUM_BITS
            if "bits" in sig.parameters:
                kwargs["bits"] = NUM_BITS
            if "group_size" in sig.parameters:
                kwargs["group_size"] = GROUP_SIZE
            if kwargs:
                packed = pack_fn(tensor, **kwargs)
            elif len(sig.parameters) >= 3:
                packed = pack_fn(tensor, GROUP_SIZE, NUM_BITS)
            else:
                packed = pack_fn(tensor)
            if isinstance(packed, tuple) and len(packed) == 3:
                code, scale, mn = packed
                unpack_sig = inspect.signature(unpack_fn)
                if len(unpack_sig.parameters) >= 5:
                    recon = unpack_fn(code, scale, mn, GROUP_SIZE, NUM_BITS)
                else:
                    recon = unpack_fn(code, scale, mn)
            elif isinstance(packed, tuple):
                recon = unpack_fn(*packed)
            else:
                recon = unpack_fn(packed, tensor.shape)
            results.append(
                SmokeResult(
                    name=pack_name,
                    ok=True,
                    path="new_pack",
                    stats=_recon_stats(tensor, recon),
                )
            )
        except Exception as exc:
            results.append(
                SmokeResult(
                    name=pack_name,
                    ok=False,
                    path="new_pack",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if hasattr(npk, "triton_quantize_and_pack_along_last_dim"):
        try:
            fn = npk.triton_quantize_and_pack_along_last_dim
            packed = fn(k_tensor, GROUP_SIZE, NUM_BITS)
            results.append(
                SmokeResult(
                    name="triton_quantize_and_pack_along_last_dim",
                    ok=True,
                    path="new_pack.triton",
                    stats={"packed_type": type(packed).__name__},
                )
            )
        except Exception as exc:
            results.append(
                SmokeResult(
                    name="triton_quantize_and_pack_along_last_dim",
                    ok=False,
                    path="new_pack.triton",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def _classify(
    imports: dict[str, Any],
    smokes: list[SmokeResult],
) -> dict[str, Any]:
    new_pack_ok = any(s.ok and s.path.startswith("new_pack") and "import" not in s.name for s in smokes)
    k_sim_false_ok = any(s.name == "k_utils_quant_simulate_false" and s.ok for s in smokes)
    llama_kivi_ok = imports.get("LlamaForCausalLM_KIVI", {}).get("ok", False)
    qwen_model = imports.get("qwen_kivi_model_file", False)

    if new_pack_ok or k_sim_false_ok:
        if llama_kivi_ok and qwen_model:
            decision = "A_packed_path_go"
            rationale = "Packed quant/dequant works; Qwen model integration appears feasible."
        else:
            decision = "B_restricted_go"
            rationale = (
                "CUDA/Triton tensor pack/unpack works on Qwen-shaped tensors; "
                "no upstream Qwen KIVI model — future factory-only tensor bridge only."
            )
    else:
        decision = "C_no_go_for_now"
        rationale = (
            "Packed CUDA/Triton path not exercised successfully; "
            "remain on Exp 009 offline simulate adapter."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "backend_adapter_packed": "no_go" if decision == "C_no_go_for_now" else "restricted_future_only",
        "replaces_experiment_009": False,
        "llama_kivi_model_import": llama_kivi_ok,
    }


def run_inspection(
    *,
    kivi_repo: Path,
    workdir: Path,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Exp 024 inspect requires CUDA (RunPod GPU).")

    repo = kivi_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    quant_path = repo / "quant"
    if quant_path.is_dir() and str(quant_path) not in sys.path:
        sys.path.insert(0, str(quant_path))

    env: dict[str, Any] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import triton

        env["triton"] = triton.__version__
    except Exception as exc:
        env["triton_error"] = str(exc)

    kivi_sha = ""
    sha_file = workdir / "kivi_repo_sha.txt"
    if sha_file.is_file():
        kivi_sha = sha_file.read_text().strip()
    elif (repo / ".git").is_dir():
        kivi_sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()[:12]

    imports = {
        "utils_quant": _try_import("models.utils_quant"),
        "dequant_cuda": _try_import("dequant_cuda"),
        "new_pack": _try_import("quant.new_pack"),
        "kivi_gemv": _try_import("kivi_gemv"),
    }
    imports["qwen_kivi_model_file"] = (repo / "models" / "qwen_kivi.py").is_file()
    imports["llama_kivi_file"] = (repo / "models" / "llama_kivi.py").is_file()
    imports["LlamaForCausalLM_KIVI"] = _try_import("models.llama_kivi")

    device = torch.device("cuda")
    k = torch.randn(*QWEN05B_KV_SHAPE, device=device, dtype=torch.float16)
    v = torch.randn(*QWEN05B_KV_SHAPE, device=device, dtype=torch.float16)

    smokes: list[SmokeResult] = []
    smokes.extend(_inspect_utils_quant(k, v))
    smokes.extend(_inspect_new_pack(k, v))

    smoke_dicts = [asdict(s) for s in smokes]
    classification = _classify(imports, smokes)

    report = {
        "experiment": "024_kivi_cuda_triton_feasibility",
        "experiment_class": "kivi_cuda_triton_feasibility",
        "kivi_repo": str(repo),
        "kivi_sha": kivi_sha,
        "workdir": str(workdir),
        "environment": env,
        "imports": imports,
        "qwen_shape": {
            "tensor_shape": list(QWEN05B_KV_SHAPE),
            "num_kv_heads": 2,
            "head_dim": 64,
            "dtype": "float16",
            "device": "cuda",
        },
        "packed_roundtrip_smoke": smoke_dicts,
        "classification": classification,
    }
    _assert_no_forbidden(report)
    return report


def generate_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    imp = report["imports"]
    cls = report["classification"]
    smokes = report["packed_roundtrip_smoke"]

    def _fmt_smoke_table() -> list[str]:
        lines = [
            "| Path | Result | MAE | Cosine | Notes |",
            "|---|---|---:|---:|---|",
        ]
        for s in smokes:
            stats = s.get("stats") or {}
            mae = stats.get("mae", "—")
            cos = stats.get("cosine_similarity", "—")
            note = s.get("error", "") if not s.get("ok") else "ok"
            if isinstance(mae, float):
                mae = f"{mae:.4f}"
            if isinstance(cos, float):
                cos = f"{cos:.4f}"
            lines.append(
                f"| `{s['name']}` | {'OK' if s['ok'] else 'FAIL'} | {mae} | {cos} | {note[:80]} |"
            )
        return lines

    lines = [
        "# Experiment 024: KIVI CUDA/Triton Packed-Path Feasibility",
        "",
        "_Generated by `scripts/research/kivi_cuda_triton_exp024_inspect.py`. "
        "V12 Phase 4 — KIVI CUDA/Triton feasibility only._",
        "",
        "> This is **KIVI CUDA/Triton feasibility only**.",
        "> This is **not** production serving.",
        "> This is **not** a speed or memory benchmark.",
        "> This does **not** claim KIVI packed-path ExactKV acceptance.",
        "> This does **not** replace Experiment 009.",
        "> ExactKV does **not** claim upstream KIVI paper results as ExactKV results.",
        "> No throughput, latency, speedup, runtime, tokens/sec, active GPU memory, "
        "or production readiness claims.",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "Determine whether KIVI CUDA/Triton **packed** quant/dequant can be exercised "
        "for future ExactKV work beyond Experiment 009 offline simulate.",
        "",
        "## 2. Why this follows Experiment 009",
        "",
        "- Exp 009: `kivi_offline_k2_v2` with `simulate=True` only (accept **0.012**).",
        "- Exp 009: `exactkv_failures == 0` but **no** CUDA/Triton kernels.",
        "- Exp 024: investigates `new_pack.py` / `dequant_cuda` / Triton paths.",
        "",
        "## 3. RunPod environment",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| Host | `{env.get('hostname', '—')}` |",
        f"| GPU | `{env.get('gpu_name', '—')}` |",
        f"| torch | `{env.get('torch', '—')}` |",
        f"| transformers | `{env.get('transformers', '—')}` |",
        f"| triton | `{env.get('triton', env.get('triton_error', '—'))}` |",
        f"| Workdir | `{report.get('workdir', '—')}` |",
        f"| KIVI SHA | `{report.get('kivi_sha', '—')}` |",
        "",
        "## 4. KIVI install result",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| Repo | `{report.get('kivi_repo', '—')}` |",
        f"| `utils_quant` import | {imp.get('utils_quant', {}).get('ok', False)} |",
        f"| `quant/` extension | see prep manifest |",
        "",
        "## 5. CUDA/Triton import result",
        "",
        f"| Component | OK |",
        f"|---|---|",
        f"| `dequant_cuda` | {imp.get('dequant_cuda', {}).get('ok', False)} |",
        f"| `quant.new_pack` | {imp.get('new_pack', {}).get('ok', False)} |",
        f"| `kivi_gemv` | {imp.get('kivi_gemv', {}).get('ok', False)} |",
        f"| `LlamaForCausalLM_KIVI` | {imp.get('LlamaForCausalLM_KIVI', {}).get('ok', False)} |",
        "",
        "## 6. Packed-path API findings",
        "",
        "- **utils_quant** (Exp 009): K/V simulate helpers.",
        "- **utils_quant simulate=False**: requires `dequant_cuda` (may be orphaned in setup.py).",
        "- **new_pack.py**: production KV pack (`quant_and_pack_kcache`, etc.).",
        "- **Triton**: `triton_quantize_and_pack_along_last_dim` when available.",
        "",
        "## 7. Qwen2.5 shape compatibility",
        "",
        f"| Check | Value |",
        f"|---|---|",
        f"| Tensor shape | `{report['qwen_shape']['tensor_shape']}` |",
        f"| num_kv_heads | {report['qwen_shape']['num_kv_heads']} |",
        f"| head_dim | {report['qwen_shape']['head_dim']} |",
        f"| Upstream `qwen_kivi.py` | {imp.get('qwen_kivi_model_file', False)} |",
        "",
        "## 8. Packed roundtrip smoke result",
        "",
        *_fmt_smoke_table(),
        "",
        "Reconstruction smoke only — **not** a performance benchmark.",
        "",
        "## 9. Model integration feasibility",
        "",
        "Upstream production path uses `LlamaForCausalLM_KIVI` / custom 9-slot cache — "
        "**no Qwen2.5 KIVI model**. HF `DynamicCache` materialization would need tensor "
        "dequant → `rebuild_cache` bridge or a new Qwen attention port.",
        "",
        "## 10. ExactKV BackendAdapter feasibility",
        "",
        f"**{cls['decision']}** — {cls['rationale']}",
        "",
        f"- Packed BackendAdapter: **{cls['backend_adapter_packed']}**",
        f"- Replaces Exp 009: **{cls['replaces_experiment_009']}**",
        "",
        "## 11. What this proves",
        "",
    ]

    if cls["decision"] != "C_no_go_for_now":
        lines.append("- KIVI packed CUDA/Triton tensor APIs can be exercised on Qwen-shaped GPU tensors.")
    else:
        lines.append("- Exp 009 offline simulate remains the only validated ExactKV KIVI path.")
    lines.extend([
        "- Exp 024 is distinct from Exp 009 (packed vs simulate-only).",
        "",
        "## 12. What this does not prove",
        "",
        "- KIVI packed-path ExactKV acceptance.",
        "- Production serving or upstream paper claims as ExactKV results.",
        "- Qwen end-to-end model integration without new KIVI port.",
        "",
        "## 13. Blockers and risks",
        "",
        "- `dequant_cuda` may be missing from `quant/setup.py`.",
        "- No upstream Qwen KIVI model file.",
        "- Strict KIVI dependency pins vs ExactKV default stack.",
        "- Post-RoPE packed layout ≠ HF `DynamicCache`.",
        "",
        "## 14. Go/no-go recommendation",
        "",
        f"**{cls['decision']}** — {cls['rationale']}",
        "",
        "## 15. Next steps",
        "",
        "- Do **not** implement production packed KIVI adapter without separate approval.",
        "- V12 Phase 5 repair-policy validation may proceed independently.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "bash scripts/research/kivi_cuda_triton_exp024_prep.sh",
        "source /workspace/kivi_exp024/.venv-kivi/bin/activate",
        "export PYTHONPATH=/workspace/kivi_exp024/KIVI:/workspace/kivi_exp024/KIVI/quant",
        "python scripts/research/kivi_cuda_triton_exp024_inspect.py \\",
        "  --kivi-repo /workspace/kivi_exp024/KIVI \\",
        "  --workdir /workspace/kivi_exp024 \\",
        "  --write-markdown docs/EXPERIMENT_024_KIVI_CUDA_TRITON_FEASIBILITY.md",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="KIVI CUDA/Triton Exp 024 inspector")
    parser.add_argument("--kivi-repo", default="/workspace/kivi_exp024/KIVI")
    parser.add_argument("--workdir", default="/workspace/kivi_exp024")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--write-markdown", default="")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    report = run_inspection(kivi_repo=Path(args.kivi_repo), workdir=workdir)

    json_out = Path(args.json_out) if args.json_out else workdir / "exp024_inspect.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"classification: {report['classification']['decision']}")

    if args.write_markdown:
        md_path = Path(args.write_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(generate_markdown(report), encoding="utf-8")
        print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
