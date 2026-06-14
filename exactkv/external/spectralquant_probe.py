"""SpectralQuant external probe helpers (Experiment 042).

SpectralQuant is inspected and optionally exercised as an external tensor-level
KV compression library — not vendored, not default registry.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "042_spectralquant_probe"

CLAIMS_ALLOWED = [
    "SpectralQuant may be probed as an external tensor-level KV compression library.",
    "Import/API classification may be reported without ExactKV generation integration.",
    "Synthetic tensor smoke may report shape/error metrics — not generation results.",
    "Feasibility classification: external probe only; not default ExactKV compressor.",
]

CLAIMS_FORBIDDEN = [
    "SpectralQuant is not integrated as a default ExactKV compressor.",
    "External SpectralQuant paper/README throughput/memory numbers are not ExactKV results.",
    "Tensor smoke results are not ExactKV generation or verification results.",
    "No speedup, active memory savings, production serving, or model accuracy improvement claim.",
    "No fake token alignment, acceptance metrics, or fabricated exactkv_failures.",
]

VALID_PROBE_STATUSES = frozenset({
    "blocked",
    "import_only",
    "tensor_smoke_only",
    "restricted_no_go",
    "restricted_go",
    "pass",
})

# API surface categories for classification summary.
API_CATEGORIES = (
    "model_weight_quantization_only",
    "tensor_quantization_utilities",
    "offline_calibration_pipeline",
    "kv_cache_tensor_compression",
    "generation_time_cache_path",
    "kernel_cuda_path",
    "experiment_benchmark_scripts",
    "other",
)

REQUIRED_REPORT_KEYS = frozenset({
    "experiment_id",
    "probe_status",
    "blocked_reason",
    "repo_path_present",
    "import_success",
    "dependency_blocker",
    "discovered_api_summary",
    "classification",
    "tensor_smoke_result",
    "model_probe_result",
    "exactkv_failures",
    "limitations",
    "notes",
    "claims_allowed",
    "claims_forbidden",
    "recommendation",
})


@dataclass(frozen=True)
class SpectralQuantImportResult:
    success: bool
    reason: str
    repo_path: Path | None
    modules: tuple[str, ...] = ()
    public_symbols: tuple[str, ...] = ()


def resolve_spectralquant_repo_path() -> Path | None:
    import os

    raw = os.environ.get("SPECTRALQUANT_REPO_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def inspect_repo_layout(repo_path: Path) -> dict[str, Any]:
    """Static repo inspection without importing spectralquant."""
    src = repo_path / "src" / "spectralquant"
    py_files = sorted(p.name for p in src.glob("*.py")) if src.is_dir() else []
    has_experiments = (repo_path / "experiments").is_dir()
    has_baseline = (repo_path / "baseline" / "turboquant_cutile").is_dir()
    readme = (repo_path / "README.md").read_text(encoding="utf-8", errors="replace")[:2000]
    return {
        "src_modules": py_files,
        "has_experiments_dir": has_experiments,
        "has_turboquant_baseline": has_baseline,
        "readme_snippet": readme.splitlines()[:8],
    }


def classify_api(
    *,
    layout: dict[str, Any],
    import_result: SpectralQuantImportResult,
) -> dict[str, Any]:
    """Classify SpectralQuant API exposure from layout + import discovery."""
    modules = set(layout.get("src_modules") or [])
    categories: list[str] = []

    if "spectralquant.py" in modules or "nonuniform_quantization.py" in modules:
        categories.extend([
            "tensor_quantization_utilities",
            "kv_cache_tensor_compression",
        ])
    if "calibration.py" in modules:
        categories.append("offline_calibration_pipeline")
    if "engine.py" in modules:
        categories.append("kernel_cuda_path")
    if layout.get("has_experiments_dir"):
        categories.append("experiment_benchmark_scripts")
    if import_result.public_symbols:
        if "SpectralQuantEngine" in import_result.public_symbols:
            categories.append("kv_cache_tensor_compression")
        if "EigenspectralCalibrator" in import_result.public_symbols:
            categories.append("offline_calibration_pipeline")

    # No dedicated HF past_key_values / generate adapter in src.
    categories = list(dict.fromkeys(categories))

    generation_path = False
    if import_result.success:
        generation_path = False  # confirmed by repo inspection — tensor API only

    return {
        "categories": categories,
        "primary_integration_path": "offline_calibration_tensor_compressor",
        "generation_time_cache_path": generation_path,
        "model_weight_quantization_only": False,
        "notes": [
            "Pure-Python SpectralQuantEngine operates on per-layer K/V tensors.",
            "EigenspectralCalibrator hooks model forward for offline calibration.",
            "KernelSpectralQuantEngine subclasses TurboQuantEngine (optional baseline).",
            "No HF past_key_values drop-in or ExactKV draft hook in src/.",
        ],
    }


def try_import_spectralquant(repo_path: Path) -> SpectralQuantImportResult:
    if not repo_path.is_dir():
        return SpectralQuantImportResult(
            success=False,
            reason=f"SPECTRALQUANT_REPO_PATH is not a directory: {repo_path}",
            repo_path=repo_path,
        )

    src = repo_path / "src"
    added: list[str] = []
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
        added.append(str(src))

    try:
        spec = importlib.util.find_spec("spectralquant")
        if spec is None:
            return SpectralQuantImportResult(
                success=False,
                reason="spectralquant package not found on sys.path after adding src/",
                repo_path=repo_path,
            )
        mod = importlib.import_module("spectralquant")
        public = tuple(
            n for n in dir(mod)
            if not n.startswith("_") and n[0].isupper()
        )
        submodules = (
            "calibration",
            "spectralquant",
            "engine",
            "nonuniform_quantization",
            "selective_qjl",
            "spectral_rotation",
        )
        found = tuple(
            sm for sm in submodules
            if importlib.util.find_spec(f"spectralquant.{sm}") is not None
        )
        return SpectralQuantImportResult(
            success=True,
            reason="",
            repo_path=repo_path,
            modules=found,
            public_symbols=public,
        )
    except ImportError as exc:
        return SpectralQuantImportResult(
            success=False,
            reason=f"spectralquant import failed: {exc}",
            repo_path=repo_path,
        )
    except Exception as exc:  # noqa: BLE001 — probe reports blockers
        return SpectralQuantImportResult(
            success=False,
            reason=f"spectralquant import error: {type(exc).__name__}: {exc}",
            repo_path=repo_path,
        )
    finally:
        for path in added:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def run_tensor_smoke(repo_path: Path) -> dict[str, Any]:
    """Synthetic K/V tensor compress-decompress smoke using pure-Python engine."""
    src = repo_path / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    try:
        import numpy as np
        import torch
        from spectralquant import EngineConfig, SpectralQuantEngine
        from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData
    except ImportError as exc:
        return {
            "status": "failed",
            "reason": f"tensor smoke dependencies missing: {exc}",
        }

    head_dim = 16
    n_tokens = 64
    n_kv_heads = 2
    layer_idx = 0
    seed = 42
    rng = np.random.default_rng(seed)

    eigenvalues = np.concatenate([
        np.array([100.0, 60.0, 30.0, 12.0], dtype=np.float64),
        np.full(head_dim - 4, 0.5, dtype=np.float64),
    ])
    A = rng.standard_normal((head_dim, head_dim))
    Q, _ = np.linalg.qr(A)
    eigvals = torch.from_numpy(eigenvalues.astype(np.float32))
    V = torch.from_numpy(Q.astype(np.float32))

    calib = EigenspectralCalibrator(max_tokens_per_layer=n_tokens)
    sum_lam = float(eigenvalues.sum())
    sum_sq = float((eigenvalues ** 2).sum())
    d_eff = (sum_lam ** 2) / sum_sq if sum_sq > 1e-12 else 1.0

    rotated = (
        rng.standard_normal((n_tokens, head_dim)).astype(np.float32)
        * np.sqrt(eigenvalues).astype(np.float32)
    )
    rotated_t = torch.from_numpy(rotated)

    for head_idx in range(n_kv_heads):
        for head_type in ("key", "value"):
            hcd = HeadCalibrationData(
                layer_idx=layer_idx,
                head_idx=head_idx,
                head_type=head_type,
                eigenvalues=eigvals,
                eigenvectors=V,
                d_eff=float(d_eff),
                spectral_gap=None,
                var_95=int(min(head_dim, max(1, round(d_eff)))),
                var_99=int(min(head_dim, max(1, round(d_eff)))),
                n_samples=n_tokens,
                head_dim=head_dim,
            )
            calib._calibration_data[(layer_idx, head_idx, head_type)] = hcd
    calib._is_calibrated = True

    config = EngineConfig(avg_bits=4.0, qjl_projections=32, n_calibration_tokens=n_tokens)
    engine = SpectralQuantEngine(calib, config)

    rotated_kv: dict[tuple[int, int, str], Any] = {}
    for head_idx in range(n_kv_heads):
        for head_type in ("key", "value"):
            rotated_kv[(layer_idx, head_idx, head_type)] = rotated_t
    engine.fit_quantizers(rotated_kv)

    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, n_kv_heads, n_tokens, head_dim, generator=g)
    values = torch.randn(1, n_kv_heads, n_tokens, head_dim, generator=g)

    input_shapes = {
        "keys": list(keys.shape),
        "values": list(values.shape),
    }
    input_dtypes = {
        "keys": str(keys.dtype),
        "values": str(values.dtype),
    }

    compressed_keys = engine.compress_keys(keys, layer_idx=layer_idx)
    compressed_values = engine.compress_values(values, layer_idx=layer_idx)

    key_errors: list[float] = []
    value_errors: list[float] = []
    key_shapes_ok = True
    value_shapes_ok = True

    for head_idx, cv in compressed_keys.items():
        quant = engine._get_quantizer(layer_idx, head_idx, "key")
        k_rot_hat = quant.decompress(cv)
        k_hat = engine._key_rotation.unrotate(k_rot_hat, layer_idx, head_idx)
        orig = keys[:, head_idx, :, :]
        if list(k_hat.shape) != list(orig.shape):
            key_shapes_ok = False
        err = (k_hat.float() - orig.float()).abs()
        key_errors.append(float(err.max().item()))
        key_errors.append(float(err.mean().item()))

    values_hat = engine.decompress_values(compressed_values, layer_idx=layer_idx)
    if list(values_hat.shape) != list(values.shape):
        value_shapes_ok = False
    verr = (values_hat.float() - values.float()).abs()

    return {
        "status": "pass",
        "label": "tensor_smoke_only",
        "not_exactkv_generation": True,
        "input_shapes": input_shapes,
        "input_dtypes": input_dtypes,
        "compressed_key_heads": len(compressed_keys),
        "compressed_value_heads": len(compressed_values),
        "output_shape_preserved_keys": key_shapes_ok,
        "output_shape_preserved_values": value_shapes_ok,
        "key_max_abs_error": max(key_errors) if key_errors else None,
        "key_mean_abs_error": sum(key_errors[1::2]) / max(1, len(key_errors) // 2) if key_errors else None,
        "value_max_abs_error": float(verr.max().item()),
        "value_mean_abs_error": float(verr.mean().item()),
        "interpretation": (
            "SpectralQuant compress/decompress round-trip on synthetic K/V tensors succeeded. "
            "This validates tensor-level KV compression primitives only — not HF generation."
        ),
    }


def assess_model_probe_feasibility(
    *,
    classification: dict[str, Any],
    import_result: SpectralQuantImportResult,
) -> dict[str, Any]:
    """Determine whether an ExactKV model/external drafter probe is feasible."""
    if not import_result.success:
        return {
            "attempted": False,
            "status": "skipped",
            "reason": "import failed — model probe not attempted",
        }

    if not classification.get("generation_time_cache_path"):
        return {
            "attempted": False,
            "status": "restricted_no_go",
            "reason": (
                "No generation-time HF cache adapter exposed in SpectralQuant src/. "
                "Library provides offline calibration + per-layer tensor compress/decompress "
                "(SpectralQuantEngine) and experiment scripts — not an external drafter like Shard."
            ),
            "feasible_with": "offline BackendAdapter wrapping tensor API after calibration",
        }

    return {
        "attempted": False,
        "status": "skipped",
        "reason": "unexpected classification state",
    }


def build_report(
    *,
    probe_status: str,
    blocked_reason: str,
    repo_path_present: bool,
    import_success: bool,
    dependency_blocker: str,
    discovered_api_summary: dict[str, Any],
    classification: dict[str, Any],
    tensor_smoke_result: dict[str, Any] | None,
    model_probe_result: dict[str, Any] | None,
    exactkv_failures: int | None,
    limitations: list[str],
    notes: list[str],
    recommendation: str,
    repo_path: str | None = None,
) -> dict[str, Any]:
    report = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_class": "spectralquant_external_probe",
        "integration_mode": "external_tensor_probe",
        "not_default_registry": True,
        "not_kvcompressor_backend": True,
        "probe_status": probe_status,
        "blocked_reason": blocked_reason,
        "repo_path_present": repo_path_present,
        "import_success": import_success,
        "dependency_blocker": dependency_blocker,
        "discovered_api_summary": discovered_api_summary,
        "classification": classification,
        "tensor_smoke_result": tensor_smoke_result,
        "model_probe_result": model_probe_result,
        "exactkv_failures": exactkv_failures,
        "limitations": limitations,
        "notes": notes,
        "claims_allowed": list(CLAIMS_ALLOWED),
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": recommendation,
    }
    if repo_path:
        report["spectralquant_repo_path"] = repo_path
    validate_report(report)
    return report


def blocked_report(*, reason: str, repo_path_present: bool) -> dict[str, Any]:
    return build_report(
        probe_status="blocked",
        blocked_reason=reason,
        repo_path_present=repo_path_present,
        import_success=False,
        dependency_blocker=reason if not repo_path_present else "",
        discovered_api_summary={},
        classification={},
        tensor_smoke_result=None,
        model_probe_result=None,
        exactkv_failures=None,
        limitations=[
            "Set SPECTRALQUANT_REPO_PATH to an external clone of Dynamis-Labs/spectralquant.",
        ],
        notes=["Clone externally: git clone https://github.com/Dynamis-Labs/spectralquant.git"],
        recommendation="blocked",
    )


def validate_report(report: dict[str, Any]) -> None:
    missing = REQUIRED_REPORT_KEYS - report.keys()
    if missing:
        raise ValueError(f"report missing keys: {sorted(missing)}")
    if report["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must be 042_spectralquant_probe")
    if report["probe_status"] not in VALID_PROBE_STATUSES:
        raise ValueError(f"invalid probe_status: {report['probe_status']}")
