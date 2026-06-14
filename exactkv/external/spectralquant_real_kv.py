"""SpectralQuant real-model KV tensor helpers (Experiments 043–044).

Factory-only / external — not default registry. Real calibration and real K/V
tensors from HF prefill; no fabricated metrics.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from exactkv.cache.utils import extract_kv_tensors
from exactkv.external.spectralquant_probe import resolve_spectralquant_repo_path

EXPERIMENT_043_ID = "043_spectralquant_real_kv_smoke"
EXPERIMENT_044_ID = "044_spectralquant_adapter_smoke"
EXPERIMENT_045_ID = "045_spectralquant_restricted_panel"

# Exp 045 promotion: restricted backend row requires >=8 prompts and exactkv_failures==0.
PANEL_PROMOTION_MIN_PROMPTS = 8

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

CLAIMS_FORBIDDEN = [
    "SpectralQuant is not integrated as a default ExactKV compressor.",
    "Tensor smoke results are not ExactKV generation or verification results.",
    "Adapter smoke, if present, is experimental and restricted.",
    "No speedup, active memory savings, production serving, or model accuracy improvement claim.",
    "External SpectralQuant paper/README metrics are not ExactKV results.",
]

REQUIRED_043_KEYS = frozenset({
    "experiment_id",
    "status",
    "label",
    "model",
    "prompt",
    "repo_path_present",
    "import_success",
    "calibration",
    "kv_capture",
    "layer_results",
    "summary",
    "limitations",
    "claims_forbidden",
    "recommendation",
})

REQUIRED_044_KEYS = frozenset({
    "experiment_id",
    "status",
    "adapter_name",
    "not_default_registry",
    "model",
    "prompt_panel",
    "exactkv_failures",
    "acceptance_summary",
    "per_prompt",
    "memory_claim_note",
    "limitations",
    "claims_forbidden",
    "recommendation",
})

REQUIRED_045_KEYS = frozenset({
    "experiment_id",
    "status",
    "adapter_name",
    "not_default_registry",
    "model",
    "prompt_count",
    "calibration",
    "panel_composition",
    "exactkv_failures",
    "acceptance_summary",
    "divergence_summary",
    "reconstruction_error_summary",
    "materializing_adapter",
    "memory_claim_note",
    "supports_real_bytes_claim",
    "leaderboard_decision",
    "limitations",
    "claims_forbidden",
    "recommendation",
    "per_prompt",
})


@dataclass(frozen=True)
class CalibrationConfig:
    n_samples: int = 2
    max_tokens_per_layer: int = 256
    avg_bits: float = 4.0
    qjl_projections: int = 32
    lloyd_max_iter: int = 50
    seed: int = 42


def ensure_spectralquant_path(repo_path: Any | None = None) -> Any:
    """Add external SpectralQuant src/ to sys.path; return resolved Path."""
    from pathlib import Path

    path = Path(repo_path) if repo_path is not None else resolve_spectralquant_repo_path()
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            "SPECTRALQUANT_REPO_PATH is not set or not a directory. "
            "Clone https://github.com/Dynamis-Labs/spectralquant.git externally."
        )
    src = path / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"SpectralQuant src/ missing under {path}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return path


def default_calibration_prompts() -> list[str]:
    return [
        "The quick brown fox jumps over the lazy dog.",
        "ExactKV verifies compressed KV drafts against a full-KV reference.",
    ]


def default_smoke_prompt() -> str:
    return "What is 2+2? Answer in one word."


def run_minimal_calibration(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    config: CalibrationConfig | None = None,
) -> Any:
    """Run EigenspectralCalibrator on a tiny prompt set."""
    from spectralquant.calibration import EigenspectralCalibrator
    from spectralquant.spectralquant import EngineConfig, SpectralQuantEngine

    cfg = config or CalibrationConfig()
    calibrator = EigenspectralCalibrator(max_tokens_per_layer=cfg.max_tokens_per_layer)
    calibrator.calibrate(
        model,
        tokenizer,
        prompts,
        n_samples=min(cfg.n_samples, len(prompts)),
    )
    if not calibrator._is_calibrated:
        raise RuntimeError("SpectralQuant calibrator finished without calibration data")
    engine_config = EngineConfig(
        avg_bits=cfg.avg_bits,
        qjl_projections=cfg.qjl_projections,
        lloyd_max_iter=cfg.lloyd_max_iter,
        rotation_seed=cfg.seed,
        lloyd_seed=cfg.seed,
        n_calibration_tokens=cfg.max_tokens_per_layer,
    )
    engine = SpectralQuantEngine(calibrator, engine_config)
    fit_quantizers_from_eigenspectrum(calibrator, engine, seed=cfg.seed)
    return calibrator, engine


def fit_quantizers_from_eigenspectrum(calibrator: Any, engine: Any, *, seed: int = 42) -> None:
    """Fit Lloyd-Max quantizers using eigenspectrum-shaped samples (SpectralQuant pattern)."""
    rotated_kv: dict[tuple[int, int, str], torch.Tensor] = {}
    for hcd in calibrator.iter_heads():
        gen = torch.Generator().manual_seed(seed + 31 * hcd.layer_idx + hcd.head_idx)
        n = max(2 * hcd.head_dim, 256)
        z = torch.randn(n, hcd.head_dim, generator=gen)
        rotated_kv[(hcd.layer_idx, hcd.head_idx, hcd.head_type)] = (
            z * hcd.eigenvalues.sqrt().float()
        )
    engine.fit_quantizers(rotated_kv)


def decompress_keys_layer(
    engine: Any,
    compressed_keys: dict[int, Any],
    layer_idx: int,
) -> torch.Tensor:
    heads: list[torch.Tensor] = []
    for head_idx in sorted(compressed_keys.keys()):
        cv = compressed_keys[head_idx]
        quant = engine._get_quantizer(layer_idx, head_idx, "key")
        k_rot_hat = quant.decompress(cv)
        k_hat = engine._key_rotation.unrotate(k_rot_hat, layer_idx, head_idx)
        heads.append(k_hat)
    return torch.stack(heads, dim=1)


def compress_decompress_layer(
    engine: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    layer_idx: int,
) -> dict[str, Any]:
    """Round-trip one layer's real K/V tensors through SpectralQuantEngine."""
    compressed_keys = engine.compress_keys(keys, layer_idx=layer_idx)
    compressed_values = engine.compress_values(values, layer_idx=layer_idx)
    keys_hat = decompress_keys_layer(engine, compressed_keys, layer_idx)
    values_hat = engine.decompress_values(compressed_values, layer_idx=layer_idx)

    k_err = (keys_hat.float() - keys.float()).abs()
    v_err = (values_hat.float() - values.float()).abs()
    return {
        "layer_idx": layer_idx,
        "input_key_shape": list(keys.shape),
        "input_value_shape": list(values.shape),
        "output_key_shape": list(keys_hat.shape),
        "output_value_shape": list(values_hat.shape),
        "key_shape_preserved": list(keys.shape) == list(keys_hat.shape),
        "value_shape_preserved": list(values.shape) == list(values_hat.shape),
        "key_max_abs_error": float(k_err.max().item()),
        "key_mean_abs_error": float(k_err.mean().item()),
        "value_max_abs_error": float(v_err.max().item()),
        "value_mean_abs_error": float(v_err.mean().item()),
        "n_kv_heads": int(keys.shape[1]),
        "seq_len": int(keys.shape[2]),
        "head_dim": int(keys.shape[3]),
    }


def capture_real_kv_tensors(
    runtime: Any,
    prompt: str,
) -> dict[str, Any]:
    """Prefill and extract per-layer K/V tensors."""
    from exactkv.runtime.prefill import prefill_to_full_state

    state = prefill_to_full_state(runtime, prompt)
    k_tensors, v_tensors, cache_format = extract_kv_tensors(state.past_key_values)
    return {
        "seq_len": state.seq_len,
        "cache_format": cache_format,
        "num_layers": len(k_tensors),
        "dtype": str(k_tensors[0].dtype),
        "device": str(k_tensors[0].device),
        "sample_layer0_key_shape": list(k_tensors[0].shape),
        "k_tensors": k_tensors,
        "v_tensors": v_tensors,
    }


def run_real_kv_tensor_smoke(
    *,
    repo_path: Any | None = None,
    model_name: str = DEFAULT_MODEL,
    prompt: str | None = None,
    calibration_prompts: list[str] | None = None,
    calibration_config: CalibrationConfig | None = None,
    layer_sample: tuple[int, ...] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Phase B/C: real model prefill + SpectralQuant compress/decompress on real K/V."""
    prompt = prompt or default_smoke_prompt()
    calibration_prompts = calibration_prompts or default_calibration_prompts()
    cfg = calibration_config or CalibrationConfig()

    repo_present = resolve_spectralquant_repo_path() is not None or repo_path is not None
    try:
        resolved = ensure_spectralquant_path(repo_path)
    except FileNotFoundError as exc:
        return _blocked_043(str(exc), repo_path_present=repo_present)

    try:
        from exactkv.runtime.model_runtime import ModelRuntime
    except ImportError as exc:
        return _blocked_043(f"ExactKV runtime deps missing: {exc}", repo_path_present=True)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_str = "float16" if dev.type == "cuda" else "float32"

    try:
        runtime = ModelRuntime(model_name, device=dev, dtype=dtype_str)
        model = runtime.model
        tokenizer = runtime.tokenizer
    except Exception as exc:  # noqa: BLE001
        return _blocked_043(f"model load failed: {exc}", repo_path_present=True)

    try:
        calibrator, engine = run_minimal_calibration(
            model, tokenizer, calibration_prompts, config=cfg
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_043(
            reason=f"calibration failed: {type(exc).__name__}: {exc}",
            model=model_name,
            prompt=prompt,
            repo_path=str(resolved),
        )

    try:
        kv = capture_real_kv_tensors(runtime, prompt)
    except Exception as exc:  # noqa: BLE001
        return _failed_043(
            reason=f"KV capture failed: {type(exc).__name__}: {exc}",
            model=model_name,
            prompt=prompt,
            repo_path=str(resolved),
            calibration=_calibration_summary(cfg, calibrator),
        )

    n_layers = kv["num_layers"]
    if layer_sample is None:
        layer_sample = (0, n_layers // 2, n_layers - 1) if n_layers >= 3 else tuple(range(n_layers))
    layer_sample = tuple(i for i in layer_sample if 0 <= i < n_layers)

    layer_results: list[dict[str, Any]] = []
    for layer_idx in layer_sample:
        k = kv["k_tensors"][layer_idx]
        v = kv["v_tensors"][layer_idx]
        try:
            layer_results.append(compress_decompress_layer(engine, k, v, layer_idx))
        except Exception as exc:  # noqa: BLE001
            layer_results.append({
                "layer_idx": layer_idx,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

    passed = all(
        r.get("key_shape_preserved") and r.get("value_shape_preserved")
        for r in layer_results
        if r.get("status") != "failed"
    ) and layer_results and not any(r.get("status") == "failed" for r in layer_results)

    key_max = max((r.get("key_max_abs_error") or 0.0) for r in layer_results)
    val_max = max((r.get("value_max_abs_error") or 0.0) for r in layer_results)

    report = {
        "experiment_id": EXPERIMENT_043_ID,
        "status": "pass" if passed else "failed",
        "label": "real_kv_tensor_smoke",
        "not_exactkv_generation": True,
        "model": model_name,
        "prompt": prompt,
        "repo_path_present": True,
        "import_success": True,
        "spectralquant_repo_path": str(resolved),
        "calibration": _calibration_summary(cfg, calibrator),
        "kv_capture": {
            "seq_len": kv["seq_len"],
            "cache_format": kv["cache_format"],
            "num_layers": kv["num_layers"],
            "dtype": kv["dtype"],
            "device": kv["device"],
            "layer0_key_shape": kv["sample_layer0_key_shape"],
            "layers_tested": list(layer_sample),
        },
        "compression_api": {
            "engine": "SpectralQuantEngine",
            "methods": ["compress_keys", "compress_values", "decompress_values", "key per-head dequant+unrotate"],
            "calibration_api": "EigenspectralCalibrator.calibrate",
        },
        "layer_results": layer_results,
        "summary": {
            "per_layer_compression_works": passed,
            "calibration_required": True,
            "key_max_abs_error": key_max,
            "value_max_abs_error": val_max,
        },
        "limitations": [
            "Real K/V tensor round-trip only — not ExactKV generation.",
            "Minimal calibration (2 prompts) — not paper-scale calibration.",
            "Subset of layers tested for smoke speed; adapter uses all layers.",
            "Materialized dequant K/V for draft — no active memory savings claim.",
        ],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": "real_kv_smoke_only" if passed else "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_043_report(report)
    return report


def _calibration_summary(cfg: CalibrationConfig, calibrator: Any) -> dict[str, Any]:
    n_heads = len(list(calibrator.iter_heads()))
    return {
        "required": True,
        "n_calibration_prompts": cfg.n_samples,
        "max_tokens_per_layer": cfg.max_tokens_per_layer,
        "avg_bits": cfg.avg_bits,
        "qjl_projections": cfg.qjl_projections,
        "n_head_entries": n_heads,
        "api": "EigenspectralCalibrator.calibrate + fit_quantizers_from_eigenspectrum",
    }


def _blocked_043(reason: str, *, repo_path_present: bool) -> dict[str, Any]:
    report = {
        "experiment_id": EXPERIMENT_043_ID,
        "status": "blocked",
        "label": "real_kv_tensor_smoke",
        "not_exactkv_generation": True,
        "model": DEFAULT_MODEL,
        "prompt": "",
        "repo_path_present": repo_path_present,
        "import_success": False,
        "calibration": {"required": True, "ran": False},
        "kv_capture": {},
        "layer_results": [],
        "summary": {"per_layer_compression_works": False, "calibration_required": True},
        "blocked_reason": reason,
        "limitations": ["Set SPECTRALQUANT_REPO_PATH to external clone."],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_043_report(report)
    return report


def _failed_043(
    *,
    reason: str,
    model: str,
    prompt: str,
    repo_path: str,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "experiment_id": EXPERIMENT_043_ID,
        "status": "failed",
        "label": "real_kv_tensor_smoke",
        "not_exactkv_generation": True,
        "model": model,
        "prompt": prompt,
        "repo_path_present": True,
        "import_success": True,
        "spectralquant_repo_path": repo_path,
        "calibration": calibration or {"required": True, "ran": False},
        "kv_capture": {},
        "layer_results": [],
        "summary": {"per_layer_compression_works": False, "calibration_required": True},
        "failure_reason": reason,
        "limitations": [],
        "claims_forbidden": list(CLAIMS_FORBIDDEN),
        "recommendation": "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_043_report(report)
    return report


def validate_043_report(report: dict[str, Any]) -> None:
    missing = REQUIRED_043_KEYS - report.keys()
    if missing:
        raise ValueError(f"043 report missing keys: {sorted(missing)}")


def validate_044_report(report: dict[str, Any]) -> None:
    missing = REQUIRED_044_KEYS - report.keys()
    if missing:
        raise ValueError(f"044 report missing keys: {sorted(missing)}")


def validate_045_report(report: dict[str, Any]) -> None:
    missing = REQUIRED_045_KEYS - report.keys()
    if missing:
        raise ValueError(f"045 report missing keys: {sorted(missing)}")


def default_calibration_prompts_panel() -> list[str]:
    """6-prompt calibration set for restricted panel (bounded runtime)."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "ExactKV verifies compressed KV drafts against a full-KV reference.",
        "Retrieval systems must return verbatim spans when asked to copy text.",
        "Long documents require summarization without inventing new facts.",
        '{"name": "test", "value": 42}',
        "def add(a, b):\n    return a + b\n",
    ]


def load_restricted_panel(*, per_suite: int = 2, max_prompts: int = 12) -> list[dict[str, Any]]:
    """Load 12-prompt restricted panel across six V10 categories."""
    from exactkv.benchmarks.v10_prompts import load_v10_suite

    specs = [
        ("core_v2", "natural_language"),
        ("retrieval_copy", "retrieval_copy"),
        ("long_context", "long_context"),
        ("tool_json", "tool_schema"),
        ("code_structured", "code_structured"),
        ("reasoning_math", "reasoning_math"),
    ]
    out: list[dict[str, Any]] = []
    for suite_name, label in specs:
        try:
            suite = load_v10_suite(suite_name)
        except (FileNotFoundError, ValueError):
            continue
        for row in suite[:per_suite]:
            entry = dict(row)
            entry["panel_category"] = label
            out.append(entry)
        if len(out) >= max_prompts:
            break
    return out[:max_prompts]


def leaderboard_promotion_decision(
    *,
    exactkv_failures: int,
    prompt_count: int,
) -> dict[str, Any]:
    """Whether Exp 045 qualifies for RESTRICTED BACKEND leaderboard row."""
    promote = exactkv_failures == 0 and prompt_count >= PANEL_PROMOTION_MIN_PROMPTS
    return {
        "promote_to_restricted_backend": promote,
        "min_prompts_required": PANEL_PROMOTION_MIN_PROMPTS,
        "exactkv_failures_required": 0,
        "tier_if_promoted": "RESTRICTED BACKEND",
        "tier_if_not": "SMOKE ONLY",
        "reason": (
            f"exactkv_failures=0 and prompt_count={prompt_count}>={PANEL_PROMOTION_MIN_PROMPTS}"
            if promote
            else (
                f"exactkv_failures={exactkv_failures} or "
                f"prompt_count={prompt_count}<{PANEL_PROMOTION_MIN_PROMPTS}"
            )
        ),
    }
