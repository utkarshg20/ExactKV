"""KV runtime probe layer (Phase D — systems credibility instrumentation).

Read-only forward hooks + simulated compression on copied KV tensors.
Diagnostic only — no weight modification, no L4 commit, no serving integration.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from exactkv.cache.utils import extract_kv_tensors, kv_seq_len, kv_total_bytes, rebuild_cache
from exactkv.metrics.exactness import first_divergence_idx
from exactkv.runtime.model_runtime import ModelRuntime
from exactkv.runtime.prefill import prefill_to_full_state
from exactkv.safety.l4_runtime_coupling_stress_panel import STRESS_PANEL_PROMPTS

PHASE_D_ID = "phaseD_runtime_probe"
DEFAULT_RUNTIME_PROBE_REPORT = Path("reports/phaseD_runtime_probe.json")
DEFAULT_MEMORY_PROFILE_REPORT = Path("reports/phaseD_memory_profile.json")
DEFAULT_LAYER_DRIFT_REPORT = Path("reports/phaseD_layer_drift.json")
DEFAULT_VISUALS_DIR = Path("reports/visuals/phaseD")

PROBE_MODES: tuple[str, ...] = ("noop", "int8_sim", "int4_sim", "kv_dropout_sim")
DEFAULT_PROBE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.1-8B",
)
DEFAULT_PROBE_PROMPT_COUNT = 4
DEFAULT_MAX_NEW_TOKENS = 4
DEFAULT_KV_DROPOUT_RATE = 0.05

REPRODUCIBLE_CLI = "python scripts/run_phase_d_runtime_probe.py --deterministic-mode"


@dataclass
class LayerCapture:
    layer_index: int
    hidden_norm: float
    hidden_variance: float
    attention_entropy_proxy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeCellResult:
    model_name: str
    prompt_id: str
    prompt_text: str
    compression_mode: str
    max_new_tokens: int
    baseline_token_ids: list[int]
    probed_token_ids: list[int]
    token_exact_match: bool
    first_divergence_index: int | None
    memory_proxy: dict[str, Any]
    divergence_metrics: dict[str, Any]
    stability_metrics: dict[str, Any]
    layer_captures: list[LayerCapture]
    kv_access_mode: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layer_captures"] = [lc.to_dict() for lc in self.layer_captures]
        return d


@dataclass(frozen=True)
class PhaseDValidationResult:
    valid: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReadOnlyProbeHooks:
    """Non-invasive forward hooks on decoder layers (read-only)."""

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self.layer_captures: list[LayerCapture] = []
        self._attention_entropy_proxies: dict[int, float] = {}

    def _layer_hook(self, layer_index: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return
            last = hidden[:, -1, :].detach().float()
            self.layer_captures.append(
                LayerCapture(
                    layer_index=layer_index,
                    hidden_norm=float(last.norm().item()),
                    hidden_variance=float(last.var().item()),
                    attention_entropy_proxy=self._attention_entropy_proxies.get(layer_index),
                ),
            )

        return hook

    def install(self, model: torch.nn.Module) -> str:
        """Install hooks; return kv_access_mode."""
        layers = _find_decoder_layers(model)
        if not layers:
            return "proxy_only"
        for idx, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._layer_hook(idx))
            self._handles.append(handle)
        return "layer_hidden_states"

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.layer_captures.clear()
        self._attention_entropy_proxies.clear()


def _find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


def _tensor_entropy_proxy(t: torch.Tensor, bins: int = 32) -> float:
    flat = t.detach().float().abs().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    hist = torch.histc(flat, bins=bins, min=0.0, max=float(flat.max().item()) + 1e-8)
    probs = hist / hist.sum().clamp(min=1e-8)
    entropy = -(probs * (probs + 1e-12).log()).sum()
    return float(entropy.item())


def simulate_compression_on_kv(
    k_tensors: list[torch.Tensor],
    v_tensors: list[torch.Tensor],
    mode: str,
    *,
    seed: int = 0,
    dropout_rate: float = DEFAULT_KV_DROPOUT_RATE,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Apply compression simulation on cloned KV tensors only."""
    k_out: list[torch.Tensor] = []
    v_out: list[torch.Tensor] = []
    device = k_tensors[0].device if k_tensors else torch.device("cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    for k, v in zip(k_tensors, v_tensors):
        kc = k.clone()
        vc = v.clone()
        if mode == "noop":
            k_out.append(kc)
            v_out.append(vc)
            continue
        if mode == "int8_sim":
            k_out.append(_quantize_dequantize_symmetric(kc, qmax=127.0))
            v_out.append(_quantize_dequantize_symmetric(vc, qmax=127.0))
        elif mode == "int4_sim":
            k_out.append(_quantize_dequantize_symmetric(kc, qmax=7.0, qmin=-8.0))
            v_out.append(_quantize_dequantize_symmetric(vc, qmax=7.0, qmin=-8.0))
        elif mode == "kv_dropout_sim":
            mask = torch.rand(vc.shape, generator=gen, device=vc.device) > dropout_rate
            v_out.append(vc * mask.to(vc.dtype))
            k_out.append(kc)
        else:
            msg = f"unknown compression mode: {mode}"
            raise ValueError(msg)
    return k_out, v_out


def _quantize_dequantize_symmetric(
    t: torch.Tensor,
    *,
    qmax: float,
    qmin: float = -128.0,
) -> torch.Tensor:
    scale = t.abs().max().clamp(min=1e-8) / qmax
    q = (t / scale).round().clamp(qmin, qmax)
    return q * scale


def _layerwise_memory_scores(
    k_tensors: list[torch.Tensor],
    v_tensors: list[torch.Tensor],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i, (k, v) in enumerate(zip(k_tensors, v_tensors)):
        k_bytes = float(k.nelement() * k.element_size())
        v_bytes = float(v.nelement() * v.element_size())
        rows.append(
            {
                "layer": float(i),
                "kv_bytes": k_bytes + v_bytes,
                "kv_activation_norm": float(k.norm().item() + v.norm().item()),
                "kv_entropy_proxy": (_tensor_entropy_proxy(k) + _tensor_entropy_proxy(v)) / 2.0,
            },
        )
    return rows


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    if a_flat.shape != b_flat.shape:
        return 1.0
    sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    return float(1.0 - sim)


@torch.no_grad()
def _greedy_decode_from_state(
    runtime: ModelRuntime,
    *,
    past_key_values: Any,
    first_token_id: int,
    max_new_tokens: int,
) -> list[int]:
    generated: list[int] = []
    next_id = first_token_id
    cache = past_key_values
    for _ in range(max_new_tokens):
        generated.append(next_id)
        if next_id == runtime.eos_token_id:
            break
        step_in = torch.tensor([[next_id]], dtype=torch.long, device=runtime.device)
        out = runtime.forward(step_in, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
    return generated


@torch.no_grad()
def run_probe_cell(
    runtime: ModelRuntime,
    *,
    prompt_id: str,
    prompt_text: str,
    compression_mode: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
) -> ProbeCellResult:
    """Run one probe cell: baseline vs compression-simulated KV path."""
    hooks = ReadOnlyProbeHooks()
    kv_access = hooks.install(runtime.model)

    # Baseline prefill + decode
    hooks.clear()
    baseline_state = prefill_to_full_state(runtime, prompt_text)
    baseline_first = int(baseline_state.metadata["next_token_id"])
    hooks.clear()
    baseline_logits_out = runtime.forward(
        torch.tensor([[baseline_first]], dtype=torch.long, device=runtime.device),
        past_key_values=copy.deepcopy(baseline_state.past_key_values),
        use_cache=True,
    )
    baseline_logits = baseline_logits_out.logits[:, -1, :]
    baseline_layer_snapshot = list(hooks.layer_captures)
    hooks.clear()

    baseline_ids = _greedy_decode_from_state(
        runtime,
        past_key_values=copy.deepcopy(baseline_state.past_key_values),
        first_token_id=baseline_first,
        max_new_tokens=max_new_tokens,
    )

    # Compressed path on copied KV
    k_tensors, v_tensors, fmt = extract_kv_tensors(baseline_state.past_key_values)
    k_cloned = [t.clone() for t in k_tensors]
    v_cloned = [t.clone() for t in v_tensors]
    k_sim, v_sim = simulate_compression_on_kv(
        k_cloned,
        v_cloned,
        compression_mode,
        seed=seed,
    )
    seq_len = kv_seq_len(baseline_state.past_key_values)
    compressed_cache = rebuild_cache(k_sim, v_sim, fmt, seq_len)

    hooks.clear()
    comp_out = runtime.forward(
        torch.tensor([[baseline_first]], dtype=torch.long, device=runtime.device),
        past_key_values=compressed_cache,
        use_cache=True,
    )
    compressed_layer_snapshot = list(hooks.layer_captures)
    baseline_hidden = baseline_layer_snapshot[-1].hidden_norm if baseline_layer_snapshot else 0.0
    compressed_hidden = compressed_layer_snapshot[-1].hidden_norm if compressed_layer_snapshot else 0.0

    probed_ids = _greedy_decode_from_state(
        runtime,
        past_key_values=compressed_cache,
        first_token_id=baseline_first,
        max_new_tokens=max_new_tokens,
    )

    hooks.remove()

    layer_drift: list[LayerCapture] = []
    for bl, cl in zip(baseline_layer_snapshot, compressed_layer_snapshot):
        drift_score = abs(bl.hidden_norm - cl.hidden_norm) / max(bl.hidden_norm, 1e-8)
        layer_drift.append(
            LayerCapture(
                layer_index=bl.layer_index,
                hidden_norm=drift_score,
                hidden_variance=cl.hidden_variance,
                attention_entropy_proxy=cl.attention_entropy_proxy,
            ),
        )

    layer_memory = _layerwise_memory_scores(k_cloned, v_cloned)
    first_div = first_divergence_idx(
        torch.tensor(baseline_ids),
        torch.tensor(probed_ids),
    )

    return ProbeCellResult(
        model_name=runtime.model_name,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        compression_mode=compression_mode,
        max_new_tokens=max_new_tokens,
        baseline_token_ids=baseline_ids,
        probed_token_ids=probed_ids,
        token_exact_match=baseline_ids == probed_ids,
        first_divergence_index=first_div,
        memory_proxy={
            "kv_total_bytes": kv_total_bytes(baseline_state.past_key_values),
            "kv_activation_norm": sum(r["kv_activation_norm"] for r in layer_memory),
            "kv_entropy_proxy": sum(r["kv_entropy_proxy"] for r in layer_memory) / max(len(layer_memory), 1),
            "layerwise_memory_score": layer_memory,
        },
        divergence_metrics={
            "cosine_hidden_drift_proxy": abs(baseline_hidden - compressed_hidden)
            / max(baseline_hidden, 1e-8),
            "logits_drift_proxy": _cosine_distance(comp_out.logits[:, -1, :], baseline_logits),
            "first_divergence_index": first_div,
            "token_level_divergence": first_div is not None,
        },
        stability_metrics={
            "per_layer_instability_score": [lc.hidden_norm for lc in layer_drift],
            "mean_layer_instability": (
                sum(lc.hidden_norm for lc in layer_drift) / max(len(layer_drift), 1)
            ),
            "compression_sensitivity": float(first_div is not None),
        },
        layer_captures=layer_drift,
        kv_access_mode=kv_access,
        seed=seed,
    )


def default_probe_prompts(count: int = DEFAULT_PROBE_PROMPT_COUNT) -> list[tuple[str, str]]:
    return list(STRESS_PANEL_PROMPTS[:count])


def build_deterministic_probe_cell(
    *,
    model_name: str,
    prompt_id: str,
    prompt_text: str,
    compression_mode: str,
    max_new_tokens: int,
    seed: int,
) -> ProbeCellResult:
    """Hash-seeded synthetic probe cell for CI (no GPU)."""
    cell_seed = abs(hash((model_name, prompt_id, compression_mode, seed))) % 10_000
    diverges = compression_mode in ("int4_sim", "kv_dropout_sim") and cell_seed % 3 == 0
    first_div = (cell_seed % max_new_tokens) if diverges else None
    baseline = [1000 + cell_seed + i for i in range(max_new_tokens)]
    probed = list(baseline)
    if diverges and first_div is not None and first_div < len(probed):
        probed[first_div] = probed[first_div] + 17

    n_layers = 24 if "0.5B" in model_name else 32
    layer_drift = [
        LayerCapture(
            layer_index=i,
            hidden_norm=0.01 * (i + 1) * (1.5 if compression_mode != "noop" else 1.0),
            hidden_variance=0.001 * (i + 1),
        )
        for i in range(min(n_layers, 8))
    ]

    return ProbeCellResult(
        model_name=model_name,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        compression_mode=compression_mode,
        max_new_tokens=max_new_tokens,
        baseline_token_ids=baseline,
        probed_token_ids=probed,
        token_exact_match=baseline == probed,
        first_divergence_index=first_div,
        memory_proxy={
            "kv_total_bytes": 32000 + cell_seed,
            "kv_activation_norm": 128.0 + cell_seed / 100.0,
            "kv_entropy_proxy": 2.5 + cell_seed / 1000.0,
            "layerwise_memory_score": [
                {
                    "layer": float(i),
                    "kv_bytes": 1000.0 + i * 50,
                    "kv_activation_norm": 10.0 + i,
                    "kv_entropy_proxy": 2.0 + i * 0.1,
                }
                for i in range(min(n_layers, 8))
            ],
        },
        divergence_metrics={
            "cosine_hidden_drift_proxy": 0.0 if compression_mode == "noop" else 0.02 * cell_seed / 1000,
            "logits_drift_proxy": 0.0 if compression_mode == "noop" else 0.05 * cell_seed / 1000,
            "first_divergence_index": first_div,
            "token_level_divergence": diverges,
        },
        stability_metrics={
            "per_layer_instability_score": [lc.hidden_norm for lc in layer_drift],
            "mean_layer_instability": sum(lc.hidden_norm for lc in layer_drift) / len(layer_drift),
            "compression_sensitivity": float(diverges),
        },
        layer_captures=layer_drift,
        kv_access_mode="deterministic_proxy",
        seed=seed,
    )


def run_phase_d_runtime_probe(
    *,
    models: Sequence[str] | None = None,
    prompts: Sequence[tuple[str, str]] | None = None,
    modes: Sequence[str] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float32",
    deterministic_mode: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Execute full Phase D probe grid."""
    model_list = list(models or DEFAULT_PROBE_MODELS)
    prompt_list = list(prompts or default_probe_prompts())
    mode_list = list(modes or PROBE_MODES)

    if deterministic_mode:
        models_evaluated = list(model_list)
        models_blocked: dict[str, str] = {}
    else:
        from exactkv.benchmarks.phase_a_scale_benchmark import detect_model_availability

        models_evaluated, models_blocked = detect_model_availability(
            model_list,
            local_files_only=local_files_only,
        )

    cells: list[ProbeCellResult] = []
    for model_name in models_evaluated:
        runtime: ModelRuntime | None = None
        if not deterministic_mode:
            runtime = ModelRuntime(model_name, device=device, dtype=dtype)
        for prompt_id, prompt_text in prompt_list:
            for mode in mode_list:
                if deterministic_mode:
                    cell = build_deterministic_probe_cell(
                        model_name=model_name,
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        compression_mode=mode,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                    )
                else:
                    assert runtime is not None
                    cell = run_probe_cell(
                        runtime,
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        compression_mode=mode,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                    )
                cells.append(cell)

    cell_dicts = [c.to_dict() for c in cells]
    return {
        "phase_id": PHASE_D_ID,
        "status": "probe_complete",
        "deterministic_mode": deterministic_mode,
        "models_evaluated": models_evaluated,
        "models_blocked": models_blocked,
        "modes": mode_list,
        "prompt_count": len(prompt_list),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "total_cells": len(cells),
        "cells": cell_dicts,
        "exactkv_generator_modified": False,
        "runtime_commit_authorized": False,
        "l4_activation": False,
        "trace_only": True,
        "instrumentation_only": True,
        "reproducible_cli_command": REPRODUCIBLE_CLI,
    }


def build_memory_profile_report(runtime_probe: Mapping[str, Any]) -> dict[str, Any]:
    """Extract memory proxy profile from runtime probe cells."""
    by_model_mode: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    layer_agg: dict[str, list[float]] = defaultdict(list)

    for cell in runtime_probe.get("cells") or []:
        mp = cell.get("memory_proxy") or {}
        key = f"{cell.get('model_name')}|{cell.get('compression_mode')}"
        by_model_mode[key]["kv_total_bytes"].append(float(mp.get("kv_total_bytes") or 0))
        by_model_mode[key]["kv_activation_norm"].append(float(mp.get("kv_activation_norm") or 0))
        by_model_mode[key]["kv_entropy_proxy"].append(float(mp.get("kv_entropy_proxy") or 0))
        for layer_row in mp.get("layerwise_memory_score") or []:
            layer_agg[f"layer_{int(layer_row.get('layer', 0))}"].append(
                float(layer_row.get("kv_activation_norm") or 0),
            )

    summary: dict[str, Any] = {}
    for key, metrics in sorted(by_model_mode.items()):
        summary[key] = {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}

    return {
        "phase_id": PHASE_D_ID,
        "report_type": "memory_profile",
        "source": str(DEFAULT_RUNTIME_PROBE_REPORT),
        "aggregates": summary,
        "layer_mean_activation_norm": {
            k: sum(v) / max(len(v), 1) for k, v in sorted(layer_agg.items())
        },
        "note": "Memory proxies from KV tensor norms/entropy — not measured GPU peak memory.",
    }


def build_layer_drift_report(runtime_probe: Mapping[str, Any]) -> dict[str, Any]:
    """Extract per-layer drift from runtime probe cells."""
    heatmap: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sensitivity: dict[str, list[float]] = defaultdict(list)

    for cell in runtime_probe.get("cells") or []:
        model = str(cell.get("model_name") or "")
        mode = str(cell.get("compression_mode") or "")
        for i, score in enumerate(
            (cell.get("stability_metrics") or {}).get("per_layer_instability_score") or [],
        ):
            heatmap[model][f"layer_{i}"].append(float(score))
        sensitivity[mode].append(
            float((cell.get("stability_metrics") or {}).get("compression_sensitivity") or 0),
        )

    heatmap_mean = {
        model: {layer: sum(v) / max(len(v), 1) for layer, v in layers.items()}
        for model, layers in heatmap.items()
    }
    sensitivity_mean = {m: sum(v) / max(len(v), 1) for m, v in sensitivity.items()}

    return {
        "phase_id": PHASE_D_ID,
        "report_type": "layer_drift",
        "source": str(DEFAULT_RUNTIME_PROBE_REPORT),
        "layer_drift_heatmap": heatmap_mean,
        "compression_sensitivity_by_mode": sensitivity_mean,
        "divergence_by_mode": _aggregate_divergence(runtime_probe),
    }


def _aggregate_divergence(runtime_probe: Mapping[str, Any]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for cell in runtime_probe.get("cells") or []:
        div = (cell.get("divergence_metrics") or {}).get("logits_drift_proxy")
        if div is not None:
            buckets[str(cell.get("compression_mode") or "")].append(float(div))
    return {k: sum(v) / max(len(v), 1) for k, v in buckets.items()}


def render_phase_d_visuals(
    runtime_probe: Mapping[str, Any],
    memory_profile: Mapping[str, Any],
    layer_drift: Mapping[str, Any],
    *,
    output_dir: Path = DEFAULT_VISUALS_DIR,
) -> dict[str, Any]:
    """Render optional matplotlib plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"output_dir": str(output_dir)}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # Layer drift heatmap
        heat = layer_drift.get("layer_drift_heatmap") or {}
        if heat:
            models = sorted(heat.keys())
            layers = sorted({l for m in heat.values() for l in m})
            mat = np.zeros((len(models), len(layers)))
            for i, m in enumerate(models):
                for j, l in enumerate(layers):
                    mat[i, j] = heat[m].get(l, 0.0)
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([m.split("/")[-1][:20] for m in models])
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, rotation=45, ha="right")
            ax.set_title("Layer-wise Drift Heatmap (Phase D)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            p = output_dir / "layer_drift_heatmap.png"
            fig.savefig(p)
            plt.close(fig)
            result["layer_drift_heatmap"] = str(p)

        # Memory proxy bar chart
        agg = memory_profile.get("aggregates") or {}
        if agg:
            labels = sorted(agg.keys())
            norms = [agg[k].get("kv_activation_norm", 0) for k in labels]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(labels)), norms, color="#1f77b4")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([l[:30] for l in labels], rotation=45, ha="right")
            ax.set_ylabel("kv_activation_norm (proxy)")
            ax.set_title("Memory Proxy Comparison")
            fig.tight_layout()
            p = output_dir / "memory_proxy_bars.png"
            fig.savefig(p)
            plt.close(fig)
            result["memory_proxy_bars"] = str(p)

        # Divergence vs compression curve
        div = layer_drift.get("divergence_by_mode") or {}
        sens = layer_drift.get("compression_sensitivity_by_mode") or {}
        if div:
            modes = sorted(div.keys())
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                modes,
                [div.get(m, 0) for m in modes],
                marker="o",
                label="logits drift proxy",
            )
            ax.plot(
                modes,
                [sens.get(m, 0) for m in modes],
                marker="s",
                label="compression sensitivity",
            )
            ax.set_xlabel("compression mode")
            ax.set_ylabel("score")
            ax.set_title("Divergence vs Compression Level")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = output_dir / "divergence_compression_curve.png"
            fig.savefig(p)
            plt.close(fig)
            result["divergence_compression_curve"] = str(p)

    except Exception as exc:  # noqa: BLE001
        result["plot_error"] = str(exc)

    return result


def validate_phase_d_report(report: Mapping[str, Any]) -> PhaseDValidationResult:
    errors: list[str] = []
    if report.get("phase_id") != PHASE_D_ID:
        errors.append("phase_id mismatch")
    if not report.get("cells"):
        errors.append("missing cells")
    if report.get("exactkv_generator_modified") is not False:
        errors.append("exactkv_generator_modified must be false")
    if report.get("runtime_commit_authorized") is not False:
        errors.append("runtime_commit_authorized must be false")
    return PhaseDValidationResult(valid=len(errors) == 0, errors=tuple(errors))


def write_phase_d_reports(
    runtime_probe: Mapping[str, Any],
    *,
    runtime_path: Path = DEFAULT_RUNTIME_PROBE_REPORT,
    memory_path: Path = DEFAULT_MEMORY_PROFILE_REPORT,
    layer_path: Path = DEFAULT_LAYER_DRIFT_REPORT,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
) -> dict[str, str]:
    memory = build_memory_profile_report(runtime_probe)
    layer = build_layer_drift_report(runtime_probe)
    visuals = render_phase_d_visuals(runtime_probe, memory, layer, output_dir=visuals_dir)

    paths: dict[str, str] = {}
    for path, data in (
        (runtime_path, runtime_probe),
        (memory_path, memory),
        (layer_path, layer),
    ):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2) + "\n")
        paths[path.stem] = str(p)

    vis_json = Path(visuals_dir) / "phaseD_visual_manifest.json"
    vis_json.write_text(json.dumps(visuals, indent=2) + "\n")
    paths["visual_manifest"] = str(vis_json)
    return paths
