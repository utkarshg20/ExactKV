# Experiment 042: SpectralQuant External Probe

_External feasibility probe — not SpectralQuant integration, not a default ExactKV compressor._

> **SpectralQuant is not integrated as a default ExactKV compressor.**
> **External SpectralQuant claims are not ExactKV results.**
> **Tensor smoke results are not generation results.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**

Artifacts (gitignored): `reports/experiment_042_spectralquant_probe.json`

---

## 1. Purpose

Determine whether SpectralQuant can be evaluated by ExactKV as:

1. a real external/compressed KV probe,
2. a tensor-level KV compression smoke, or
3. `restricted_no_go` with exact blockers.

This follows completion of the bounded Shard external-drafter series (Exp 038–041).

## 2. Setup

| Item | Value |
| --- | --- |
| External repo | [Dynamis-Labs/spectralquant](https://github.com/Dynamis-Labs/spectralquant) |
| Local clone | `~/spectralquant` (not vendored; not committed) |
| Script | `scripts/probe_spectralquant.py` |
| Run date | 2026-06-14 |

```bash
git clone https://github.com/Dynamis-Labs/spectralquant.git ~/spectralquant
export SPECTRALQUANT_REPO_PATH=~/spectralquant

# Default: blocked if repo path unset
python3 scripts/probe_spectralquant.py

# Full probe (import + tensor smoke + model feasibility)
python3 scripts/probe_spectralquant.py --try-import --try-tensor-smoke --try-model-probe
```

No `pip install -e` required for probe — `src/` is added to `sys.path` at runtime.

## 3. Repo path

| Check | Result |
| --- | --- |
| `SPECTRALQUANT_REPO_PATH` | `/Users/utkarshgupta/spectralquant` |
| Vendored into ExactKV | **No** |
| `baseline/turboquant_cutile` present | **No** (kernel path optional) |

## 4. Import result

| Field | Result |
| --- | --- |
| `import_success` | **true** |
| `dependency_blocker` | _(none — torch available locally)_ |
| Modules discovered | `calibration`, `spectralquant`, `engine`, `nonuniform_quantization`, `selective_qjl`, `spectral_rotation` |
| Public symbols (sample) | `SpectralQuantEngine`, `EigenspectralCalibrator`, `EngineConfig`, `NonUniformQuantizer`, `KernelSpectralQuantEngine` |

## 5. API classification

| Category | Present |
| --- | --- |
| Model weight quantization only | **No** |
| Tensor quantization utilities | **Yes** — `NonUniformQuantizer`, Lloyd-Max, selective QJL |
| Offline calibration pipeline | **Yes** — `EigenspectralCalibrator` |
| KV-cache tensor compression | **Yes** — `SpectralQuantEngine.compress_keys/values` |
| Generation-time cache path | **No** — no HF `past_key_values` / generate adapter in `src/` |
| Kernel CUDA path | **Optional** — `KernelSpectralQuantEngine` needs `turboquant_cutile` |
| Experiment benchmark scripts | **Yes** — `experiments/` directory |

**Primary integration path for ExactKV:** offline calibration + tensor `BackendAdapter` (mirrors TurboQuant Python adapter pattern).

## 6. Tensor smoke result

| Metric | Value |
| --- | --- |
| Status | **pass** (tensor smoke only) |
| Input shapes | keys/values `(1, 2, 64, 16)` |
| Output shapes preserved | **Yes** (keys and values) |
| Key max abs error | 1.282 |
| Key mean abs error | 0.269 |
| Value max abs error | 1.188 |
| Value mean abs error | 0.277 |

Synthetic K/V tensors with mock calibration data — **not** HF model generation or ExactKV verification.

## 7. Model probe result

| Field | Value |
| --- | --- |
| Attempted | **No** |
| Status | `restricted_no_go` |
| Reason | No generation-time HF cache adapter; tensor API only |

ExactKV external-drafter probe (Shard-style) is **not feasible** without building a new offline adapter that extracts K/V from `past_key_values`, calibrates, compresses, and materializes dequantized tensors for draft forwards.

## 8. Exact blockers

1. **No generation-time cache hook** — SpectralQuant does not expose Shard-like HF generate integration.
2. **Calibration required** — `EigenspectralCalibrator` must run on model forwards before compression.
3. **Tensor API, not past_key_values drop-in** — adapter must bridge HF cache ↔ per-layer tensors.
4. **Optional kernel deps** — `turboquant_cutile` baseline not present in minimal clone.

## 9. What this proves

- SpectralQuant is a **real tensor-level KV compression library** with importable pure-Python engine.
- Synthetic compress/decompress round-trip **works** on K/V tensors (shape preserved).
- ExactKV can probe SpectralQuant **without vendoring** via external clone + `sys.path`.
- Model-level ExactKV crash-test probe is **blocked** until an offline adapter is built.

## 10. What this does not prove

- SpectralQuant preserves exactness under ExactKV verification (no model probe ran).
- SpectralQuant speedup, memory savings, or serving readiness.
- Paper/README headline metrics — those are external results.
- That tensor smoke error bounds transfer to real model K/V at scale.

## 11. Limitations

- Single synthetic tensor smoke — not calibrated on real model statistics.
- No Qwen/Llama model probe attempted (correctly skipped).
- `pip install -e` may fail on some environments (SpectralQuant `pyproject.toml` build backend); probe uses `src/` directly.
- `exactkv_failures` not applicable — no generation probe.

## 12. Next recommendation

**`tensor_smoke_only`** — proceed to optional **Phase 5c / experimental offline BackendAdapter** wrapping `SpectralQuantEngine` after real calibration on a small Qwen panel. Do **not** add to default registry. Do **not** claim ExactKV generation results until a bounded adapter experiment runs with full-KV verification.

Alternative if adapter scope is declined: **`restricted_no_go`** for generation integration; keep SpectralQuant on leaderboard as **FUTURE CANDIDATE** only.

## Related

- [`EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md`](EXPERIMENT_032_ADDENDUM_SHARD_SPECTRALQUANT.md)
- [`EXPERIMENT_041_SHARD_COMBINED_STRESS.md`](EXPERIMENT_041_SHARD_COMBINED_STRESS.md)
