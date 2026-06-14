# Experiment 043: SpectralQuant Real Model K/V Tensor Smoke

_Phase 10F — real KV capture + SpectralQuant compress/decompress on HF prefill tensors._

> **SpectralQuant is not integrated as a default ExactKV compressor.**
> **Tensor smoke results are not ExactKV generation or verification results.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**
> External SpectralQuant paper/README metrics are **not** ExactKV results.

---

## 1. Purpose

Move SpectralQuant feasibility from synthetic K/V (Exp 042) to **real model K/V tensors** captured after HF prefill. Validates that `SpectralQuantEngine` can round-trip actual `past_key_values` / DynamicCache K/V from a small model after minimal eigenspectral calibration.

This is **`real_kv_tensor_smoke`** — not an ExactKV generation probe.

---

## 2. Real model KV capture setup

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` (CPU or GPU) |
| Capture | `prefill_to_full_state` → `extract_kv_tensors` |
| Smoke prompt | Short factual prompt (default: `"What is 2+2? Answer in one word."`) |
| Layers tested | Subset smoke: layer 0, mid, last (all layers used by adapter) |

---

## 3. SpectralQuant API used

| Component | API |
|---|---|
| Calibration | `EigenspectralCalibrator.calibrate(model, tokenizer, prompts, n_samples=2)` |
| Quantizer fit | `SpectralQuantEngine.fit_quantizers(rotated_kv)` from eigenspectrum samples |
| Compression | `SpectralQuantEngine.compress_keys` / `compress_values` per layer |
| Decompression | Per-head key dequant + `SpectralRotation.unrotate`; `decompress_values` for V |

External clone via `SPECTRALQUANT_REPO_PATH` (not vendored).

---

## 4. Tensor smoke result

Run:

```bash
export SPECTRALQUANT_REPO_PATH=~/spectralquant
python3 scripts/research/run_exp043_spectralquant_real_kv_smoke.py
```

Report: `reports/experiment_043_spectralquant_real_kv_smoke.json` (gitignored).

Expected on success:
- `status`: `pass`
- `label`: `real_kv_tensor_smoke`
- Per-layer shape preservation on tested layers
- Key/value max abs error reported (lossy quant — non-zero expected)

---

## 5. Calibration requirement

**Yes — required.** SpectralQuant needs `EigenspectralCalibrator` statistics before `SpectralQuantEngine` can compress real K/V.

Minimal smoke settings (not paper-scale):
- 2 calibration prompts
- `max_tokens_per_layer=256`
- `avg_bits=4.0`, `qjl_projections=32`

---

## 6. What this proves

- SpectralQuant can import and calibrate on a real HF model via external clone.
- Real post-RoPE K/V tensors from Qwen2.5-0.5B prefill can be compressed and decompressed per layer.
- Per-layer shape preservation holds on tested layers after round-trip.

---

## 7. What this does not prove

- ExactKV generation equivalence (no draft/verify loop in this experiment).
- Accepted-prefix metrics, divergence rates, or `exactkv_failures`.
- Speed, active GPU memory savings, or production serving readiness.
- Paper-scale calibration quality or TurboQuant kernel path.

---

## 8. Blockers

| Blocker | Status |
|---|---|
| `SPECTRALQUANT_REPO_PATH` unset | Script exits `blocked` |
| Import / torch / transformers missing | Reported in JSON |
| Calibration API failure | `status=failed` with reason |
| Per-layer compress failure | Layer entry marked `failed` |

---

## 9. Next step

If Exp 043 passes → run Exp 044 factory-only `spectralquant_experimental` adapter smoke (`run_exp044_spectralquant_adapter_smoke.py`).

If Exp 043 fails → remain at `tensor_smoke_only` (Exp 042) on leaderboard; do not claim adapter feasibility.

---

## Run commands

```bash
export SPECTRALQUANT_REPO_PATH=~/spectralquant
python3 scripts/research/run_exp043_spectralquant_real_kv_smoke.py
pytest tests/test_spectralquant_real_kv_smoke.py -q
```
