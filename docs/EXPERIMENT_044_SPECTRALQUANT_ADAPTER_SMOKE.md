# Experiment 044: SpectralQuant Experimental Adapter Smoke

_Phase 10F — factory-only `spectralquant_experimental` BackendAdapter ExactKV smoke._

> **SpectralQuant is not integrated as a default ExactKV compressor.**
> **Adapter smoke is experimental and restricted — not a production backend.**
> **No speedup, active memory savings, production serving, or model accuracy improvement claim is made.**

---

## 1. Purpose

After Exp 043 real-KV tensor smoke passes, exercise a **factory-only** ExactKV adapter that:
1. Calibrates SpectralQuant minimally on load.
2. Compresses cloned K/V tensors on `compress()`.
3. Materialises dequant full K/V for the draft path.
4. Leaves full-KV verification unchanged.

---

## 2. Adapter design

| Field | Value |
|---|---|
| Class | `SpectralQuantExperimentalAdapter` |
| Module | `exactkv/external/spectralquant_adapter.py` |
| Factory | `create_spectralquant_experimental_adapter(runtime)` |
| Registry | **Not** registered — explicit factory import only |
| Base | `BackendAdapter` tensor path (`_backend_compress` / `_backend_materialize`) |
| Lossy draft | Overrides `_get_next_token_id` (materialize + forward on prefix) |

---

## 3. Why factory-only

- SpectralQuant requires external clone + calibration — not suitable for default `get_compressor()`.
- Experimental / restricted tier — same pattern as TurboQuant Python, KIVI offline, SnapKV experimental.
- Prevents accidental promotion to built-in compressor without explicit experiment wiring.

---

## 4. ExactKV smoke result

Run (requires Exp 043 path + deps):

```bash
export SPECTRALQUANT_REPO_PATH=~/spectralquant
python3 scripts/research/run_exp044_spectralquant_adapter_smoke.py
```

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Prompts | 2–4 (V10 first-prompt panel or built-in smoke pair) |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| Verification | Default sequential full-KV (unchanged) |

Report: `reports/experiment_044_spectralquant_adapter_smoke.json` (gitignored).

---

## 5. exactkv_failures

Reported only when adapter smoke **actually runs**. If repo missing or adapter init fails, `exactkv_failures` is `null` and `status=blocked`.

When run: sum of per-prompt failures (0 = all outputs match full greedy).

---

## 6. Accepted-prefix result

Per-prompt `mean_acceptance` and `accepted_prefix_lengths` from ExactKV traces — **only** when generation smoke runs. Not comparable to full-panel compressor leaderboard rows.

---

## 7. Memory claim note

The adapter **compresses then immediately materialises** full dequant K/V tensors for draft forwards. `stored_kv_bytes` counts compressed CPU payloads; `materialized_working_kv_bytes` reflects full-precision layout during draft.

**No active GPU memory savings claim.** `supports_real_bytes_claim=False`.

---

## 8. Limitations

- Minimal calibration (2 prompts) — not paper reproduction.
- Small prompt smoke — not V10 full panel.
- CPU-safe path; CUDA kernel engine not required.
- Lossy quant may diverge from full KV on some prompts.
- External SpectralQuant metrics remain **not** ExactKV results.

---

## 9. No-overclaim section

**Allowed:**
- Factory-only adapter smoke ran on N prompts with reported `exactkv_failures`.
- Mean acceptance on cited smoke panel only.
- Adapter is experimental / restricted.

**Forbidden:**
- “SpectralQuant is integrated.”
- Speedup, throughput, latency, tokens/sec.
- Active VRAM / memory savings.
- Production serving or vLLM integration.
- Model accuracy improvement.
- Ranking against full-panel compressors without tier separation.

---

## Run commands

```bash
export SPECTRALQUANT_REPO_PATH=~/spectralquant
python3 scripts/research/run_exp044_spectralquant_adapter_smoke.py
pytest tests/test_spectralquant_adapter_smoke.py -q
```
