# Faithful External Compressor Integration

**Status:** Phase D3 — KIVI r32 + SnapKV panel wiring (June 2026).

ExactKV's headline panel uses built-in simulations (`int4_sim`, `int6_sim`,
`int4_per_vec_sim`, `h2o_sim`). This document covers **faithful external
compressor adapters** that wrap upstream libraries' real algorithms (not ExactKV
tensor approximations).

---

## 1. Integration tiers

| Tier | Compressor | Backend | Faithful to upstream? | Production CUDA? |
|------|------------|---------|----------------------|----------------|
| **A — Faithful quant math** | `kivi_offline_r32` | jy-yuan/KIVI `models.utils_quant` | Yes (simulate path + r=32 streaming policy) | No |
| **A — Faithful eviction** | `snapkv_experimental` | kvpress `SnapKVPress` | Partial (kvpress replay, not paper-exact) | No |
| **A — Faithful eviction** | `kvpress_knorm_experimental` | kvpress `KnormPress` | Partial (kvpress replay; Exp 005 ~0.97 accept) | No |
| **B — Faithful simquant** | `kvquant_sim` | KVQuant `QuantLinearSim` | Yes (pre-RoPE simquant) | No |
| **B — Faithful quant (Python)** | `turboquant_experimental` | TheTom/turboquant_plus | Yes (offline NumPy bridge; Exp 008) | No |
| **C — Diagnostic (incomplete policy)** | `kivi_offline` | KIVI simulate, no residual window | Algorithm incomplete vs Kinds KIVI | No |
| **D — Blocked** | KIVI production CUDA | `dequant_cuda`, `kivi_gemv` | N/A | Blocked (Exp 024) |

**Headline claim boundary:** Tier A–B adapters are evaluated in separate
`reports/external_panels/faithful/` panels. They are **not** merged into the
8,132-cell headline count unless explicitly integrated after GPU validation.

---

## 2. KIVI offline r32 (`kivi_offline_r32`)

### What changed (Phase D3)

The original `kivi_offline` adapter quantized the **entire** KV sequence uniformly.
KIVI production keeps the last `residual_length` tokens (default **32**) in fp16
before quantizing the prefix — a streaming policy critical for generation quality.

Phase D3 adds `residual_length` to `KIVIOfflineAdapter`:

- Factory: `create_kivi_offline_adapter(runtime, residual_length=32)`
- Panel alias: `kivi_offline_r32` → name `kivi_offline_k2_v2_r32`
- Legacy `kivi_offline` (no residual) retained as **adapter diagnostic** (640-cell
  panel showed 100% divergence — incomplete policy, not a KIVI algorithm claim)

### Environment

```bash
git clone https://github.com/jy-yuan/KIVI.git /tmp/kivi_research
export PYTHONPATH=/tmp/kivi_research
# Or: bash scripts/setup_faithful_compressor_env.sh
```

### Run panel (RunPod GPU)

```bash
tmux new-session -s faithful
bash scripts/run_faithful_compressor_panel.sh 2>&1 | tee reports/faithful_panel.log
```

Outputs: `reports/external_panels/faithful/{longbench,bfcl,mbpp}_{MODEL}_raw.json`

---

## 3. SnapKV experimental (`snapkv_experimental`)

Uses [kvpress](https://github.com/NVIDIA/kvpress) `SnapKVPress` with replay prefill
under `with press(model):`. Isolated draft model clone; verifier uses full-KV runtime.

```bash
pip install kvpress
```

Included in `run_faithful_compressor_panel.sh` alongside `kivi_offline_r32` and `int8`
control.

**Claim boundary:** Restricted experimental adapter — not paper-exact SnapKV, not
production SnapKV serving.

---

## 3b. KnormPress experimental (`kvpress_knorm_experimental`)

Same kvpress replay pattern as SnapKV but uses `KnormPress` (Exp 005 validated,
~0.97 mean acceptance on core suite). **Best candidate** for a faithful external
compressor that may show non-catastrophic drift on structured tasks.

Wired in `evidence_plus_panel.py`. Included in wave-2 smoke:

```bash
bash scripts/run_faithful_external_wave2_smoke.sh
```

---

## 3c. TurboQuant Python (`turboquant_experimental`)

Offline NumPy `KVCacheCompressor` bridge (Exp 008, `exactkv_failures=0`, accept ~0.435
on core suite). Requires:

```bash
INSTALL_TURBOQUANT=1 bash scripts/setup_faithful_compressor_env.sh
export EXACTKV_TURBOQUANT_ROOT=/tmp/turboquant_plus
```

Included in wave-2 smoke for side-by-side comparison with KnormPress and SnapKV.

---

## 4. KVQuant sim (`kvquant_sim`) — Llama path (future)

KVQuant simquant is implemented (`exactkv/compressors/kvquant_adapter.py`) and
validated on Qwen 0.5B/1.5B (Exp 010/023). Extending to Llama-3.1-8B requires:

1. Isolated venv with `transformers~=4.44` (conflicts with ExactKV 5.x panels)
2. Calibration run → `quantizers.pickle` for Llama-3.1-8B
3. `EXACTKV_KVQUANT_QUANTIZERS=/path/to/quantizers_llama8b.pickle`

Reference scripts: `scripts/research/kvquant_runpod_synthetic_calib.py`,
`scripts/run_experiment_023_kvquant_larger_model.py`.

---

## 5. Paper / site integration checklist

After GPU panel completes:

1. Add §6.17 Faithful external compressor panel with `kivi_offline_r32` + SnapKV results
2. Update benchmark card appendix with faithful panel cell counts
3. Distinguish `kivi_offline` (diagnostic, no r32) from `kivi_offline_r32` (faithful policy)
4. Update limitations: production KIVI CUDA still blocked (Exp 024)

---

## 6. Tests

```bash
pytest tests/test_kivi_residual_window.py -q          # no KIVI clone needed
PYTHONPATH=/tmp/kivi_research pytest tests/test_kivi_adapter.py -q  # full adapter
```
