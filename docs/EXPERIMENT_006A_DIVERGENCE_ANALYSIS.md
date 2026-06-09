# Experiment 006A: Proxy Divergence Analysis (V7 Phase A)

_Generated 2026-06-09 by ExactKV analysis tooling. Analysis-only — no new compressors,
no generation runs._

## 1. Purpose

**Question:** Where do lossy drafts diverge from full-KV greedy output, and how does
that relate to acceptance, rejection, and correction behaviour across existing
Experiments 003, 004, and 005?

This is **proxy divergence analysis** unless actual attention weights are logged.
Experiments 003–005 contain `first_divergence_idx`, acceptance traces, and rejection
counts — but **no raw attention weights**. No attention weights are fabricated.
ExactKV does **not** claim causal attention importance from this report alone.

---

## 2. Data sources

| Source | Path | Cells | Notes |
|---|---|---:|---|
| Experiment 003 | `reports/experiment_003_asymmetric_kv_sweep.json` | 612 | 9 compressors × 34 prompts × 2 draft lengths; `max_new_tokens=24` |
| Experiment 004 | `reports/experiment_004_workspace_memory.json` | 340 | 10 compressors × 34 prompts; `max_new_tokens=16` |
| Experiment 005 | `reports/experiment_005_kvpress_knorm.json` | 272 | 8 compressors × 34 prompts; `max_new_tokens=16`; includes `kvpress_knorm_restricted` |

All three JSON reports were present locally. Analysis module:
`exactkv/analysis/attention_weighted.py`.

Markdown experiment docs (`docs/EXPERIMENT_003_*.md`, etc.) were used for
cross-checking aggregate acceptance tables only.

**Attention weights available:** **No** (`has_attention_weights == False` on all
three reports).

---

## 3. What is measured

From existing report fields only:

- **Lossy divergence rate** — fraction of cells where `lossy.token_exact_match == False`
- **First-divergence position** — `lossy.first_divergence_idx` (token index in output)
- **Acceptance rate** — `exactkv.acceptance.acceptance_rate` per cell
- **Rejection and correction counts** — `total_rejected`, `total_corrections`
- **Cross-report divergence rates** — for compressors present in multiple experiments

---

## 4. What is not measured

- Raw attention weights, attention entropy, or per-head attention maps
- Causal importance of high-attention vs low-attention tokens
- Throughput, latency, wall-clock runtime, speedup, or production serving behaviour
- GPU memory peaks (V5 accounting fields are not re-analysed here)
- New generation runs (this phase reuses existing reports)

---

## 5. Divergence by compressor

### Experiment 003 (612 cells, `exactkv_failures == 0`)

| Compressor | Divergence rate | Mean first-div idx | Lossy cells |
|---|---|---:|---:|
| `k8_v2_sim` | 0.971 | 2.1 | 66 / 68 |
| `int4_sim` | 0.882 | 2.5 | 60 / 68 |
| `k4_v8_sim` | 0.853 | 3.1 | 58 / 68 |
| `k4_v_full_sim` | 0.853 | 3.1 | 58 / 68 |
| `k8_v4_sim` | 0.676 | 6.8 | 46 / 68 |
| `k_full_v4_sim` | 0.647 | 8.0 | 44 / 68 |
| `int8` | 0.353 | 9.4 | 24 / 68 |
| `k8_v_full` | 0.353 | 9.2 | 24 / 68 |
| `k_full_v8` | 0.088 | 8.0 | 6 / 68 |

### Experiment 005 (272 cells, `exactkv_failures == 0`)

| Compressor | Divergence rate | Mean first-div idx | Lossy cells |
|---|---|---:|---:|
| **`kvpress_knorm_restricted`** | **0.971** | **1.9** | **33 / 34** |
| `int4_sim` | 0.882 | 2.5 | 30 / 34 |
| `k8_v4_sim` | 0.559 | 4.2 | 19 / 34 |
| `int8` | 0.265 | 6.2 | 9 / 34 |
| `k8_v_full` | 0.265 | 5.9 | 9 / 34 |
| `k_full_v8` | 0.059 | 3.0 | 2 / 34 |
| `noop` / `backend_passthrough` | 0.000 | — | 0 / 34 |

**Pattern:** Aggressive compressors diverge on **most** cells; divergence tends to
occur **early** (mean index 1.9–3.1 for the worst offenders). Conservative
asymmetric policies (`k_full_v8`) diverge rarely and late.

---

## 6. Rejection / correction by compressor (Experiment 005)

| Compressor | Mean accept rate | Total rejected | Total corrections |
|---|---|---:|---:|
| `noop` / `backend_passthrough` | 1.000 | 0 | 0 |
| `k_full_v8` | 0.990 | 6 | 2 |
| `k8_v_full` / `int8` | ~0.96 | 22–23 | 10–11 |
| `k8_v4_sim` | 0.891 | 57 | 24 |
| `int4_sim` | 0.628 | 280 | 114 |
| **`kvpress_knorm_restricted`** | **0.413** | **469** | **184** |

Rejection and correction totals reconcile with per-cell acceptance traces.
High divergence rate co-occurs with high rejection count — expected for lossy
compressors; final ExactKV output remains exact (`exactkv_failures == 0`).

---

## 7. First-divergence position patterns

### Experiment 005 — position buckets (`kvpress_knorm_restricted` vs baselines)

| Compressor | `no_divergence` | `1-4` (early) | `5-16` | `17+` |
|---|---:|---:|---:|---:|
| `kvpress_knorm_restricted` | 1 | **30** | 3 | 0 |
| `int4_sim` | 4 | **26** | 4 | 0 |
| `k8_v4_sim` | 15 | 14 | 5 | 0 |
| `k_full_v8` | 32 | 1 | 1 | 0 |
| `noop` | 34 | 0 | 0 | 0 |

**Observation (proxy only):** The most aggressive compressors (`kvpress_knorm_restricted`,
`int4_sim`) show **early** first divergence (bucket `1-4` on 26–30 of 34 prompts).
This is consistent with lossy KV errors affecting routing before many tokens are
generated — but this report does **not** log attention weights to confirm
high-attention positions are involved.

---

## 8. Acceptance vs divergence relationship

Experiment 005 joint summary (sorted by divergence rate):

| Compressor | Divergence rate | Mean accept rate | Mean first-div idx |
|---|---:|---:|---:|
| `kvpress_knorm_restricted` | 0.971 | 0.413 | 1.9 |
| `int4_sim` | 0.882 | 0.628 | 2.5 |
| `k8_v4_sim` | 0.559 | 0.891 | 4.2 |
| `int8` / `k8_v_full` | 0.265 | ~0.96 | ~6.0 |
| `k_full_v8` | 0.059 | 0.990 | 3.0 |
| `noop` | 0.000 | 1.000 | — |

**Proxy correlation:** Lower acceptance rate co-occurs with higher divergence rate
and earlier mean first-divergence index. This is **not** a causal attention claim.

---

## 9. Simulated compressors vs kvpress Knorm (Experiment 005)

| Aspect | Simulated (`int4_sim`, `k8_v4_sim`) | `kvpress_knorm_restricted` |
|---|---|---|
| `is_simulated` | `True` | `False` |
| `supports_real_bytes_claim` | `False` (int8 containers) | `True` (pruned DynamicCache bytes) |
| Divergence rate | 0.559–0.882 | **0.971** |
| Mean first-div idx | 2.5–4.2 | **1.9** |
| Accept rate | 0.628–0.891 | **0.413** |

**kvpress KnormPress** is the most divergence-heavy compressor in Experiment 005 —
comparable to `k8_v2_sim` in Experiment 003/004 (0.971 divergence rate, mean idx ~2.1).
Token-dropping and aggressive simulated quantization both produce **early, frequent**
lossy divergence under this proxy analysis.

Simulated byte counts must not be compared to kvpress pruned-cache bytes without
explicit labelling (V5/V6 honesty rules).

---

## 10. Cross-report stability (overlapping compressors)

Compressors present in multiple experiments show **stable qualitative ordering**:

| Compressor | Div. rate exp003 | Div. rate exp004 | Div. rate exp005 |
|---|---:|---:|---:|
| `k8_v2_sim` | 0.971 | 0.971 | — |
| `int4_sim` | 0.882 | 0.882 | 0.882 |
| `k8_v4_sim` | 0.676 | 0.559 | 0.559 |
| `k_full_v8` | 0.088 | 0.059 | 0.059 |
| `int8` | 0.353 | 0.265 | 0.265 |

**Caveat:** Experiment 003 used `max_new_tokens=24` and two draft lengths; Experiments
004–005 used `max_new_tokens=16` and `draft_len=4` only. Absolute divergence rates
are not directly comparable across experiments — only compressor **ranking** is stable.

---

## 11. What this suggests for V7 Phase B

1. **Analysis-first gate passed:** Existing reports already separate aggressive vs
   conservative compressors by divergence timing. Phase B should not jump to real
   TurboQuant/KVQuant/KIVI adapters without a narrower hypothesis.

2. **Early divergence cluster:** Aggressive policies diverge in output positions 1–4
   on most prompts. A Phase B **simulated layer-aware or position-aware V policy**
   should be scoped to test whether protecting early decode steps improves acceptance
   — not whether attention weights are "high" at those positions (that requires
   optional attention logging in a future sub-phase).

3. **kvpress is not a stand-in for asymmetric quant:** KnormPress token-dropping
   behaves like the most aggressive simulated compressors on divergence metrics.
   Real asymmetric quant comparison (KIVI-style) remains a **separate approval**
   path (Experiment 006b), not the default Phase B entry.

4. **Do not build Sparse V or pre-RoPE yet:** Proxy analysis does not justify
   implementation complexity until optional attention logging confirms whether
   divergence aligns with attention concentration.

---

## 12. What it does not prove

- **No causal attention importance** — proxy indices only; no attention weights logged.
- **No speedup, throughput, latency, runtime, or production readiness claim.**
- **No reproduction of TurboQuant+, KV-AdaQuant, KIVI, or KVQuant paper numbers.**
- **No claim that early divergence implies high-attention tokens** — that requires
  future instrumentation (see §13).
- **No claim that a new compressor policy will improve acceptance** — Phase B is
  conditional on separate approval.

---

## 13. Related work context

| External work | Relevance to this proxy analysis |
|---|---|
| **KV-AdaQuant** | V4 Experiment 003 showed key compression hurts acceptance more than value compression; this analysis shows **early divergence** for key-fragile policies — aligned direction, not a reproduction. |
| **TurboQuant+** | Reports aggressive V may be "nearly free" with rotation; ExactKV's `k8_v2_sim` still diverges early (0.971) — naive sim ≠ TurboQuant; proxy analysis supports testing policy-shaped experiments in Phase B, not claiming TurboQuant+ results. |
| **KIVI / KVQuant** | Motivate real asymmetric and pre-RoPE adapters; not evaluated in 006A. |
| **PyramidKV / eviction methods** | Motivate layer-aware and token-dropping policies; kvpress Knorm is token-dropping only — divergence patterns inform eviction vs quant separation, not eviction implementation. |

External-paper accuracy, perplexity, and speed claims are **not** ExactKV results.

---

## 14. Optional attention instrumentation (design note — not implemented)

To upgrade from **proxy** to **true attention-weighted** analysis in a future
sub-phase:

| Requirement | Rationale |
|---|---|
| Optional one-pass attention logging | Record per-step attention entropy or top-k weight sum at divergence positions |
| Prompt subset only | Core suite subset (e.g. 10 prompts) to limit storage |
| No generation-logic change | Logging is read-only side channel; draft-verify-commit unchanged |
| Additive report fields only | e.g. `attention_entropy_at_first_divergence` per cell |
| Memory caveats | Attention tensors are large; log scalars only |
| No performance claims | Logging must not introduce `runtime_seconds` or throughput fields |

**Not implemented in V7 Phase A** — existing reports lack these fields.

---

## 15. No-performance-claim note

ExactKV does not measure tokens/second, throughput, latency, wall-clock runtime,
speedup, or production readiness in this analysis.

Forbidden fields (`tokens_per_second`, `throughput`, `latency`, `speedup`,
`runtime_seconds`) do not appear in analysis module outputs.

---

## Reproduce

```bash
python -c "
import json
from pathlib import Path
from exactkv.analysis.attention_weighted import (
    divergence_by_compressor, compare_reports_for_divergence,
)
reports = {
    'experiment_003': json.loads(Path('reports/experiment_003_asymmetric_kv_sweep.json').read_text()),
    'experiment_005': json.loads(Path('reports/experiment_005_kvpress_knorm.json').read_text()),
}
print(compare_reports_for_divergence(reports)['overlapping_compressors'])
"
```

---

## VeriCache attribution

Draft-then-verify algorithm: Yao et al., arXiv:2605.17613, 2026. ExactKV implements
verification; it does not claim to have invented it.
