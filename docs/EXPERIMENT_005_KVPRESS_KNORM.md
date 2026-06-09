# Experiment 005: Restricted KVPress KnormPress Comparison

_Generated 2026-06-09 by ExactKV in the isolated `[kvpress]` environment. See disclaimers below._

## 1. Purpose

**Question:** How does the restricted experimental `KVPressKnormAdapter` (KnormPress
token-dropping only) compare to ExactKV's built-in baselines on exactness,
acceptance behaviour, divergence/rejection/correction counts, and V5
workspace-memory accounting?

This experiment compares **acceptance and memory honesty only**. It does not
measure throughput, latency, wall-clock runtime, speedup, or production serving
behaviour.

**KVPressKnormAdapter is restricted to KnormPress.** It is **not** in the default
compressor registry. It requires the `[kvpress]` optional environment
(`.venv-kvpress`). It uses an **isolated compression model** (`deepcopy` of the
verification model) to avoid mutating the model used for verification and commit.

---

## 2. Environment

| Item | Value |
|---|---|
| Virtualenv | `.venv-kvpress` (`pip install -e ".[kvpress]"`) |
| Python | 3.13.3 |
| transformers | 5.2.0 |
| kvpress | 0.5.3 |
| torch | 2.12.0 |
| Default ExactKV env | transformers 5.8.x, **no kvpress** (unchanged) |
| `fire` workaround | `fire>=0.7.1` (Python 3.13; not added to default deps) |

Reproduce:

```bash
.venv-kvpress/bin/python scripts/run_experiment_005_kvpress_knorm.py
```

Artifacts (gitignored): `reports/experiment_005_kvpress_knorm.json`,
`reports/experiment_005_kvpress_knorm.csv`.

---

## 3. Restrictions

- **KnormPress only** — no DecodingPress, AdaKVPress, ComposedPress, or
  `KVPressTextGenerationPipeline`.
- **Not in default registry** — `kvpress_knorm_restricted` is constructed only
  via `create_kvpress_knorm_adapter()` inside the experiment script.
- **No default kvpress import** — `import exactkv.compressors` does not load kvpress.
- **`import kvpress` may globally patch** `ALL_ATTENTION_FUNCTIONS`; this experiment
  runs only in the isolated venv.
- **No broader kvpress pipelines or serving integrations** are implemented by ExactKV.
- ExactKV does **not** implement KIVI, KVQuant, TurboQuant, vLLM, LMCache, or
  PagedAttention.

---

## 4. Model and prompt suite

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`, float32, CPU |
| Prompt suite | `core` (34 prompts) |
| `draft_len` | 4 |
| `max_new_tokens` | 16 |
| KVPress `compression_ratio` | 0.5 |
| Total cells | **272** (34 prompts × 8 compressors) |

---

## 5. Compressor set

| Compressor | Type | Notes |
|---|---|---|
| `noop` | Identity baseline | Lossless |
| `int8` | Real INT8 quantization | `supports_real_bytes_claim=True` |
| `int4_sim` | Simulated INT4 in int8 containers | `is_simulated=True` |
| `k8_v4_sim` | Asymmetric simulated K8/V4 | `is_simulated=True` |
| `k_full_v8` | Real K-full / V8 | `is_simulated=False` |
| `k8_v_full` | Real K8 / V-full | `is_simulated=False` |
| `backend_passthrough` | V6 Phase B adapter PoC | Lossless |
| `kvpress_knorm_restricted` | **Restricted KnormPress adapter** | Not in default registry |

---

## 6. Exactness result

| Metric | Value |
|---|---|
| Total runs | 272 |
| **ExactKV failures** | **0** ✓ |
| `exactkv_output_ids == full_output_ids` | 272 / 272 |

ExactKV failure means ExactKV output differs from `generate_full_greedy` on the
same model — a correctness bug. **Zero failures** across all compressors including
`kvpress_knorm_restricted`.

---

## 7. Acceptance by compressor

| compressor | simulated | accept_rate | avg_accept_len | drafted | accepted | rejected | corrections | runs | exactkv_fail |
|---|---|---|---|---|---|---|---|---|---|
| noop | no | 1.000 | 3.85 | 489 | 489 | 0 | 0 | 34 | 0 |
| backend_passthrough | no | 1.000 | 3.85 | 489 | 489 | 0 | 0 | 34 | 0 |
| k_full_v8 | no | 0.990 | 3.79 | 493 | 487 | 6 | 2 | 34 | 0 |
| int8 | no | 0.961 | 3.62 | 501 | 478 | 23 | 11 | 34 | 0 |
| k8_v_full | no | 0.963 | 3.62 | 501 | 479 | 22 | 10 | 34 | 0 |
| k8_v4_sim | yes | 0.891 | 3.33 | 522 | 465 | 57 | 24 | 34 | 0 |
| int4_sim | yes | 0.628 | 2.23 | 655 | 375 | 280 | 114 | 34 | 0 |
| **kvpress_knorm_restricted** | no | **0.413** | **1.50** | 774 | 305 | 469 | 184 | 34 | 0 |

Lossy compressors show lower draft acceptance; **kvpress has the lowest acceptance
rate** in this set because KnormPress aggressively prunes KV tokens. Final ExactKV
output remains exact because verification uses authoritative full KV.

---

## 8. Divergence / rejection / correction summary

| Metric | Value |
|---|---|
| Lossy divergences (lossy ≠ full greedy) | 102 / 272 cells |
| Aggregate drafted tokens | 4,424 |
| Aggregate accepted | 3,567 |
| Aggregate rejected | 857 |
| Aggregate corrections | 355 |

**kvpress_knorm_restricted:** 469 rejected draft tokens, 184 corrections across 34
prompts. Lossy draft divergence is **expected**; it is not an ExactKV failure.

---

## 9. Workspace-memory accounting table

Example prefill snapshot (`core_nat_001`, 5 prompt tokens; values vary by prompt):

| Compressor | Stored KV | Materialized KV | Metadata | Temp | Total footprint † | Real bytes? | Simulated? |
|---|---|---|---|---|---|---|---|
| `noop` | 120.0 KiB | 120.0 KiB | 0 B | 0 B | 240.0 KiB | no | no |
| `int8` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | yes | no |
| `int4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no | yes ⚠️ |
| `k8_v4_sim` | 30.0 KiB | 120.0 KiB | 384 B | 0 B | 150.4 KiB | no | yes ⚠️ |
| `k_full_v8` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | yes | no |
| `k8_v_full` | 75.0 KiB | 120.0 KiB | 192 B | 0 B | 195.2 KiB | yes | no |
| `backend_passthrough` | 120.0 KiB | 120.0 KiB | 0 B | 0 B | 240.0 KiB | no | no |
| **`kvpress_knorm_restricted`** | **48.0 KiB** | **48.0 KiB** | 0 B | 0 B | **96.0 KiB** | **yes** | no |

† Total footprint = stored + materialized + metadata + temporary workspace. This is a
**conservative accounting sum, NOT a measured peak GPU memory value**. Active GPU
memory is **not** reported.

**kvpress byte semantics:** `stored_kv_bytes` and `materialized_working_kv_bytes`
reflect **real pruned `DynamicCache` tensor bytes**, not packed low-bit quantization.
For kvpress, `materialized_working_kv_bytes == stored_kv_bytes` (token-dropping;
no separate dequantize step). `supports_real_bytes_claim=True` applies to this
pruned-cache storage only.

**Simulated compressors** (`int4_sim`, `k8_v4_sim`) store sub-INT8 values in int8
containers — `stored_kv_bytes` reflects int8 container reality, not packed-bit savings.

---

## 10. Hook-safety result

All 34 `kvpress_knorm_restricted` cells passed:

| Check | Result |
|---|---|
| Verification model hooks before/during/after | **0** always |
| Compression model hooks before compress | 0 |
| Compression model hooks during `with press(model):` | 24 (24 Qwen layers) |
| Compression model hooks after compress | 0 |
| `verify_hooks_always_zero` | **PASS** |
| `compress_hooks_return_to_zero` | **PASS** |

---

## 11. Physical vs logical sequence result

All 34 kvpress cells:

| Field | Result |
|---|---|
| `logical_seq_len` | Equals full prefill length (alignment invariant) |
| Physical `kv_seq_len` (pruned cache) | Strictly less than logical on every prompt |
| `pruning_on_all_prompts` | **PASS** |

Example: `core_nat_001` — logical 5, physical 2 under `compression_ratio=0.5`.

---

## 12. What this proves

- A **restricted real-backend adapter** (`KVPressKnormAdapter`, KnormPress only)
  can plug into ExactKV's draft-verify-commit loop with **`exactkv_failures == 0`**
  on the full core suite.
- Hook isolation and full-state verification gates hold when compression uses an
  isolated model copy.
- **Honest workspace accounting** distinguishes kvpress pruned-cache bytes from
  simulated int8-container bytes and from dequantise-to-full INT8 baselines.
- Lossy draft behaviour (low acceptance, frequent corrections) is measurable and
  separable from ExactKV exactness.

---

## 13. What this does not prove

- **No speedup, throughput, latency, runtime, or production-readiness claim.**
  ExactKV does not measure tokens/second or wall-clock time.
- **No GPU memory measurement** — byte counts are tensor-size accounting only.
- **No claim about kvpress's external benchmark numbers** as ExactKV results.
- **No broader kvpress support** — DecodingPress, AdaKVPress, pipelines, and serving
  stacks are out of scope.
- **Single model / CPU only** — GPU behaviour and larger models not evaluated.
- ExactKV does **not** implement broader compression backends (KIVI, KVQuant, etc.).

---

## 14. Relation to V6 and future V7/V8

- **V6 Phase D (this experiment):** First real-backend comparison report; restricted
  kvpress KnormPress only.
- **V7:** Deeper attention-aware and asymmetric real-backend experiments (KIVI,
  KVQuant-style adapters deferred from V6).
- **V8:** Serving-stack evaluation context only; active GPU profiling deferred.

See [`docs/V6_SCOPE_STATEMENT.md`](V6_SCOPE_STATEMENT.md).

---

## 15. VeriCache attribution

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV implements this verification pattern; it does not claim to have invented it.

---

## 16. kvpress attribution

kvpress is an external Hugging Face–compatible KV-cache compression library
(NVIDIA, [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)).

ExactKV does **not** implement kvpress. This experiment wraps **KnormPress only**
via a restricted adapter in an isolated optional environment. kvpress performance
or serving claims are **not** cited as ExactKV results.

---

## Interpretation notes

> * **Lossy divergence is expected and is not an ExactKV failure.** The compressor
>   alters the draft KV cache; verification corrects via the full authoritative state.
> * **ExactKV failure** means ExactKV output differs from full-KV greedy output.
> * **Simulated compressors** store sub-INT8 values in int8 containers; do not cite
>   their byte counts as packed-bit savings.
> * **kvpress stored bytes** are real pruned DynamicCache bytes, not quantization.
> * **total_kv_footprint_bytes** is a conservative accounting sum, not measured peak
>   GPU memory. Active GPU memory is not reported.
> * **ExactKV does not claim speedup, throughput, latency, runtime, or production
>   readiness.**
