# ExactKV — Project Status (v0.4.0)

**Status date:** 2026-06-08
**Current tag:** `v0.4.0`
**Audience:** internal. This is a private status document, not a public release announcement.

---

## 1. Current status

ExactKV is a correctness-first, compressor-agnostic KV-cache verification runtime
and benchmark suite, inspired by the VeriCache paper. As of `v0.4.0` it can run a
lossy KV-cache compressor to draft tokens, verify each drafted token against
full-KV greedy decoding, and correct any divergence so the final output is
identical to standard full-KV inference under greedy decoding. The project has
moved from a bare correctness prototype (v0.1.0) through a benchmark/analysis
framework (v0.2.0), a presentation and reporting layer (v0.3.0), and now an
asymmetric K/V compression experiment layer (v0.4.0). The headline v0.4.0 result
is Experiment 003: a 612-run asymmetric K/V sweep with **0 ExactKV failures** that
shows keys are far more fragile than values under compression. ExactKV reports
exactness, acceptance, divergence, rejection, and correction behaviour only — it
makes no speedup, throughput, latency, runtime, or production-readiness claims.

---

## 2. Version timeline

| Version | Theme | What it delivered |
|---|---|---|
| `v0.1.0` | Correctness prototype | `generate_full_greedy` matches `model.generate`; draft-verify-commit loop; accept/reject/correction bookkeeping; `exactkv_output_ids == full_output_ids`. |
| `v0.2.0` | Framework, CLI, reports, sweeps, analysis | Compressor registry + capabilities; `int4_sim`; JSON/CSV reports with manifest; `run_sweep`; analysis layer; CLI (`list-compressors`, `bench`, `sweep`, `analyze`). |
| `v0.3.0` | Prompt suites, Markdown reporting, experiment docs | `core`/`structured`/`code`/`stress` suites; histogram + example analysis; `exactkv/reporting/` Markdown layer; `report` CLI; Experiment 002. |
| `v0.4.0` | Asymmetric K/V compressor experiments | `AsymmetricQuantSimCompressor` + 7 named compressors; K/V bit-width metadata in capabilities, reports, leaderboards; Experiment 003. |

---

## 3. What ExactKV currently proves

* Under greedy decoding, ExactKV output token IDs **exactly match** full-KV
  greedy output token IDs for every compressor tested (`exactkv_failures == 0`).
* Lossy divergence (the unverified compressed-KV draft differing from full-KV
  output) is **detected and corrected** by the verification engine. Lossy
  divergence is expected and is **not** an ExactKV failure.
* Accept / reject / correction bookkeeping reconciles
  (`drafted == accepted + rejected`).
* The compressor registry, sweep orchestration, analysis layer, and Markdown
  reporting pipeline function correctly across symmetric and asymmetric
  compressors.
* Acceptance behaviour differs sharply by compression policy, and asymmetric
  K/V policies can be compared directly against symmetric ones.

---

## 4. What ExactKV does not currently prove

* **No speedup, throughput, latency, or runtime claim.** ExactKV does not measure
  wall-clock time, tokens/second, or any performance metric.
* **No real memory savings for simulated compressors.** `_sim` compressors store
  sub-INT8 values in `int8` containers; their byte counts reflect `int8` storage,
  not real packed savings. `supports_real_bytes_claim=False` for these.
* **No real compressor backends.** No KIVI, kvpress, TurboQuant, KVQuant, or
  SnapKV. All compressors are PyTorch research implementations.
* **No real INT4/INT2 bit-packing.**
* **No workspace-aware memory accounting** (stored vs materialized vs scratch).
  Deferred to V5.
* **No production-readiness claim.** ExactKV runs with locally cached weights
  under a research/experimental framework.
* **No serving stack** (vLLM, LMCache, Triton, CUDA kernels, CPU offload), no
  batching, no sampling, no parallel verification, no bonus-token acceptance.
* **Average effective bit width is a comparison aid only**, not a real memory
  metric.

---

## 5. Experiment 003 headline result

A 612-run core-suite sweep (`Qwen/Qwen2.5-0.5B`, 34 prompts × 9 compressors ×
2 draft lengths, `max_new_tokens=24`):

| Metric | Value |
|---|---|
| Total runs | 612 |
| **ExactKV failures** | **0** |
| Lossy divergences (corrected) | 386 |
| Mean acceptance rate | 0.739 |
| Acceptance @ draft_len=4 | 0.782 |
| Acceptance @ draft_len=8 | 0.695 |

**Keeping keys at full precision while compressing values to INT8 (`k_full_v8`)
gave the best acceptance rate at 0.988.** Compressing keys to simulated 4-bit
(`k4_v8_sim`, `k4_v_full_sim`) collapsed acceptance to ~0.56, even when values
stayed at INT8 or full precision.

See [`EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md)
for the full report.

---

## 6. Asymmetric K/V result table

| Compressor | K bits | V bits | Simulated | Accept rate | Rejected | ExactKV failures |
|---|---|---|---|---|---|---|
| `k_full_v8` | full | 8 | no | **0.988** | 22 | 0 |
| `k8_v_full` | 8 | full | no | 0.953 | 86 | 0 |
| `int8` | 8 | 8 | no | 0.953 | 89 | 0 |
| `k_full_v4_sim` ⚠️ | full | 4 | yes | 0.890 | 174 | 0 |
| `k8_v4_sim` ⚠️ | 8 | 4 | yes | 0.858 | 240 | 0 |
| `k4_v8_sim` ⚠️ | 4 | 8 | yes | 0.562 | 1253 | 0 |
| `k4_v_full_sim` ⚠️ | 4 | full | yes | 0.561 | 1255 | 0 |
| `int4_sim` ⚠️ | 4 | 4 | yes | 0.553 | 1272 | 0 |
| `k8_v2_sim` ⚠️ | 8 | 2 | yes | 0.330 | 2302 | 0 |

> ⚠️ Simulated compressors store sub-INT8 values in `torch.int8` containers.
> `_sim` means simulated sub-INT8 numeric quantization, not real bit-packing.
> `k8_v_full` and `k_full_v8` carry no `_sim` suffix because they use only full
> precision and INT8.

---

## 7. Why the result matters

* It supports the V4 thesis: **keys and values should not necessarily be
  compressed symmetrically.** Keys feed the query–key dot product and softmax,
  so key errors propagate multiplicatively through attention routing; value
  errors are additive after attention weights are formed.
* Empirically, **values tolerated simulated 4-bit far better than keys.** The
  best asymmetric policy (`k_full_v8`) beat symmetric `int8`, while aggressive
  key compression (`k4_*`) was consistently the most damaging.
* It demonstrates the value of evaluating KV compression by **acceptance
  behaviour under full-KV verification** rather than by raw reconstruction MSE.
  Two compressors at the same average effective bit width can have very
  different acceptance rates.
* Across all 612 runs, **lossy divergence was always corrected** — ExactKV
  failures stayed at 0 — so the framework characterises compressor quality
  without ever sacrificing output correctness.

### Evidence, not universal proof

These V4 results are **evidence aligned with** the asymmetric-K/V thesis, not
universal proof of it. Specifically:

* Results are on a **single small model** (`Qwen/Qwen2.5-0.5B`) and do not
  guarantee cross-family behaviour.
* All asymmetric compressors are **simulated** sub-INT8 quantizers in `int8`
  containers — not real packed-bit backends. The acceptance differences reflect
  numeric quantization error, not a specific production format.
* The cleanest matched-budget comparison is **`k8_v4_sim` (0.858) vs
  `k4_v8_sim` (0.562)** — same average bit budget, keys-favoured wins — rather
  than simply "`k_full_v8` is best."

The direction matches external work (KV-AdaQuant's key–value norm-disparity
theory; KIVI's per-channel key handling; TurboQuant+'s "degradation comes from K"
findings), which strengthens confidence — but ExactKV's evidence on its own
remains one small-model, simulated-compressor data point.

### Related-work note

For how ExactKV sits relative to KV-cache quantization (KIVI, KVQuant,
KV-AdaQuant, TurboQuant), eviction (SnapKV, H2O, StreamingLLM, PyramidKV),
transform coding (Palu, KVTC), and serving systems (vLLM/PagedAttention,
LMCache) — and an explicit statement of what ExactKV does **not** implement — see
[`RELATED_WORK_KV_CACHE_COMPRESSION.md`](RELATED_WORK_KV_CACHE_COMPRESSION.md).
ExactKV is a verification/evaluation framework; it implements none of those
backends and makes no performance or production-readiness claims.

---

## 8. Current supported compressors

### Symmetric (V1–V3)

| Name | K bits | V bits | Simulated | Supports real bytes claim |
|---|---|---|---|---|
| `noop` | full | full | no | yes |
| `int8` | 8 | 8 | no | yes |
| `int4_sim` ⚠️ | 4 | 4 | yes | no |
| `debug_noise` | full | full | no | yes |

### Asymmetric (V4)

| Name | K bits | V bits | Simulated | Supports real bytes claim |
|---|---|---|---|---|
| `k8_v4_sim` ⚠️ | 8 | 4 | yes | no |
| `k8_v2_sim` ⚠️ | 8 | 2 | yes | no |
| `k4_v8_sim` ⚠️ | 4 | 8 | yes | no |
| `k_full_v4_sim` ⚠️ | full | 4 | yes | no |
| `k4_v_full_sim` ⚠️ | 4 | full | yes | no |
| `k8_v_full` | 8 | full | no | yes |
| `k_full_v8` | full | 8 | no | yes |

---

## 9. Known limitations

* **CPU sweep runtime.** The 612-run Experiment 003 sweep takes ~44 min on CPU.
  Reduce `--max-new-tokens` or use `--suite smoke` for quick iteration.
* **Sub-INT8 simulation.** `_sim` compressors store values in `int8` containers;
  memory figures reflect `int8` storage, not real packed savings.
* **Average effective bit width** treats full precision as 32 bits regardless of
  model dtype. It is a metadata comparison aid only.
* **`DynamicCache` brittleness.** Cache utilities target a specific transformers
  internal structure and may break across versions.
* **Sequential verification only.** Single-token draft-verify loop; no parallel
  speculative verification.
* **Single model family.** Targets `Qwen/Qwen2.5-0.5B`; no cross-family guarantee.

---

## 10. Recommended next step

**V5 — workspace-aware memory accounting and real backend planning.** The most
valuable next increment is an honest memory schema that distinguishes stored
compressed cache from materialized working cache, metadata, and temporary
scratch buffers, plus a design (not yet an implementation) for plugging real
quantisation backends into the existing `KVCompressor` protocol. See
[`V5_SCOPE_DRAFT.md`](V5_SCOPE_DRAFT.md). V5 must continue to make no speed or
production-readiness claims unless they are carefully measured with real
hardware and explicit disclaimers.
