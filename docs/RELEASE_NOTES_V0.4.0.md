# ExactKV v0.4.0 — Release Notes

**Release date:** 2026-06-08
**Tag:** `v0.4.0`
**Previous release:** `v0.3.0` (V3 presentation and storytelling layer)

---

## What V1 proved (v0.1.0-phase1)

1. `generate_full_greedy` exactly matches `model.generate(do_sample=False)`.
2. Cache alignment invariant: after every round, `FullKVState` and
   `CompressedKVState` represent the same committed token prefix.
3. Accept/reject/correction bookkeeping reconciles: `drafted == accepted + rejected`.
4. Three-mode benchmark runner (full / lossy / exactkv) emits structured JSON.
5. Primary correctness criterion: `exactkv_output_ids == full_output_ids`.

---

## What V2 added (v0.2.0)

* **`int4_sim` compressor** — simulated INT4 (numeric INT4 range, `int8` container
  storage; flagged `is_simulated=True`, `supports_real_bytes_claim=False`).
* **Compressor registry** — `register_compressor`, `get_compressor`,
  `list_compressors`; `CompressorCapabilities` metadata on every compressor.
* **Shared prefill helper** — `prefill_to_full_state` centralises prefill logic.
* **`BenchmarkConfig`** — unified experiment-level config alongside `ExactKVConfig`.
* **Reporting layer** — stable JSON/CSV reports with manifest provenance, memory
  honesty fields (`memory_claim_note` for `int4_sim`).
* **Sweep orchestration** — `run_sweep` over multiple compressors × draft lengths.
* **Analysis layer** — acceptance tables, mismatch position summaries, failure
  classification (lossy divergence ≠ ExactKV failure).
* **CLI** — `list-compressors`, `bench`, `sweep`, `analyze`.
* **`smoke.jsonl`** — 16-prompt fast CI suite.

---

## What V3 added (v0.3.0)

V3 turned the V2 benchmark framework into a **presentation and storytelling layer** —
making results legible to humans while remaining performance-silent.

* **Four curated JSONL prompt suites** — `core` (34), `structured` (28), `code` (30),
  `stress` (25). All five named suites (`smoke`, `core`, `structured`, `code`,
  `stress`) resolve from Python and the CLI.
* **`exactkv/analysis/histograms.py`** — accepted-length, first-divergence, and
  rejection-count histogram builders.
* **`exactkv/analysis/examples.py`** — lossy-divergence, ExactKV-failure, and
  rejection example extractors; no model re-run.
* **`exactkv/reporting/` package** — `render_markdown_report` / `write_markdown_report`,
  acceptance leaderboards, example renderers, histogram text tables. No plots,
  no images.
* **CLI `report` command** — `python -m exactkv report` renders any existing
  JSON report to Markdown without re-running the model.
* **`docs/EXPERIMENT_002_CORE_SWEEP.md`** — 204-run core-suite sweep
  (`noop × int8 × int4_sim`, draft lengths 4/8, `max_new_tokens=24`);
  0 ExactKV failures.

---

## What V4 adds (v0.4.0)

V4 brings **asymmetric K/V compression experiments** — the ability to compress
keys and values at different bit-widths and compare acceptance behaviour across
these policies.

### Phase A — Capability metadata extension

`CompressorCapabilities` now carries three new fields on every compressor:

| Field | Type | Purpose |
|---|---|---|
| `key_bit_width` | `int \| None` | K-side bit width; `None` = full precision |
| `value_bit_width` | `int \| None` | V-side bit width; `None` = full precision |
| `asymmetric` | `bool` | `True` when K and V have different widths |

All four V1–V3 compressors were backfilled (`int8`: 8/8/False; `int4_sim`: 4/4/False;
`noop`/`debug_noise`: None/None/False).

### Phase B — AsymmetricQuantSimCompressor core

New `AsymmetricQuantSimCompressor(k_bits, v_bits, name=None)` in
`exactkv/compressors/asymmetric_sim.py`:

* Supports `k_bits`/`v_bits` of `None`/`"full"`, `8`, `4`, `2`.
* `None` = full-precision passthrough for that side (bit-identical, no quantisation).
* Per-tensor symmetric quantisation with independent K and V scales.
* `is_simulated=True` and `supports_real_bytes_claim=False` when either side is
  sub-INT8 (4-bit or 2-bit).
* `is_simulated=False` and `supports_real_bytes_claim=True` when both sides are
  full precision or INT8 only.

### Phase C — Named asymmetric compressors and registry

Seven named, no-arg subclasses registered in the compressor registry:

| Name | K bits | V bits | `is_simulated` | Notes |
|---|---|---|---|---|
| `k8_v4_sim` ⚠️ | 8 | 4 | yes | V quantised to INT4 range, int8 container |
| `k8_v2_sim` ⚠️ | 8 | 2 | yes | V quantised to INT2 range, int8 container |
| `k4_v8_sim` ⚠️ | 4 | 8 | yes | K quantised to INT4 range, int8 container |
| `k_full_v4_sim` ⚠️ | full | 4 | yes | V quantised to INT4 range, int8 container |
| `k4_v_full_sim` ⚠️ | 4 | full | yes | K quantised to INT4 range, int8 container |
| `k8_v_full` | 8 | full | **no** | Real INT8 K, full-precision V |
| `k_full_v8` | full | 8 | **no** | Full-precision K, real INT8 V |

**Naming rule:** `_sim` is present only when a sub-INT8 simulated width is involved.
`k8_v_full` and `k_full_v8` carry no `_sim` suffix because they use only real INT8
and full precision — no simulated storage.

All seven compressors pass the ExactKV correctness gate: `exactkv_failures == 0`
across all tested prompts and draft lengths.

### Phase D — Report schema and leaderboard surface

* **JSON/CSV** — `key_bit_width`, `value_bit_width`, `asymmetric` added to every
  report row; backward compatible (legacy reports without these fields render
  safely).
* **`average_effective_bit_width(k, v, full=32)`** — new public helper in
  `exactkv/reporting/leaderboard.py` (and exported from `exactkv/reporting`).
  Returns `(k_bits + v_bits) / 2`, treating full precision as 32 bits.
  **Metadata comparison aid only — not a real memory measurement.**
* **Leaderboard** — `render_compressor_leaderboard` and
  `render_compressor_x_draft_leaderboard` now accept optional `compressor_caps`
  and include **K bits**, **V bits**, **avg eff bits** columns.
  Full-precision sides display as `"full"`.
* **Markdown report** — new **K/V Compression Metadata** section (table + notes)
  when asymmetric compressors are present; updated "What this does not prove"
  with sub-INT8 simulation note and avg-eff-bits disclaimer.
* **CLI `list-compressors`** — now prints `key_bit_width`, `value_bit_width`,
  and `asymmetric` for every registered compressor.

---

## Experiment 003 summary

**`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`** — 612-run core-suite sweep
comparing 9 compressors (symmetric and asymmetric) × 2 draft lengths × 34 prompts.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Prompt suite | `core.jsonl` (34 prompts) |
| Compressors | `int8`, `int4_sim`, `k8_v4_sim`, `k8_v2_sim`, `k4_v8_sim`, `k_full_v4_sim`, `k4_v_full_sim`, `k8_v_full`, `k_full_v8` |
| Draft lengths | 4, 8 |
| Max new tokens | 24 |
| Total runs | **612** |
| **ExactKV failures** | **0** |
| Lossy divergences | 386 (expected) |
| Mean accept rate | 0.739 |

Headline acceptance results:

| Compressor | K bits | V bits | Accept rate | ExactKV failures |
|---|---|---|---|---|
| `k_full_v8` | full | 8 | **0.988** | **0** |
| `k8_v_full` | 8 | full | 0.953 | **0** |
| `int8` | 8 | 8 | 0.953 | **0** |
| `k_full_v4_sim` ⚠️ | full | 4 | 0.890 | **0** |
| `k8_v4_sim` ⚠️ | 8 | 4 | 0.858 | **0** |
| `k4_v8_sim` ⚠️ | 4 | 8 | 0.562 | **0** |
| `k4_v_full_sim` ⚠️ | 4 | full | 0.561 | **0** |
| `int4_sim` ⚠️ | 4 | 4 | 0.553 | **0** |
| `k8_v2_sim` ⚠️ | 8 | 2 | 0.330 | **0** |

See [`docs/EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md`](EXPERIMENT_003_ASYMMETRIC_KV_SWEEP.md)
for the full acceptance leaderboard, K/V metadata table, histograms, and
divergence examples.

---

## Supported compressors in v0.4.0

### Symmetric (V1–V3)

| Name | Type | K bits | V bits | Simulated | Supports real bytes claim |
|---|---|---|---|---|---|
| `noop` | identity | full | full | no | yes |
| `int8` | quantization | 8 | 8 | no | yes |
| `int4_sim` ⚠️ | quantization | 4 | 4 | **yes** | **no** |
| `debug_noise` | noise injection | full | full | no | yes |

### Asymmetric (V4 new)

| Name | K bits | V bits | Simulated | Supports real bytes claim |
|---|---|---|---|---|
| `k8_v4_sim` ⚠️ | 8 | 4 | **yes** | **no** |
| `k8_v2_sim` ⚠️ | 8 | 2 | **yes** | **no** |
| `k4_v8_sim` ⚠️ | 4 | 8 | **yes** | **no** |
| `k_full_v4_sim` ⚠️ | full | 4 | **yes** | **no** |
| `k4_v_full_sim` ⚠️ | 4 | full | **yes** | **no** |
| `k8_v_full` | 8 | full | no | yes |
| `k_full_v8` | full | 8 | no | yes |

> ⚠️ Simulated compressors store sub-INT8 values in `torch.int8` containers
> with no real bit-packing. Do not interpret their `compressed_kv_bytes` as
> evidence of real memory savings.

---

## CLI commands in v0.4.0

| Command | Description |
|---|---|
| `list-compressors` | Print all registered compressors with capabilities (now includes K/V widths, asymmetric) |
| `bench` | Single-compressor benchmark over a prompt suite |
| `sweep` | Multi-compressor × multi-draft-length sweep |
| `analyze` | Analyse existing JSON report (acceptance tables, failure report) |
| `report` | Render existing JSON report to Markdown |

---

## What remains out of scope in v0.4.0

* **No real compressor backends.** KIVI, kvpress, TurboQuant, KVQuant, SnapKV,
  and similar remain V5 candidates.
* **No real INT4 or INT2 bit-packing.** All `*_sim` compressors are simulated;
  `supports_real_bytes_claim=False`.
* **No performance, throughput, latency, or speedup metrics.** ExactKV measures
  correctness and acceptance behaviour only.
* **No workspace-aware memory accounting.** Deferred to V5.
* **No production-readiness claim.** ExactKV runs with locally cached model weights
  under a research/experimental framework.
* **No serving stack.** No vLLM, LMCache, Triton, CUDA kernels, CPU offload.
* **No parallel verification.** Sequential draft-verify-commit loop only.
* **No bonus-token acceptance.** Disabled; still disabled.
* **No images or plots.** Markdown reports use text tables only.
* **No model-family compatibility guarantee.** Targets `Qwen/Qwen2.5-0.5B` on
  transformers 5.8.x.

---

## Known limitations

* **CPU sweep runtime.** 612-run Experiment 003 sweep takes ~44 min on CPU.
  Reduce `--max-new-tokens` or use `--suite smoke` for quick iteration.
* **Sub-INT8 simulation.** `*_sim` compressors store values in `int8` containers.
  Memory figures reflect `int8` storage, not real packed savings.
* **Average effective bit width** counts full precision as 32 bits regardless of
  model dtype (`bfloat16` is also 32 elements/byte in PyTorch float). This is
  intentional — it is a metadata comparison aid, not a real memory estimate.
* **`DynamicCache` brittleness.** Cache utilities target transformers 5.8.x
  internal structure. May break across versions.
* **Sequential verification only.** Single-token draft-verify loop; no parallel
  speculative verification.
* **Text truncation in examples.** Rendered text excerpts are truncated to 200
  characters.

---

## Recommended next direction for V5

1. **Workspace-aware memory accounting.** Extend `MemorySummary` to report
   `stored_kv_bytes`, `materialized_working_kv_bytes`, `metadata_bytes`, and
   `temporary_workspace_bytes`. Crucial for honest per-token memory comparisons.
   See [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](FUTURE_RESEARCH_ASYMMETRIC_KV.md) §4.

2. **Real backend adapter planning.** Design a `BackendAdapter` interface so
   real quantisation backends (bitsandbytes, GPTQ, KIVI, kvpress) can be plugged
   into the existing `KVCompressor` protocol. This is a design and interface
   exercise, not a correctness exercise.

3. **KIVI or kvpress adapter evaluation.** Once a `BackendAdapter` interface is
   stable, evaluate one real backend under ExactKV to characterise real
   acceptance rates and real memory savings — the first legitimate performance
   signal in the project.

4. **Attention-aware divergence analysis.** Correlate first-divergence position
   with prompt attention entropy. The Experiment 003 divergence data provides
   a rich starting point.

5. **Parallel verification.** Single-pass speculative verification for more
   realistic acceptance-rate measurement.

> **V5 must not add speed claims** unless they are carefully measured with
> real hardware, real bit-packing, and explicit disclaimers about experimental
> conditions. The correctness-first principle remains.

---

## Attribution

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is
a compressor-agnostic, Hugging Face-first implementation, a structured benchmark
harness, and a framework for evaluating compressors by acceptance behaviour under
full-KV verification. V4 extends that framework with asymmetric K/V compression
experiments — simulated compressors, K-only/V-only ablations, acceptance-rate
comparisons, K/V metadata reporting, and asymmetric leaderboard tables — without
adding any performance claims.
