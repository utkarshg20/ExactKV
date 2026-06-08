# ExactKV v0.3.0 — Release Notes

**Release date:** 2026-06-08
**Tag:** `v0.3.0`
**Previous release:** `v0.2.0` (V2 experimental framework)

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

## What V3 adds (v0.3.0)

V3 turns the V2 benchmark framework into a **presentation and storytelling layer**
— making results legible to humans while remaining performance-silent.

### Phase A — Stronger prompt suites

Four curated JSONL suites under `benchmarks/prompts/`:

| Suite | Prompts | Purpose |
|---|---|---|
| `core.jsonl` | 34 | Broad real-world prompts; default for documented experiments |
| `structured.jsonl` | 28 | JSON, tables, schemas, strict formatting |
| `code.jsonl` | 30 | Code generation, completion, debugging |
| `stress.jsonl` | 25 | Longer, harder, higher-entropy prompts |

Named-suite resolution updated in `exactkv/benchmarks/prompts.py` and the CLI.
All five named suites (`smoke`, `core`, `structured`, `code`, `stress`) resolve
consistently from Python functions and the CLI `--suite` flag.

### Phase B — Analysis compute additions

* **`exactkv/analysis/histograms.py`** — `accepted_length_histogram`,
  `first_divergence_histogram`, `rejection_count_histogram`. Support flat and
  grouped-by-compressor/draft-len/category views.
* **`exactkv/analysis/examples.py`** — `extract_lossy_divergence_examples`,
  `extract_exactkv_failure_examples`, `extract_rejection_examples`. Extract
  concrete text-comparison records from existing reports; no model re-run.

### Phase C — Markdown rendering layer

New `exactkv/reporting/` package (pure renderers; no model, no heavy compute):

* **`markdown.py`** — `render_markdown_report` / `write_markdown_report`: full
  12-section Markdown document including title, manifest, correctness summary,
  acceptance leaderboards, histogram tables, lossy-divergence examples, ExactKV
  failure examples, memory honesty notes, "what this proves / does not prove",
  and required disclaimers.
* **`leaderboard.py`** — `render_compressor_leaderboard`,
  `render_draft_len_leaderboard`, `render_compressor_x_draft_leaderboard`.
* **`examples.py`** — `render_lossy_divergence_examples`,
  `render_exactkv_failure_examples`, `render_rejection_examples`.
* **`histograms.py`** — `render_accepted_length_table`,
  `render_first_divergence_table`, `render_rejection_count_table`.
  Text tables only — no images, no plots.

### Phase D — CLI `report` command and Experiment 002

* **`python -m exactkv report`** — renders any existing JSON report to Markdown;
  no model re-run, no timing output.
* **`docs/EXPERIMENT_002_CORE_SWEEP.md`** — generated from a real 204-run
  core-suite sweep (`noop × int8 × int4_sim` × draft lengths `4, 8`,
  `max_new_tokens=24`, 34 prompts from `core.jsonl`).

---

## Experiment 002 headline results

| Compressor | Runs | Accept rate | ExactKV failures |
|---|---|---|---|
| `noop` | 68 | **1.000** | **0** |
| `int8` | 68 | **0.951** | **0** |
| `int4_sim` ⚠️ | 68 | 0.553 | **0** |

> Total: 204 runs, 86 lossy divergences (expected), **0 ExactKV failures**.
> See [`docs/EXPERIMENT_002_CORE_SWEEP.md`](EXPERIMENT_002_CORE_SWEEP.md).

---

## Supported compressors in v0.3.0

| Name | Type | Simulated | Supports real bytes claim |
|---|---|---|---|
| `noop` | identity | no | yes |
| `int8` | quantization | no | yes |
| `int4_sim` | quantization | **yes** | **no** |
| `debug_noise` | noise injection | no | yes |

> ⚠️ `int4_sim` stores values in `torch.int8` containers. Its memory figures
> reflect `int8` storage, not real packed 4-bit savings.

---

## CLI commands in v0.3.0

| Command | Description |
|---|---|
| `list-compressors` | Print all registered compressors with capabilities |
| `bench` | Single-compressor benchmark over a prompt suite |
| `sweep` | Multi-compressor × multi-draft-length sweep |
| `analyze` | Analyse existing JSON report (acceptance tables, failure report) |
| `report` | (**V3 new**) Render existing JSON report to Markdown |

---

## What is still out of scope in v0.3.0

* **No real compressor backends.** KIVI, kvpress, TurboQuant, KVQuant, SnapKV,
  and similar remain V4 candidates.
* **No real INT4 bit-packing.** `int4_sim` is simulated; `supports_real_bytes_claim=False`.
* **No performance, throughput, latency, or speedup metrics.** ExactKV measures
  correctness and acceptance behaviour only.
* **No production-readiness claim.** ExactKV runs with locally cached model weights
  under a research/experimental framework.
* **No serving stack.** No vLLM, LMCache, Triton, CUDA kernels, CPU offload.
* **No parallel verification.** Sequential draft-verify-commit loop only.
* **No bonus-token acceptance.** Disabled in V1; still disabled.
* **No images or plots.** Markdown reports use text tables only.
* **No model-family compatibility guarantee.** Targets `Qwen/Qwen2.5-0.5B` on
  transformers 5.8.1.

---

## Known limitations

* **Core suite runtime.** 204-run core sweep takes ~18 min on CPU. Reduce
  `--max-new-tokens` (e.g. 12) or use `--suite smoke` for quick iteration.
* **`DynamicCache` brittleness.** Cache utilities target transformers 5.8.1
  internal structure (`DynamicLayer.keys/values`). May break across versions.
  See `docs/V1_SCOPE_STATEMENT.md`.
* **Sequential verification only.** The draft-verify loop processes one token at
  a time. Parallel (single-pass) verification is deferred to a future version.
* **Text truncation in examples.** Rendered text excerpts are truncated to 200
  characters; long outputs are clipped.

---

## Recommended next direction for V4

1. **Asymmetric K/V compressor.** Implement `AsymmetricQuantCompressor(k_bits, v_bits)`.
   See [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](FUTURE_RESEARCH_ASYMMETRIC_KV.md).
2. **Real compressor backend.** Integrate one real quantisation backend (e.g. a
   bitsandbytes INT4 or GPTQ-style quantiser) using the existing `KVCompressor`
   protocol.
3. **Workspace-aware memory accounting.** Extend `MemorySummary` to report
   `stored_kv_bytes`, `materialized_working_kv_bytes`, `metadata_bytes`, and
   `temporary_workspace_bytes`.
4. **Attention-aware divergence analysis.** Correlate first-divergence position
   with attention entropy to understand which positions are compression-sensitive.
5. **Parallel verification.** Implement single-pass speculative verification for
   more realistic acceptance-rate measurement.

---

## Attribution

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is
a compressor-agnostic, Hugging Face-first implementation, a structured benchmark
harness, and a presentation layer for evaluating compressors by acceptance
behaviour under full-KV verification.
