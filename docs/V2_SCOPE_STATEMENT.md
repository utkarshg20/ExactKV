# ExactKV V2 Scope Statement

## What V2 is

V2 turns the V1 correctness prototype into a cleaner **experimental framework for
compressor comparison and acceptance/failure analysis**.

V1 proved the core draft-verify-commit loop is correct. V2 makes ExactKV
*extensible* and *analyzable*: new compressors plug in through a registry,
experiments run from a CLI, results are written as JSON and CSV, and an analysis
layer turns raw traces into acceptance tables, mismatch-position breakdowns, and
failure reports.

V2 is still **correctness-first and performance-silent**. It measures *acceptance
and exactness behavior*, never throughput or latency.

## Relationship to the roadmap

`docs/ROADMAP.md` originally split V2 (framework layer) and V3 (benchmark suite,
CLI, CSV, plots). V2 as scoped here pulls the **non-performance** V3 items forward
(CLI, CSV, sweeps, analysis tooling) because they directly serve the V2 goal of
compressor comparison. It explicitly **defers** plotting and any timing/throughput
work to a later version so the "no speedup claims" rule stays clean.

## What V2 proves

1. **Compressor extensibility.**
   A new compressor can be added by registering it and implementing the
   `KVCompressor` interface — with no changes to the verification engine, runner,
   or CLI.

2. **Simulated INT4 correctness.**
   A simulated INT4 compressor (`int4_sim`) produces ExactKV output token IDs
   that exactly equal `generate_full_greedy` output, across multiple prompts and
   draft lengths. Its lossy mode may diverge from full; ExactKV still corrects it.

3. **Reproducible reporting.**
   The benchmark runner emits JSON (full fidelity) and CSV (one row per
   prompt × compressor × draft_len) plus a run manifest (model, seed, git tag,
   timestamp). JSON round-trips losslessly.

4. **Sweep correctness.**
   A draft_len × compressor sweep over a single loaded model completes and reports
   `exactkv_failures == 0`.

5. **Analysis correctness.**
   - Acceptance counts reconcile under aggregation: `drafted == accepted + rejected`.
   - Mismatch-position analysis correctly locates first-divergence and per-round
     rejection positions.
   - Failure analysis flags zero ExactKV failures and correctly lists lossy
     divergences by prompt, category, and compressor.

6. **CLI usability.**
   `python -m exactkv bench --suite smoke` runs with locally cached model weights
   and writes valid reports.

7. **Primary correctness criterion (unchanged from V1).**
   For all prompts and all registered compressors:
   ```python
   exactkv_output_ids == full_output_ids
   ```
   This remains a hard requirement. If it fails, no other result from that run is
   meaningful.

## What V2 explicitly does NOT prove / does NOT add

1. **No real compressor backends.**
   KIVI, kvpress, TurboQuant, KVQuant, SnapKV, and similar are **not** integrated.
   V2 only *prepares* the boundary (registry + capabilities). Real backends are V4.

2. **No real INT4 memory savings.**
   INT4 in V2 is **simulated** (quant/dequant emulated; stored in fp32/int8) and
   flagged `is_simulated=True`. V2 makes no real-bytes or real-memory claim for it.

3. **No performance, throughput, or speedup metrics.**
   No tokens/sec, no latency, no wall-clock timing, no speedup language anywhere in
   V2 code or docs. Memory numbers remain byte-count *estimates*, as in V1.

4. **No plotting.**
   V2 emits CSV that *can* be plotted later, but ships no plotting code or images.
   Plots are deferred to V3.

5. **No serving stack.**
   No vLLM, LMCache, Triton, CUDA kernels, CPU offload, async transfer, or
   scheduler integration.

6. **No batching, no sampling, no parallel verification, no bonus-token acceptance.**
   These remain out of scope, unchanged from V1.

7. **No broad model-family compatibility guarantee.**
   V2 continues to target `Qwen/Qwen2.5-0.5B` on the tested transformers version.

## V2 scope boundary

| In scope | Out of scope |
|---|---|
| Compressor registry + capabilities | Real compressor backends (KIVI, kvpress, TurboQuant) |
| Simulated INT4 (`int4_sim`, flagged) | Real INT4 bit-packing / real memory savings |
| Unified config (`ExactKVConfig` + `BenchmarkConfig`) | Sampling, beam search, temperature |
| CLI (`run` / `bench` / `analyze` / `list-compressors`) | Production serving entry points |
| JSON + CSV reports + run manifest | Plots, dashboards, leaderboards |
| Draft_len × compressor sweeps | Throughput / latency / speedup metrics |
| Acceptance tables, mismatch + failure analysis | Learned acceptance predictors |
| Stronger prompt suites (`core`, `stress`) | Distributed / multi-GPU benchmarking |
| Greedy decoding, sequential verification | Parallel (single-pass) verification |

## Planned V2 additions (summary)

- **Framework:** `compressors/registry.py`, `CompressorCapabilities` in
  `compressors/base.py`, `compressors/int4_sim.py`.
- **Config:** unified `ExactKVConfig` + `BenchmarkConfig` in `config.py`.
- **Runtime refactor:** shared `runtime/prefill.py` helper (deduplicates the
  prefill → `FullKVState` logic currently repeated in `generation.py`,
  `exactkv_generator.py`, and `metrics/memory.py`).
- **Reporting:** `benchmarks/reports.py` (JSON + CSV + manifest),
  `benchmarks/sweeps.py` (grid orchestration).
- **Analysis package:** `analysis/acceptance_tables.py`, `analysis/mismatch.py`,
  `analysis/failure_report.py`.
- **CLI:** `cli.py` + `__main__.py`.
- **Prompt suites:** `benchmarks/prompts/core.jsonl`, `stress.jsonl` (smoke stays
  the fast CI suite).
- **Docs:** this file, README V2 usage section, updated `COMPRESSOR_INTERFACE.md`.

## V1 refactors required before V2 features (behavior-preserving)

All of the following must land first and must keep the V1 test suite green:

1. **Shared prefill helper** — extract `prefill_to_full_state()`; remove the
   three duplicated prefill blocks.
2. **Compressor registry** — replace the hardcoded `_make_compressor()` in the
   runner; populate the currently-empty `compressors/__init__.py`.
3. **Capabilities metadata** — add `CompressorCapabilities` and declare it on
   `noop`, `int8`, `debug_noise`.
4. **Unified config** — collapse the runner's `RunConfig` into a `BenchmarkConfig`
   derived from `ExactKVConfig` (keep a thin back-compat alias).
5. **Reports conventions** — add a gitignored `reports/` directory and a run
   manifest format.

## V2 exit criteria

All of the following must pass before V2 is considered complete:

- [ ] All V1 gates still pass (full baseline, NoOp, INT8, DebugNoise rejection,
      benchmark runner) — regression bar.
- [ ] INT4-sim ExactKV gate: `int4_sim` output_ids == full output_ids across
      ≥2 prompts × ≥2 draft lengths.
- [ ] Registry gate: every registered compressor resolves by name and runs
      end-to-end with `exactkv_failures == 0`.
- [ ] Sweep gate: a draft_len × compressor sweep completes with zero failures.
- [ ] Reporting gate: JSON re-loads losslessly; CSV schema is stable and
      one-row-per-cell.
- [ ] Analysis gate: acceptance counts reconcile; mismatch and failure reports
      are correct on synthetic and real traces.
- [ ] CLI gate: `python -m exactkv bench --suite smoke` runs with locally cached
      model weights and writes valid reports.
- [ ] No-performance-claim audit: no tokens/sec, latency, or speedup language in
      V2 code or docs.

## How V2 prepares for real compressor backends (V4)

- **Stable registry boundary:** real backends register under the same API; the
  runner and CLI never hardcode compressor names.
- **Capabilities contract:** `is_simulated`, `supports_real_bytes_claim`,
  `dtype_support` let reports correctly label real vs simulated numbers.
- **Adapter-friendly interface:** the four-method contract
  (`compress` / `materialize_for_draft` / `update_after_commit` / `stats`) is the
  only thing a backend must satisfy; external libraries are wrapped behind it.
- **Cache-compat isolation:** all transformers `DynamicCache` reconstruction stays
  inside `cache/utils.py` (already documented as brittle). Real backends produce
  and consume KV through that single layer.
- **Determinism + no-mutation invariants:** codified in
  `COMPRESSOR_INTERFACE.md` — `compress` / `materialize` must not mutate
  `FullKVState` and must be deterministic under a fixed seed. These are exactly the
  properties a real backend must satisfy to pass the ExactKV gate.
- **Report schema stability:** the JSON/CSV schema locked in V2 lets V4 backends
  slot into existing analysis (acceptance tables, mismatch, failure) with no
  downstream changes.

## Citation and novelty note

The draft-then-verify compressed-KV algorithm is from:

> VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is a
compressor-agnostic, Hugging Face-first implementation, a structured benchmark
harness, and a framework for evaluating compressors by acceptance behavior under
full-KV verification. V2 extends that framework with a compressor registry, a CLI,
structured reporting, and an acceptance/failure analysis layer.
