# ExactKV v0.2.0 — Release Notes

**Release date:** 2026-06-07
**Tag:** `v0.2.0`
**Previous release:** `v0.1.0-phase1` (V1 correctness prototype)

---

## What V1 proved (v0.1.0-phase1)

V1 established the core correctness invariant and benchmark plumbing:

1. **Full-KV baseline gate.** The custom `generate_full_greedy` function produces
   exactly the same token IDs as `model.generate(do_sample=False, num_beams=1)`.
2. **Cache alignment invariant.** After every draft-verify-commit round, the
   authoritative `FullKVState` and the `CompressedKVState` represent the same
   logical committed token prefix. Lengths are asserted at every round boundary.
3. **Accept/reject/correction bookkeeping.** Accepted tokens match greedily from
   the left; the first mismatch commits the verifier's correction token; all
   subsequent draft positions are rejected.
4. **Three-mode benchmark runner.** Every prompt runs under `full`, `lossy`, and
   `exactkv` modes in a single call and produces a structured JSON report.
5. **Primary correctness criterion.**
   ```
   exactkv_output_ids == full_output_ids
   ```
   This remains the hard, non-negotiable gate for every compressor in every
   version.

**V1 compressors:** `noop` (identity), `int8` (per-tensor symmetric INT8),
`debug_noise` (intentional perturbation, forces rejection).

---

## What V2 adds (v0.2.0)

V2 turns the correctness prototype into a structured experimental framework for
compressor comparison and acceptance/failure analysis.

### Phase A — Behavior-preserving refactors

- **Shared prefill helper** (`exactkv/runtime/prefill.py`).  The repeated
  encode → forward → `FullKVState` pattern is now centralised in
  `prefill_to_full_state(runtime, prompt)`.  All callers (`generation.py`,
  `exactkv_generator.py`, `metrics/memory.py`) are refactored to use it.
- **Compressor registry** (`exactkv/compressors/registry.py`).  Compressors are
  now looked up by name via `get_compressor(name)` and `list_compressors()`.
  The runner no longer hardcodes compressor construction.
- **Compressor capabilities** (`CompressorCapabilities` in
  `exactkv/compressors/base.py`).  Every registered compressor exposes a
  `capabilities` attribute declaring `is_simulated`, `supports_real_bytes_claim`,
  `supports_quantization`, `supports_token_dropping`, `compressor_type`, and
  `notes`.
- **Unified config** (`BenchmarkConfig` in `exactkv/config.py`).  A higher-level
  configuration object for multi-compressor sweeps, compatible with
  `ExactKVConfig`.
- **Reports directory convention.**  `reports/` is gitignored (generated outputs);
  `reports/.gitkeep` is tracked to preserve the directory.

### Phase B — Simulated INT4 compressor

- **`int4_sim` compressor** (`exactkv/compressors/int4_sim.py`).  Per-tensor
  symmetric quantisation into the signed 4-bit range `[-8, 7]`; stored in
  `torch.int8` (1 byte/element, not real 4-bit packed).
- `capabilities.is_simulated = True`; `supports_real_bytes_claim = False`.
- INT4-sim ExactKV gate: `int4_sim` output IDs match `generate_full_greedy`
  across ≥2 prompts × ≥2 draft lengths.

### Phase C — Reporting layer

- **`exactkv/benchmarks/reports.py`** — stable JSON and CSV reports.
  - `build_run_manifest`: provenance record (model, seed, git commit,
    transformers/torch versions, timestamp).  No timing fields.
  - `write_json_report` / `load_json_report`: human-readable, lossless JSON.
    Enriches results with `compressor_capabilities`, `is_simulated`,
    `supports_real_bytes_claim`, and `memory_claim_note`.
  - `write_csv_report` / `flatten_report_to_rows`: one CSV row per prompt result.
    Includes all compressor metadata columns.
  - Both writers auto-create parent directories.
  - Forbidden performance fields (`tokens_per_second`, `throughput`, `latency`,
    `speedup`, `runtime_seconds`) raise `ValueError` if present.

### Phase D — Sweep orchestration

- **`exactkv/benchmarks/sweeps.py`** — `run_sweep(runtime, prompts,
  compressor_names, draft_lengths, max_new_tokens)`.
  - Iterates the full prompt × compressor × draft_len grid.
  - Single model load across the entire sweep.
  - Returns a sweep report compatible with `reports.py`: `manifest`, `results`
    (flat list), `aggregate` (total_runs, exactkv_failures, lossy_divergence_count,
    mean_acceptance_rate, total_drafted/accepted/rejected/corrections).

### Phase E — Analysis layer

- **`exactkv/analysis/acceptance_tables.py`** — `build_acceptance_table`,
  `group_acceptance_by_compressor`, `group_acceptance_by_draft_len`,
  `group_acceptance_by_category`, `write_acceptance_table_csv`.
- **`exactkv/analysis/mismatch.py`** — `first_lossy_divergences`,
  `mismatch_position_summary`, `rejection_position_summary`.
- **`exactkv/analysis/failure_report.py`** — `build_failure_report`,
  `list_exactkv_failures`, `list_lossy_divergences`, `write_failure_report_json`.
- All analysis functions operate on existing report dicts (no model re-run).
- **Key distinction explicitly enforced:** lossy divergence is expected and
  is *not* an ExactKV failure.  ExactKV failure (output ≠ full greedy) is a
  correctness bug and must always be zero.

### Phase F — CLI

- **`exactkv/cli.py`** and **`exactkv/__main__.py`** — `python -m exactkv`.
- **`list-compressors`**: prints all registered compressors with their
  capabilities.
- **`bench`**: runs a single-compressor benchmark over a prompt suite; writes
  JSON and CSV; prints a no-timing summary.
- **`sweep`**: runs a multi-compressor × multi-draft-length sweep; writes JSON
  and CSV; prints the aggregate summary.
- **`analyze`**: loads an existing JSON report; writes acceptance CSV and/or
  failure JSON; exits 0 if `exactkv_failures == 0`, 1 otherwise.
- Compressor names are validated before model loading (fast error path).
- `--suite-file PATH` overrides `--suite NAME` to accept any custom JSONL file.
- No timing, throughput, latency, or speedup output from any CLI subcommand.

---

## Compressors supported in v0.2.0

| Name | Type | Simulated | Real bytes claim |
|---|---|---|---|
| `noop` | identity | No | No |
| `int8` | quantization (per-tensor symmetric) | No | Yes |
| `int4_sim` | quantization (simulated INT4) | **Yes** | **No** |
| `debug_noise` | debug (forced rejection) | Yes | No |

> **`int4_sim` disclaimer:** Values are quantised into `[-8, 7]` but stored in
> `torch.int8` (1 byte per element). This is **not** real 4-bit bit-packing.
> `compressed_kv_bytes` for `int4_sim` reflects actual `int8` storage, not a
> theoretical 2× memory reduction. Do not cite `int4_sim` memory numbers as
> evidence of real INT4 savings.

---

## CLI commands

```
python -m exactkv list-compressors
python -m exactkv bench   --model MODEL --suite smoke --compressor int8 ...
python -m exactkv sweep   --model MODEL --suite smoke --compressors noop,int8 --draft-lengths 4,8 ...
python -m exactkv analyze --report reports/sweep.json --acceptance-csv ... --failure-json ...
```

---

## Test suite

**344 tests across 22 test files.**  All pass with `Qwen/Qwen2.5-0.5B` in
`float32` on CPU with `TRANSFORMERS_OFFLINE=1`.

| Gate | Test file |
|---|---|
| Full baseline | `test_full_generation.py` |
| NoOp ExactKV | `test_noop_exactkv.py` |
| INT8 ExactKV | `test_int8_exactkv.py` |
| INT4-sim ExactKV | `test_int4_sim_exactkv.py` |
| DebugNoise rejection | `test_debug_noise_exactkv.py` |
| Reporting (JSON/CSV) | `test_reports.py` |
| Sweep | `test_sweeps.py` |
| Analysis (acceptance, mismatch, failure) | `test_analysis_*.py` |
| CLI | `test_cli.py` |

---

## What is still out of scope in v0.2.0

- **No real compressor backends.** KIVI, kvpress, TurboQuant, KVQuant, SnapKV
  and similar are not integrated.  The registry boundary is prepared for them.
- **No real INT4 bit-packing.**  `int4_sim` is a simulation only.
- **No performance, throughput, or speedup metrics.**  ExactKV measures
  correctness and acceptance behaviour.  It does not measure tokens/sec, latency,
  or memory bandwidth.
- **No plots or dashboards.**  V2 emits CSV that can be plotted externally.
- **No production serving stack.**  No vLLM, LMCache, Triton, CUDA kernels, CPU
  offload, or async transfer.
- **No batching, no sampling, no parallel verification, no bonus-token acceptance.**
- **No broad model-family compatibility guarantee.**  Tested exclusively against
  `Qwen/Qwen2.5-0.5B` on transformers 5.8.1.

---

## Known limitations

1. **`DynamicCache` brittleness.**  ExactKV reconstructs `DynamicCache` via
   direct attribute injection into `DynamicLayer` objects.  This targets
   transformers 5.8.1 and may break across versions.  See
   `docs/V1_SCOPE_STATEMENT.md § Known V1 limitations`.

2. **Sequential verification only.**  The verification engine runs one draft
   token per forward pass.  This is correct but slower than the single-pass
   parallel verify from VeriCache.  Parallel verification is deferred.

3. **`int4_sim` memory numbers are conservative.**  The `compressed_kv_bytes`
   field reflects actual `int8` container storage (same number of bytes as INT8).
   A real packed INT4 implementation would use half as many bytes.

4. **Prompt suite is small.**  `smoke.jsonl` has 16 prompts across 6 categories.
   Larger `core` and `stress` suites are planned for a future release.

---

## Next planned version direction

**V3 (planned):**
- Larger and more diverse prompt suites (`core.jsonl`, `stress.jsonl`).
- CSV plotting utilities (acceptance rate curves, divergence histograms).
- Broader model family support (Llama-3, Mistral, Phi-3).
- Version-pinned `DynamicCache` compatibility layer.

**V4 (planned):**
- Real compressor backend integration (KIVI, kvpress, or TurboQuant) behind the
  existing registry boundary.
- Real INT4 bit-packing with honest memory accounting.

**V5 (planned):**
- Parallel (single-pass) verification matching VeriCache's algorithm.
- Speculative bonus-token acceptance.
- Serving-stack integration experiments.

---

## Attribution

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm.  ExactKV's contribution
is a compressor-agnostic, Hugging Face-first implementation, a structured
benchmark harness, and a framework for evaluating compressors by acceptance
behaviour under full-KV verification.
