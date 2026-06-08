# ExactKV V3 Scope Statement

## What V3 is

V3 turns the V2 benchmark **framework** into a benchmark **presentation and
storytelling layer**.

V2 made ExactKV extensible and analyzable: a compressor registry, a CLI, JSON/CSV
reports, sweep orchestration, and an analysis layer. But the outputs are still
machine-oriented (raw JSON, flat CSV, dict-returning analysis functions). V3 makes
the project **publicly legible**: it renders existing reports into human-readable
markdown, acceptance leaderboards, concrete lossy-vs-ExactKV divergence examples,
and text-based mismatch histograms — plus stronger prompt suites and a second
documented experiment.

V3 is still **correctness-first and performance-silent**. It presents *acceptance
and exactness behaviour*, never throughput, latency, or speedup. V3 adds **no new
generation logic** — it only reads, computes over, and renders artifacts that V2
already produces.

## Relationship to the roadmap

`docs/ROADMAP.md` originally scoped V3 as "Benchmark suite" including throughput
metrics and PNG plotting. Those non-performance pieces (benchmark runner, prompt
suites, JSON/CSV, acceptance metrics, sweeps, analysis) were already **pulled
forward into V2**. This V3 is re-scoped as the **presentation layer** for those
existing artifacts. It explicitly **excludes** the roadmap's "throughput metrics"
and treats PNG plotting as optional/deferred — text tables are the V3 deliverable
so the "no speedup claims" and "no required images" rules stay clean.

---

## 1. What V3 should add

### 1.1 Stronger prompt suites
Four curated JSONL suites under `benchmarks/prompts/`, each following the existing
schema (`prompt_id`, `category`, `prompt`):

- **`core.jsonl`** — ~30–40 prompts, broad coverage across all categories. The
  "default real" suite for documented experiments (larger than `smoke`, still
  CPU-tractable).
- **`structured.jsonl`** — JSON / key-value / tool-call-like structured-output
  prompts. Tests acceptance behaviour on highly-templated output.
- **`code.jsonl`** — code-generation and code-completion prompts across a few
  languages. Tests acceptance on syntactic, low-entropy continuations.
- **`stress.jsonl`** — longer, harder, higher-entropy prompts (long context setup,
  multi-step reasoning prefixes) designed to *drive down* acceptance and surface
  more lossy divergence. Lengths kept moderate to remain CPU-runnable.

`smoke.jsonl` stays the fast CI suite and is unchanged.

### 1.2 Markdown report generator
A renderer that converts an existing JSON report (single-bench **or** sweep) into a
docs-ready markdown document, with no model re-run:

- Run-context header (model, suite, compressors, draft lengths, max_new_tokens,
  seed, dtype, device) drawn from the report manifest.
- Correctness summary (ExactKV failures, status pass/fail).
- Acceptance leaderboard tables.
- Lossy-divergence example blocks.
- Mismatch histograms.
- Explicit "what this proves / does not prove" and `int4_sim` simulation
  disclaimer boilerplate.

### 1.3 Acceptance leaderboard
Markdown tables ranking compressors by acceptance behaviour:

- **Compressor × draft-length** acceptance grid (rate per cell).
- Per-compressor roll-up (rate, drafted, accepted, rejected, corrections,
  lossy divergences, exactkv failures).
- Per-draft-length roll-up.
- Sorted "leaderboard" view (highest acceptance first), with `noop` clearly
  labelled as the identity baseline so it is not mistaken for a "winner".

This is a **rendering** layer on top of the existing
`exactkv/analysis/acceptance_tables.py` computations.

### 1.4 Lossy divergence examples
For a bounded number of diverging runs, render a side-by-side block:

```
Prompt:  <prompt text>
Full   : <full-KV greedy output text>
Lossy  : <compressed-only output text>        ← diverges at token N
ExactKV: <verified output text>               ← matches Full exactly
```

Each block states explicitly: **lossy divergence is expected; ExactKV failure is
the real failure.** Examples are built entirely from the `output_text` / token IDs
already stored in the report — no re-running the model.

### 1.5 Mismatch histograms (text tables)
Text-based distributions over existing trace/result data:

- **Accepted-length histogram** — distribution of accepted-run lengths per
  verification round, bucketed.
- **First-divergence bucket table** — distribution of `lossy_first_divergence_idx`
  into token-position buckets (e.g. `0–2`, `3–7`, `8–15`, `16+`, `none`).
- Optionally grouped by compressor and/or category.

No images required. PNG plotting is **not** in V3 scope.

### 1.6 README result polish
- A compact benchmark table near the top.
- A link to a generated markdown report and to the experiment docs.
- Keep all existing honesty disclaimers (`int4_sim` simulated, no speedup).

### 1.7 Second documented experiment
- **`docs/EXPERIMENT_002_CORE_SWEEP.md`** — same structure as
  `EXPERIMENT_001_SMOKE_SWEEP.md`, run over the larger `core.jsonl` suite, with the
  markdown generator used to produce its tables.

### 1.8 Optional CLI surface (presentation only)
A `report` subcommand that wraps the markdown generator:

```
python -m exactkv report --report reports/sweep.json --md-out docs/generated/sweep.md
```

No new generation flags; reads an existing report and writes markdown.

---

## 2. What V3 explicitly should NOT add

1. **No real compressor backends.** KIVI, kvpress, TurboQuant, KVQuant, SnapKV,
   and similar remain out of scope. V3 only presents results for the existing
   `noop` / `int8` / `int4_sim` / `debug_noise` compressors. Real backends are V4.
2. **No real INT4 bit-packing / no real memory-savings claim.** `int4_sim` stays
   simulated and flagged; every rendered memory figure keeps `is_simulated` /
   `supports_real_bytes_claim` / `memory_claim_note`.
3. **No performance, throughput, latency, or speedup metrics or language** — in
   any code, report, table, or doc. The no-performance-field audit is mandatory in
   every V3 phase.
4. **No production-readiness claims.**
5. **No new generation logic.** No sampling, no batching, no parallel verification,
   no bonus-token acceptance. The draft-verify-commit loop is untouched.
6. **No serving stack.** No vLLM, LMCache, Triton, CUDA kernels, CPU offload.
7. **No required plotting / images.** PNG/plot generation is deferred; V3 ships
   text tables only.
8. **No model re-run inside the presentation layer.** The markdown generator,
   leaderboard, examples, and histograms operate **only** on existing report dicts.
9. **No broad model-family compatibility guarantee.** Still targets
   `Qwen/Qwen2.5-0.5B` on the tested transformers version.

---

## 3. Final V3 file tree

New and changed paths only; everything else from V2 stays as-is.

```
exactkv/
├── cli.py                         # CHANGED: add `report` subcommand
├── analysis/
│   ├── acceptance_tables.py       # (V2, reused by leaderboard renderer)
│   ├── mismatch.py                # (V2, reused by examples/histograms)
│   ├── failure_report.py          # (V2, reused)
│   └── histograms.py              # NEW: accepted_length_histogram,
│   │                              #      first_divergence_buckets (compute only)
│   └── examples.py                # NEW: select_divergence_examples (compute only,
│                                  #      pulls full/lossy/exactkv text from report)
└── reporting/                     # NEW PACKAGE (render-only, no model, no compute-heavy logic)
    ├── __init__.py                # public re-exports
    ├── markdown.py                # build_markdown_report(report)->str,
    │                              # write_markdown_report(report, path)
    ├── leaderboard.py             # render_acceptance_leaderboard(report)->str (markdown)
    ├── examples.py                # render_divergence_examples(report)->str (markdown)
    └── histograms.py              # render_histograms(report)->str (markdown text tables)

benchmarks/
└── prompts/
    ├── smoke.jsonl                # unchanged (fast CI suite)
    ├── core.jsonl                 # NEW: ~30–40 broad prompts
    ├── structured.jsonl           # NEW: JSON / structured-output prompts
    ├── code.jsonl                 # NEW: code prompts
    └── stress.jsonl               # NEW: longer/harder prompts

docs/
├── V3_SCOPE_STATEMENT.md          # this file
├── EXPERIMENT_002_CORE_SWEEP.md   # NEW: core-suite experiment write-up
└── generated/                     # NEW: home for generated markdown reports
    └── .gitkeep                   # (generated .md are gitignored like reports/)

tests/
├── test_prompt_suites.py          # NEW: suite loading + schema + uniqueness
├── test_analysis_histograms.py    # NEW: histogram + bucket compute correctness
├── test_analysis_examples.py      # NEW: divergence-example selection correctness
├── test_reporting_markdown.py     # NEW: markdown generator gate
├── test_reporting_leaderboard.py  # NEW: leaderboard render gate
└── test_cli_report.py             # NEW: CLI `report` subcommand gate

reports/                           # unchanged convention (gitignored outputs)
```

**Separation of concerns (kept strict):**
- `analysis/` = **compute** structured numbers from a report dict (returns
  dicts/lists; no markdown).
- `reporting/` = **render** those numbers into markdown strings/files (no model,
  no heavy compute).
- This boundary mirrors the V2 split (`runner` produces, `reports` serializes,
  `analysis` summarizes) and keeps each layer independently testable.

---

## 4. Implementation phases

Each phase is independently shippable and must keep the full existing test suite
green plus pass the no-performance-field audit.

### Phase A — Stronger prompt suites
- Author `core.jsonl`, `structured.jsonl`, `code.jsonl`, `stress.jsonl`.
- Extend the suite loader / CLI `--suite` name resolution to recognize the new
  named suites (currently only `smoke` is a named suite; others via `--suite-file`).
- No rendering code yet.

### Phase B — Analysis compute additions
- `analysis/histograms.py`: `accepted_length_histogram`, `first_divergence_buckets`
  (return count dicts; reconcile to totals).
- `analysis/examples.py`: `select_divergence_examples` (return structured records:
  prompt, full text/ids, lossy text/ids, exactkv text/ids, first_divergence_idx,
  exactkv_matches_full flag).
- Pure functions over report dicts; no markdown, no model.

### Phase C — Reporting / rendering layer
- `reporting/leaderboard.py`, `reporting/examples.py`, `reporting/histograms.py`:
  render the Phase-B/V2-analysis outputs to markdown sections.
- `reporting/markdown.py`: assemble the full markdown report (header from manifest,
  correctness summary, leaderboard, examples, histograms, disclaimers).
- Handles both single-bench and sweep reports (delegates to analysis which already
  normalizes both shapes).

### Phase D — CLI, README polish, second experiment
- Add `report` subcommand to `cli.py`.
- Polish README: compact benchmark table + links to generated report and experiment
  docs.
- Run a `core.jsonl` sweep and author `docs/EXPERIMENT_002_CORE_SWEEP.md`, using the
  generator for its tables.

---

## 5. Tests and gates per phase

### Phase A gates — prompt suites
- Each new suite loads via the loader without error.
- Every entry has `prompt_id`, `category`, `prompt`; all `prompt_id` values unique
  within a suite.
- Category values are from the documented category set.
- CLI `--suite core|structured|code|stress` resolves to the right file.
- **Correctness spot-check:** a small subset of each suite runs end-to-end through
  ExactKV (`int8`) with `exactkv_failures == 0` (regression bar — proves the new
  prompts don't break the loop).

### Phase B gates — analysis compute
- **Histogram reconciliation:** histogram bucket counts sum to the total number of
  accepted-run observations; first-divergence buckets sum to total results
  (including a `none` bucket for non-diverging runs).
- **Example selection correctness:** every selected example is genuinely a lossy
  divergence (`lossy.token_exact_match == False`) and every example has
  `exactkv_matches_full == True` (since the gate is zero ExactKV failures).
- Works on both a synthetic report dict and a real sweep report.
- No forbidden performance fields in any returned structure.

### Phase C gates — rendering
- **Markdown generator gate:** given a sweep JSON, `build_markdown_report` returns a
  non-empty string containing the expected section headers (run context,
  correctness summary, leaderboard, examples, histograms, disclaimer).
- Leaderboard contains one row per compressor and a compressor × draft-length grid
  consistent with the analysis numbers.
- `int4_sim` rows/sections carry the simulation disclaimer text.
- Rendered output contains the literal "lossy divergence is expected" /
  "ExactKV failure" distinction.
- `write_markdown_report` creates parent dirs automatically (matches V2 writer
  convention) and writes valid UTF-8.
- **No-forbidden-fields audit:** rendered markdown contains none of
  `tokens_per_second`, `throughput`, `latency`, `speedup`, `runtime_seconds`
  (except inside explicit negation sentences in static boilerplate, audited the
  same way as the experiment docs).

### Phase D gates — CLI + docs
- `python -m exactkv report --report <json> --md-out <path>` exits 0 and writes a
  markdown file.
- Invalid/missing report path exits non-zero cleanly.
- README compact table renders and links resolve to real files.
- `EXPERIMENT_002_CORE_SWEEP.md` exists, reports `exactkv_failures == 0`, and passes
  the no-performance-word audit.

### Global gates (every phase)
- Full existing test suite stays green (V1 + V2 = current 344 tests).
- No-performance-field audit passes across code, reports, and docs.
- Primary correctness criterion unchanged: `exactkv_output_ids == full_output_ids`.

---

## 6. How V3 prepares for real backends (V4)

V3 is a presentation layer, so its main contribution to V4 is **making real-backend
results immediately legible and honestly labelled** the moment a backend lands:

1. **Capabilities-driven rendering.** The markdown generator, leaderboard, and
   memory tables key off the existing `CompressorCapabilities`
   (`is_simulated`, `supports_real_bytes_claim`, `compressor_type`). A V4 real
   backend that registers correct capabilities is rendered with correct honesty
   labels automatically — no renderer changes needed.
2. **Schema-stable inputs.** Rendering consumes only the locked V2 report schema
   (`manifest` / `results` / `aggregate` + per-result capabilities + memory honesty
   fields). V4 backends slot into the same schema and inherit all V3 presentation
   for free.
3. **Leaderboard framing ready for real vs simulated.** The leaderboard already
   distinguishes identity baseline (`noop`), real quantization (`int8`), and
   simulated (`int4_sim`). Adding a real INT4/KIVI/etc. row requires no new
   presentation logic, and the simulated-vs-real distinction is already visible.
4. **Divergence examples as a backend acceptance tool.** When a real backend is
   added, the same example/histogram tooling immediately characterises *where* and
   *how often* it diverges — a practical way to evaluate a new backend's acceptance
   quality before any performance work.
5. **No performance scaffolding to unwind.** Because V3 deliberately avoids timing
   fields, V4/V5 can introduce real performance measurement cleanly, in one place,
   without retrofitting honesty caveats into presentation code.

---

## 7. V2 refactors needed first

V3 is intentionally additive; the V2 schema already carries everything the
presentation layer needs (per-result `prompt`, `full/lossy/exactkv` `output_text`
and `output_ids`, `first_divergence_idx`, acceptance summary, memory honesty
fields, compressor capabilities, manifest). **No mandatory V2 refactors.**

Small, optional clean-ups (nice-to-have, not blocking):

1. **Named-suite resolution.** V2's CLI only treats `smoke` as a named suite
   (others require `--suite-file`). Phase A will generalize the name→path mapping;
   this is an extension, not a breaking refactor.
2. **Manifest on sweep reports.** `run_sweep` output currently surfaces
   `manifest`/`results`/`aggregate`; confirm the manifest is consistently present
   so the markdown header always has run context. If any path omits it, add it in
   Phase C (additive).
3. **Shared no-performance audit helper.** The forbidden-field check exists in
   `reports.py` and is duplicated ad hoc in tests/docs audits. Optionally extract a
   single reusable audit utility for reuse by the renderer tests. Cosmetic.

None of these block starting Phase A.

---

## V3 exit criteria

V3 is complete when all of the following hold:

- [x] Four new prompt suites (`core`, `structured`, `code`, `stress`) load, validate,
      and resolve as named suites; correctness spot-checks pass with zero ExactKV
      failures.
- [x] Histogram and example analysis functions reconcile counts and pass on both
      synthetic and real reports.
- [x] Markdown generator produces a complete, readable report from a sweep JSON,
      including leaderboard, divergence examples, and histograms.
- [x] Every rendered artifact preserves `int4_sim` simulation labelling and the
      "lossy divergence expected vs ExactKV failure" distinction.
- [x] `python -m exactkv report` writes a markdown report from an existing JSON.
- [x] README has a compact benchmark table and links to a generated report and the
      experiment docs.
- [x] `docs/EXPERIMENT_002_CORE_SWEEP.md` is written from a real core-suite sweep
      with `exactkv_failures == 0`.
- [x] No-performance-field audit passes across all V3 code, reports, and docs.
- [x] Full prior test suite remains green (542 tests).

---

## Future V4/V5 research candidates

The following directions are documented for future consideration but are
**not part of V3**:

* **Asymmetric K/V compression** — Keys and values play different roles in
  attention; compressing them symmetrically is likely suboptimal. Future work
  should evaluate K8/V4, K-full/V-compressed, and similar asymmetric policies
  using ExactKV's acceptance-rate metrics rather than MSE.
* **Workspace-aware memory accounting** — Stored compressed bytes do not
  capture the peak memory during decode (which includes dequantisation scratch
  buffers). Future `MemorySummary` should distinguish `stored_kv_bytes`,
  `materialized_working_kv_bytes`, `metadata_bytes`, and
  `temporary_workspace_bytes`.
* **Attention-aware divergence analysis** — Correlate first-divergence
  position with attention entropy to understand which prompts and positions
  are most sensitive to compression.

See [`docs/FUTURE_RESEARCH_ASYMMETRIC_KV.md`](FUTURE_RESEARCH_ASYMMETRIC_KV.md)
for a full writeup.

---

## Citation and novelty note

The draft-then-verify compressed-KV algorithm is from:

> VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is a
compressor-agnostic, Hugging Face-first implementation, a structured benchmark
harness, and a framework for evaluating compressors by acceptance behaviour under
full-KV verification. V3 extends that framework with a presentation and
storytelling layer — markdown reports, acceptance leaderboards, divergence
examples, mismatch histograms, and stronger prompt suites — without adding any
performance claims.
