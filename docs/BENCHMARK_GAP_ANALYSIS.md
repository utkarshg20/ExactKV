# Benchmark Scores Can Stay Green While KV Drift Happens

**Phase 10I — benchmark-gap comparison matrix**

> Outcome benchmarks and ExactKV answer **different questions**.  
> Outcome benchmarks can tell you whether the answer scored well. ExactKV tells you whether KV compression changed the model's path before the answer looked fine.

Companion: [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md) · [`EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md`](EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md) · [`leaderboard.md`](leaderboard.md)

**Not a launch claim.** No new experiments in Phase 10I. External benchmark leaderboard numbers are **not** cited here as ExactKV results.

---

## 1. Core distinction

| Approach | What it measures | What it can miss |
|---|---|---|
| **LongBench / QA-style outcome** | Final answer quality vs a reference or rubric | Whether compressed KV changed the **token path** before the answer looked fine |
| **RULER / needle retrieval** | Whether the model retrieved the right span | Whether **compressed KV** altered the generation route to that span |
| **Perplexity / log-likelihood** | Distribution-level fit on held-out text | **Exact greedy generation identity** step-by-step under compression |
| **Compression-ratio / throughput reports** | Storage, bits, or vendor throughput claims | **Behavioral drift** — draft tokens diverging from full-KV reference |
| **ExactKV** | Full-KV **behavioral equivalence** and verifier repair on tested panels | Task quality, speed, active memory savings, production serving |

ExactKV is **complementary to outcome benchmarks** — not a replacement.

---

## 2. Existing ExactKV examples

### A. LongBench-style drift demo (Exp 037)

| Field | Value |
|---|---|
| Trace | `lb_md_001` × `int4_sim` |
| Benchmark-style outcome | Answer still contains target entity (outcome heuristic **green**) |
| Hidden issue | Lossy path diverged (`billing` → `answer` on token path) |
| ExactKV signal | First divergence detected; verifier rejected lossy draft; full-KV path committed |
| `exactkv_failures` | **0** |
| Caveat | LongBench-**style** only — not official LongBench evaluation |

### B. Shard external drafter (Exp 041)

| Field | Value |
|---|---|
| Method | Shard external-drafter probe (Mode B) |
| Setting | `stream_bits=4`, `max_new_tokens=128`, 32-prompt panel |
| Benchmark-style outcome | Not the focus — would require a separate task scorer |
| ExactKV signal | **18/32 draft divergences (56.25%)** under combined stress |
| `exactkv_failures` | **0** |
| Caveat | External drafter only · Llama-only · **not** integrated · **not** default registry · no speed/memory/serving claim |

### C. SpectralQuant experimental adapter (Exp 045)

| Field | Value |
|---|---|
| Method | `spectralquant_experimental` — factory-only **materializing** adapter |
| Setting | 6-prompt calibration · 12-prompt restricted panel · Qwen2.5-0.5B |
| Final exact match | **12/12** |
| Draft divergence | **11/12** prompts (verifier corrected) |
| Mean acceptance | **0.481** |
| `exactkv_failures` | **0** |
| Caveat | Small panel · materializing (no active memory savings) · **not** default registry · no speed/memory/serving claim |

### D. Core ExactKV V13 grids

| Field | Value |
|---|---|
| Scope | V10 full suites (Exp 012, 015, 016), span grid (Exp 029), Llama small suite (Exp 033), repair policies (Exp 025) |
| Signal | Tested panels preserve **full-KV greedy output** when `exactkv_failures == 0` |
| `exactkv_failures` | **0** on reported published grids |
| Caveat | **Tested panels only** — not universal proof for all compressors, models, or prompts |

---

## 3. What a normal benchmark might miss

Conceptual example (not a fetched external score):

```text
Final answer:     correct
Benchmark score:  pass
Lossy draft path: diverged
ExactKV:          caught and repaired
exactkv_failures: 0
```

The outcome layer can look fine while the compressed-KV draft path already diverged from full-KV greedy decoding. ExactKV surfaces that gap on tested panels.

---

## 4. Comparison matrix (ExactKV evidence only)

| Example | Outcome / task layer | Path / equivalence layer (ExactKV) | exactkv_failures |
|---|---|---|---|
| Exp 037 LongBench-style | Outcome heuristic green | Token path drifted; repaired | 0 |
| Exp 041 Shard combined | Not scored here | 56.25% draft divergence | 0 |
| Exp 045 SpectralQuant | 12/12 final exact match | 11/12 draft divergence; accept 0.481 | 0 |
| Exp 029 span grid | N/A (exactness grid) | Sequential ≡ span exactness | 0 |
| Exp 012 V10 suite | Acceptance varies by compressor | Full-KV output preserved on panel | 0 |

See also: [`docs/assets/benchmark_gap_matrix.md`](assets/benchmark_gap_matrix.md) (compact table artifact).

---

## 5. What this does not prove

- **Does not** mean LongBench, RULER, or other outcome benchmarks are bad or flawed.
- **Does not** mean ExactKV **replaces** task or outcome benchmarks.
- **Does not** prove speedup, throughput, or latency improvement.
- **Does not** prove active GPU memory savings or production serving readiness.
- **Does not** prove model accuracy improvement from compression.
- **Does not** prove all compressors drift on all prompts — only what was measured on cited panels.
- **Does not** prove all prompts are safe under lossy KV without testing.

---

## 6. Recommended public wording

Use when describing ExactKV next to benchmark-style evaluation:

- **ExactKV is complementary to outcome benchmarks.**
- **Outcome benchmarks measure whether the final answer scored well.**
- **ExactKV measures whether KV compression changed what the same model would have generated** under full-KV greedy reference on tested panels.

Required distinction:

> Outcome benchmarks can tell you whether the answer scored well. ExactKV tells you whether KV compression changed the model's path before the answer looked fine.

---

## 7. Optional public card copy

Short lines for README, launch visuals, or thread cards (use with caveats above):

1. **The answer can look right while the KV path drifted.**
2. **ExactKV catches the drift outcome scores hide.**
3. **KV compression should not be trusted. It should be crash-tested.**
4. **Green task scores ≠ identical generation path.**
5. **Complementary, not competitive — crash-test the cache, then score the task.**

---

## 8. Related experiments

| Exp | Doc |
|---|---|
| 037 | [`EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md`](EXPERIMENT_037_LONGBENCH_STYLE_DRIFT_DEMO.md) |
| 041 | [`EXPERIMENT_041_SHARD_COMBINED_STRESS.md`](EXPERIMENT_041_SHARD_COMBINED_STRESS.md) |
| 045 | [`EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md`](EXPERIMENT_045_SPECTRALQUANT_RESTRICTED_PANEL.md) |
| 029 | [`EXPERIMENT_029_SPAN_VERIFICATION_GRID.md`](EXPERIMENT_029_SPAN_VERIFICATION_GRID.md) |
| 10H | [`PHASE_10_EXTERNAL_METHODS_SUMMARY.md`](PHASE_10_EXTERNAL_METHODS_SUMMARY.md) |

---

## 9. Claims boundary

- No speedup, active memory savings, production serving, or model accuracy improvement claim.
- Shard and SpectralQuant rows are **restricted external-method probes** — not full-panel compressor rankings.
- External Shard/SpectralQuant paper or benchmark leaderboard numbers are **not** ExactKV results.
