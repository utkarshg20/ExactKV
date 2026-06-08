# Experiment 001 — v0.2.0 Smoke Sweep

**Version:** ExactKV v0.2.0  
**Date:** 2026-06-08  
**Reports:** `reports/v0.2.0_smoke_sweep.json`, `reports/v0.2.0_smoke_sweep.csv`,
`reports/v0.2.0_acceptance.csv`, `reports/v0.2.0_failure_report.json`

---

## 1. Purpose

This experiment verifies that the ExactKV v0.2.0 draft-verify-commit loop produces
output token IDs that **exactly match** full-KV greedy decoding across three
compressors and two draft lengths on a representative prompt suite.

It also characterises *acceptance behaviour*: how often each compressor's drafted
tokens are accepted without correction, and where lossy divergence (difference
between compressed and full output) first appears.

ExactKV is a research framework inspired by the
[VeriCache paper](https://arxiv.org/abs/2605.17613) (Yao et al., 2026). This
experiment tests ExactKV's correctness and acceptance-analysis tooling. It does
**not** test throughput, tokens/second, latency, or production performance.

---

## 2. Setup

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Precision | `float32` |
| Device | CPU |
| Decoding | Greedy (argmax, `do_sample=False`) |
| Verification | Sequential (one draft token per forward pass) |

---

## 3. Prompt Suite

6 prompts selected from `benchmarks/prompts/smoke.jsonl`, one per category:

| Prompt ID | Category | Prompt (truncated) |
|---|---|---|
| `nat_001` | `natural_language` | The capital of France is |
| `code_001` | `code` | Write a Python function that computes the factorial of n: |
| `json_001` | `json` | Generate a JSON object describing a person with name, age… |
| `cmd_002` | `command` | The git command to create and switch to a new branch… |
| `trans_001` | `translation` | Translate 'Hello, how are you?' into French: |
| `long_001` | `long_prompt` | …summarize the following passage about climate change… |

---

## 4. Compressors Tested

| Compressor | Type | Simulated | Real bytes claim | Description |
|---|---|---|---|---|
| `noop` | identity | No | No | Returns full KV cache unchanged; correctness baseline |
| `int8` | quantization | No | **Yes** | Per-tensor symmetric INT8 quantisation |
| `int4_sim` | quantization | **Yes** | **No** | Simulated INT4: values in `[-8, 7]`, stored in `torch.int8` |

> **`int4_sim` is a simulation.** It quantises values into the signed 4-bit
> numeric range but stores them in `torch.int8` containers (1 byte per element,
> not real 4-bit bit-packing). The `compressed_kv_bytes` it reports reflects
> actual `int8` storage, not a theoretical 2× improvement over `int8`. Do not
> interpret `int4_sim` memory numbers as evidence of real packed-INT4 savings.

---

## 5. Draft Lengths Tested

`4` and `8` tokens drafted per verification round.

---

## 6. Max New Tokens

`32` tokens generated per prompt in all three modes (`full`, `lossy`, `exactkv`).

---

## 7. Experiment Grid

**6 prompts × 3 compressors × 2 draft lengths = 36 total runs.**

Each run executes three modes independently:
- **`full`** — standard full-KV greedy generation (ground truth).
- **`lossy`** — greedy generation with compressed KV, no verification.
- **`exactkv`** — draft-verify-commit loop using the compressor, verified against
  full-KV greedy predictions at each position.

---

## 8. ExactKV Failure Count

> **0 out of 36 runs.**

In every run, across all compressors and draft lengths, the ExactKV output token
IDs were **exactly equal** to the full-KV greedy output token IDs.

```
exactkv_output_ids == full_output_ids   ✅  for all 36 runs
```

This is the primary correctness criterion for ExactKV. A non-zero count would
indicate a bug in the draft-verify-commit loop.

---

## 9. Lossy Divergence Count

> **16 out of 36 runs.**

Lossy divergence is when the *unverified* lossy output (`generate_lossy_greedy`)
differs from full-KV output. This is **expected** for quantisation compressors —
it demonstrates that the compressor alters the output and explains why verification
is necessary.

**Lossy divergences by compressor:**

| Compressor | Runs with lossy divergence | Runs without |
|---|---|---|
| `noop` | 0 / 12 | 12 / 12 |
| `int8` | 4 / 12 | 8 / 12 |
| `int4_sim` | 12 / 12 | 0 / 12 |

`noop` never diverges (identity compressor). `int4_sim` diverges on every prompt
because simulated INT4 aggressively quantises the KV tensors. Despite this, ExactKV
corrected **all** divergences.

**First-divergence index by prompt (for diverging runs):**

| Prompt | Compressor | Draft len | First divergence at token |
|---|---|---|---|
| `nat_001` | `int8` | 4 | token 5 |
| `nat_001` | `int8` | 8 | token 5 |
| `nat_001` | `int4_sim` | 4 | token 1 |
| `nat_001` | `int4_sim` | 8 | token 1 |
| `code_001` | `int4_sim` | 4 | token 1 |
| `code_001` | `int4_sim` | 8 | token 1 |
| `json_001` | `int4_sim` | 4 | token 1 |
| `json_001` | `int4_sim` | 8 | token 1 |
| `cmd_002` | `int4_sim` | 4 | token 2 |
| `cmd_002` | `int4_sim` | 8 | token 2 |
| `trans_001` | `int4_sim` | 4 | token 2 |
| `trans_001` | `int4_sim` | 8 | token 2 |
| `long_001` | `int8` | 4 | token 12 |
| `long_001` | `int8` | 8 | token 12 |
| `long_001` | `int4_sim` | 4 | token 2 |
| `long_001` | `int4_sim` | 8 | token 2 |

`int4_sim` consistently diverges at token 1 or 2, reflecting aggressive early
quantisation error. `int8` diverges later and only on 2 of 6 prompts.

---

## 10. Acceptance-Rate Summary by Compressor

Over 36 total runs (6 prompts × 2 draft lengths per compressor):

| Compressor | Runs | Total drafted | Total accepted | Total rejected | Total corrections | Accept rate |
|---|---|---|---|---|---|---|
| `noop` | 12 | 304 | 304 | 0 | 0 | **1.000** |
| `int8` | 12 | 319 | 297 | 22 | 7 | **0.931** |
| `int4_sim` | 12 | 503 | 231 | 272 | 73 | **0.459** |

**Interpretation:**

- `noop` is the identity baseline — 100% acceptance, by design.
- `int8` retains 93.1% of drafted tokens; ExactKV must correct only 7 tokens
  across 12 runs. INT8 is the most practical compressor in this experiment.
- `int4_sim` accepts only 45.9% of drafted tokens. More drafts are rejected than
  accepted. ExactKV still corrects every divergence, but each rejected position
  requires an additional full-KV forward pass to recover the ground-truth token.
  The high rejection rate is consistent with INT4-level quantisation error.

---

## 11. Acceptance-Rate Summary by Draft Length

| Draft length | Runs | Total drafted | Total accepted | Total rejected | Accept rate |
|---|---|---|---|---|---|
| 4 | 18 | 504 | 417 | 87 | **0.827** |
| 8 | 18 | 622 | 415 | 207 | **0.667** |

Longer draft sequences have lower acceptance rates. This is expected: a draft of 8
tokens must match the full-KV predictions at all 8 positions; a single mismatch
anywhere causes the remaining positions to be rejected. Shorter drafts are safer
but draft fewer tokens per round.

---

## 12. Acceptance-Rate Summary by Prompt Category

| Category | Runs | Accept rate | Lossy divergences |
|---|---|---|---|
| `json` | 6 | **0.903** | 2 |
| `long_prompt` | 6 | **0.752** | 4 |
| `code` | 6 | **0.745** | 2 |
| `translation` | 6 | **0.745** | 2 |
| `command` | 6 | **0.715** | 2 |
| `natural_language` | 6 | **0.622** | 4 |

JSON prompts have the highest acceptance rate (90.3%), possibly because structured
JSON output has high token-level predictability. Natural-language open-ended
prompts have the lowest acceptance (62.2%).

---

## 13. Memory Estimate Summary

Measurements are byte estimates for the KV cache after a single prefill pass of
each prompt. Reported for `nat_001` with draft length 4 as a representative sample.

| Compressor | Full KV (bytes) | Compressed KV (bytes) | Compression ratio | Memory reduction factor |
|---|---|---|---|---|
| `noop` | 122,880 | 122,880 | 1.000 | 1.00× |
| `int8` | 122,880 | 31,104 | 0.253 | **3.95×** |
| `int4_sim` | 122,880 | 31,104 | 0.253 | 3.95× ⚠️ |

> ⚠️ **`int4_sim` memory disclaimer.** Although `int4_sim`'s `compressed_kv_bytes`
> equals `int8`'s (both use `torch.int8` storage), `int4_sim` is marked
> `is_simulated=True` and `supports_real_bytes_claim=False`. The 3.95× reduction
> figure for `int4_sim` reflects `int8` container size, not a real 4-bit packed
> representation (which would use half as many bytes). These numbers must not be
> cited as evidence of real INT4 memory savings.
>
> `int8` is marked `supports_real_bytes_claim=True` — its 3.95× reduction is a
> genuine memory estimate.

**Compression ratio** = `compressed_bytes / full_bytes` (< 1 means smaller).  
**Memory reduction factor** = `full_bytes / compressed_bytes` (> 1 means savings).

---

## 14. Failure Report Summary

```json
{
  "exactkv_failure_count": 0,
  "lossy_divergence_count": 16,
  "status": "pass",
  "exactkv_failures": [],
  "lossy_divergences": [/* 16 entries — see section 9 */]
}
```

**Status: `pass`.**

The failure report distinguishes two separate counts:

| Metric | Value | Meaning |
|---|---|---|
| `exactkv_failure_count` | **0** | Correctness bugs — ExactKV output ≠ full greedy. Must always be zero. |
| `lossy_divergence_count` | **16** | Expected lossy behaviour — unverified compressed output ≠ full greedy. Demonstrates why verification is needed. |

Lossy divergence is not a failure of ExactKV. It is evidence that the compressor
meaningfully alters the KV cache and that the verification step is doing its job.

---

## 15. What This Experiment Proves

1. **Primary correctness criterion holds across all 36 runs.**
   ```
   exactkv_output_ids == full_output_ids   for noop, int8, int4_sim × dl=4, dl=8
   ```

2. **Acceptance bookkeeping is correct.**
   `total_drafted == total_accepted + total_rejected` in every run.

3. **ExactKV corrects aggressive quantisation.**
   Even with `int4_sim` (45.9% acceptance, diverges on every prompt), ExactKV
   produces exactly the same output as full-KV greedy — every time.

4. **The compressor registry, CLI, reporting, and analysis pipeline are functional.**
   - `python -m exactkv sweep` generates valid JSON and CSV reports.
   - `python -m exactkv analyze` builds acceptance tables and failure reports from
     those outputs without re-running the model.
   - All compressor capabilities, simulated-compressor flags, and memory-claim
     honesty fields appear correctly in all output formats.

5. **Draft length 4 has meaningfully higher acceptance than draft length 8.**
   82.7% vs 66.7% overall. The optimal draft length depends on the compressor
   quality and prompt characteristics.

---

## 16. What This Experiment Does Not Prove

1. **No throughput, latency, or speedup claim.**
   ExactKV v0.2.0 uses sequential verification (one draft token per forward pass)
   on CPU. It is slower than full-KV inference for most configurations. Sequential
   verification is a correctness prototype, not a production serving strategy.

2. **No production readiness.**
   This experiment runs with a single request, a single device, `float32`
   precision, and no batching. It is a research correctness and analysis exercise.

3. **No real INT4 memory savings.**
   `int4_sim`'s memory numbers reflect `int8` container storage. Real packed-INT4
   compressors are not implemented in ExactKV v0.2.0.

4. **No broad model-family generalisation.**
   All results are for `Qwen/Qwen2.5-0.5B` only. Other model architectures may
   produce different acceptance rates and divergence patterns.

5. **No statistical significance.**
   6 prompts is sufficient to verify the ExactKV correctness invariant, but is too
   small to draw statistically robust conclusions about acceptance rates by category
   or prompt type.

6. **No claim of VeriCache-level throughput recovery.**
   VeriCache proposes parallel (single-pass) batch verification to recover most of
   the throughput lost to the verify step. ExactKV V2 does not implement this.
   The acceptance rate this experiment measures characterises *potential* throughput
   recovery; it does not measure actual inference speed.

---

## 17. Known Limitations

1. **Small prompt suite.**
   6 prompts across 6 categories. The `smoke.jsonl` suite (16 prompts) and planned
   `core` / `stress` suites provide broader coverage.

2. **Sequential verification overhead.**
   Each drafted token that fails verification requires an additional full-KV
   forward pass to compute the correction token. This is expected and documented
   in `docs/V1_SCOPE_STATEMENT.md`.

3. **`DynamicCache` brittleness.**
   ExactKV reconstructs `DynamicCache` by directly injecting attributes into
   `DynamicLayer` objects. This targets `transformers 5.8.1` and may break
   across transformers versions. See `docs/V1_SCOPE_STATEMENT.md` for details.

4. **`int4_sim` acceptance rate is low by design.**
   45.9% acceptance reflects simulated INT4's quantisation noise, not a property
   of real INT4 algorithms. A real, well-designed INT4 compressor (e.g., KIVI or
   KVQuant) would likely exhibit a higher acceptance rate.

---

## Citation

The draft-then-verify compressed-KV algorithm is from:

> **VeriCache: Turning Lossy KV Cache into Lossless LLM Inference.**
> Yao et al., arXiv:2605.17613, 2026.

ExactKV does not claim to have invented this algorithm. ExactKV's contribution is
a compressor-agnostic, Hugging Face-first implementation, a structured benchmark
harness, and a framework for evaluating compressors by acceptance behaviour under
full-KV verification.
