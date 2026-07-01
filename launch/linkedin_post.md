# ExactKV — LinkedIn Posts (3 variants)

All three are claim-safe. Numbers trace to on-disk release artifacts.
Each includes the required caveats (Phase F microbenchmark; stored tensor byte ratios;
SpectralQuant fallback/proxy; Shard probe-first; not production; does not reproduce VeriCache).

---

## Variant 1 — Technical / research-heavy

**When does a compressed KV cache start lying?**

Most KV-cache compression evaluations report aggregate quality. They don't tell
you *when* the compressed cache first produces a different token than full
precision would have. Under greedy decoding, one perturbed logit flips the argmax
— and the trajectories split.

I built **ExactKV**, a compressor-agnostic crash-test and leaderboard that makes
that split a first-class measurement. It runs a draft/verify/commit loop — draft
from compressed KV, verify against the full-KV reference, accept the matching
prefix, correct on mismatch — and records the **first-divergence index**,
**acceptance rate**, **verifier agreement**, and **exactness failures** per cell.

**8,132 completed GPU cells. exactkv_failures = 0 throughout.**

Four main findings:

1. **Task type dominates drift.** int4_sim diverges 6% on Python code (MBPP), 11%
   on short tool-calling (BFCL), 50% on long-gen tool-calling (mnt=256), and **90%**
   on open-text reading/summarization (HF LongBench). H2O-style eviction at 75% kept:
   **100%** on LongBench — worse than int4_sim at matched memory budget.

2. **Generation length is the within-task driver.** On BFCL, int4_sim divergence
   scales from 9% (mnt=16) to 62% (mnt=256) — a 7x increase. int8 stays near-zero.

3. **Three distinct failure modes.** Logit autopsy over 1,103 divergent cells:
   near-tie noise (int8, mean rank 2.4), distribution shift (int4_sim, rank 3.5),
   attention destruction (H2O-style, rank 6.7, diverges at token 1). Same verifier
   corrects all three.

4. **ExactKV preserves 100% of downstream validity.** Despite 50% token drift,
   all 106/106 full-KV valid BFCL tool calls are preserved by the verifier (both
   models, all four task categories).

**v3.0 GPU panel (both models, 1,568 cells):** int6_sim and int4_per_vec_sim both show
0% divergence on BFCL and MBPP. On HF LongBench: int6_sim 37–47%, int4_per_vec_sim
56–57% — both non-catastrophic, exactkv_failures=0. Per-vector granularity helps on
structured tasks; bit-width still matters at 8K context.

**Phase D3 — faithful external adapters (864 cells, both models):** Real upstream
libraries wired into the same crash-test grid — **not** to crown a winner. **`int8`**
is the only non-catastrophic real compressor (~8–9% combined drift). Faithful SnapKV
via kvpress runs end-to-end but shows **90–97% drift** (stress-test failure mode).
KIVI offline r32: **100% drift** on every cell (adapter diagnostic, not production
KIVI). `exactkv_failures=0` throughout. Wave-2 KnormPress/TurboQuant smoke (128 cells)
pending RunPod recovery.

Caveats: ExactKV is a research-grade evaluation framework, not a production serving
system, and does not reproduce VeriCache. Phase F kernel results are a microbenchmark
only. Compression ratios are stored byte ratios, not active GPU memory savings.

#LLM #Inference #KVCache #MLSystems #Evaluation

---

## Variant 2 — Punchy / startup-style

KV compression looks amazing in the demo. Then it quietly changes a token — and
your output drifts.

**ExactKV** is a crash-test for compressed KV caches. It pinpoints the *first*
token where a compressed cache diverges from the full-KV verifier, explains *why*
it fails, and checks whether downstream tasks survive.

The receipts (8,132 GPU cells, exactkv_failures = 0):

→ int4_sim: 6% drift on code, **90% on reading/summarization** — same model, different task  
→ H2O-style eviction at 75% kept: **100% divergence** on LongBench (worse than int4)  
→ Generation length alone: 9% → **62%** drift as output budget grows 16→256 tokens  
→ Three failure modes, three forensic logit traces. One verifier that catches them all.  
→ Despite 50% token drift: **106/106 full-KV valid tool calls preserved**  
→ int6_sim + int4_per_vec_sim: **0% divergence on BFCL/MBPP** (both models, GPU-validated)  
→ **Faithful adapters (864 cells):** int8 ~8–9% drift; SnapKV 90–97% (real kvpress, mostly fails); KIVI r32 100% (diagnostic only)  
→ Wave-2 KnormPress/TurboQuant smoke pending RunPod recovery (not claim-ready yet)  

Not a production system. Not a VeriCache reproduction. Not a memory-savings claim.
An honest, reproducible measurement of when compressed KV starts lying.

If you build on compressed KV, you should know this number.

Repo + report in comments.

#AI #LLMs #Inference #KVCache

---

## Variant 3 — Recruiter / generalist-friendly

I just packaged up a project I'm proud of: **ExactKV**.

Large language models cache intermediate state (the "KV cache") to run fast.
Teams compress that cache to save resources — but compression can quietly change
the model's output. ExactKV is a testing framework that measures **exactly when**
a compressed cache starts producing different results than the uncompressed one,
identifies *why* it fails at the mechanism level, and publishes a leaderboard
comparing methods.

What I think makes it interesting:
- It's built around **correctness first** — a verifier checks every token.
- It's **reproducible**: 8,132 GPU cells across five benchmark families, with
  every published number tracing to a saved artifact, one-command regeneration.
- It's **honest about limits**: it's a research evaluation framework, not a
  production system — it documents exactly what it does and does not prove.
- It found something real: the same compressor diverges **6% on code but 90% on
  reading tasks** — task type, not just quantization level, is the dominant driver.
- New compressors (int6_sim, int4_per_vec_sim) GPU-validated on **both models** as
  non-catastrophic on structured tasks — with a nuanced finding on long-context reading.
- **Faithful external adapters (Phase D3):** 864-cell adapter smoke on both models
  confirms the harness wires real upstream code; int8 is the only non-catastrophic real
  compressor; SnapKV/KIVI mostly fail under token-level crash testing.

Process, correctness, and reproducibility over hype.

→ https://github.com/utkarshg20/ExactKV  
→ Technical report: https://github.com/utkarshg20/ExactKV/blob/main/paper/ExactKV_Technical_Report.md  
→ Landing page: https://github.com/utkarshg20/ExactKV/blob/main/site/index.html

#SoftwareEngineering #MachineLearning #OpenSource
