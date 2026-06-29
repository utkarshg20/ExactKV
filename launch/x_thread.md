# ExactKV — X / Twitter Thread

> Claim-safe launch thread. Every number traces to on-disk release artifacts.
> Strong but evidence-bounded. No "beats X", no "first ever", no speedup/memory claims.

---

**1/**
KV compression looks fine — until the first wrong token.

I built **ExactKV**: a crash-test framework that measures *exactly* when a compressed KV cache starts lying, why it fails, and whether downstream tasks are actually affected.

---

**2/**
The setup:

→ draft tokens from **compressed** KV  
→ verify each against the **full-KV** reference  
→ accept the matching prefix, correct on mismatch  
→ log the *first divergence*, the mechanism, and any downstream impact

Everyone reports average quality. ExactKV reports the first wrong token.

---

**3/**
**7,348 completed GPU cells. exactkv_failures = 0.**

Llama-3.1-8B + Mistral-7B across five benchmark families:
LongBench (reading/summarization), BFCL (tool-calling), MBPP (code), RULER, HumanEval.

Six compressors: noop, int8, int4_sim, H2O-style eviction, int6_sim, int4_per_vec_sim.

Source: `reports/scale_7b/raw.json` + `reports/external_panels/`

---

**4/**
**Finding 1: Task type dominates drift.**

int4_sim divergence is *not* a single number:

- MBPP code: **6%**
- BFCL short tool-calling: **11%**
- BFCL long-gen (mnt=256): **50%**
- HF LongBench reading: **90%**

H2O-style eviction at 75% kept: **100%** on LongBench — *worse* than int4_sim at matched memory budget.

---

**5/**
**Finding 2: Generation length is the within-task driver.**

On BFCL, int4_sim divergence scales **7x** from generation budget:

mnt=16 → 9%  
mnt=32 → 17%  
mnt=128 → 45%  
mnt=256 → **62%**

int8 stays near-zero throughout. Same task, same context — generation budget alone explains the gap.

---

**6/**
**Finding 3: Three distinct failure modes.**

Top-k logit autopsy over **1,103 divergent cells**:

**Near-tie noise** (int8): 66% near-tie, mean lossy rank 2.4, fdi=22  
**Distribution shift** (int4_sim): 83% flip, rank 3.5, fdi=8  
**Attention destruction** (H2O-style): 100% flip, rank 6.7, fdi=1

Three compressor classes. Three different failure signatures. The same verifier corrects all three.

---

**7/**
**Finding 4: ExactKV preserves 100% of downstream validity.**

Despite 50% token drift, the verifier protects **106/106 full-KV valid BFCL tool calls** (both models, all four task categories: simple, parallel, multi-turn, AST-eval).

Drift ≠ task failure — but only because the verifier catches it.

---

**8/**
**v3.0 GPU results: two new compressors validated (Mistral-7B, 784 cells).**

`int6_sim` (6-bit per-tensor):
- BFCL / MBPP: **0% divergence**
- LongBench: **37.5%** (vs 86.1% for int4_sim)

`int4_per_vec_sim` (4-bit per-vector, KIVI/KVQuant-style):
- BFCL / MBPP: **0% divergence** — matches int8
- LongBench: **55.6%** — non-catastrophic but higher than int6_sim

Key nuance: per-vector granularity eliminates drift on structured tasks, but 4-bit resolution still matters at 8K context. "Granularity > bit-width" is task-conditional.

exactkv_failures = 0 throughout.

---

**9/**
Reproduce it:

```
python3 scripts/exactkv_repro.py --reports-only
bash scripts/build_paper_pdf.sh
```

Full technical report: `paper/ExactKV_Technical_Report.md`

---

**10/** Read this before you cite anything:

- **Not** a production serving system; does **not** reproduce VeriCache  
- Phase F int8/int4 ratios = **kernel microbenchmark only** (not end-to-end)  
- Compression ratios = **stored tensor byte ratios** (no VRAM savings claim)  
- v3.0 Llama-3.1-8B panels in progress (HF auth resolved, running now)  
- External panels are **drift measurements**, not official benchmark scores  

---

**11/**
The strongest honest version of this:

**ExactKV tells you exactly when compressed KV cache behavior stops matching the verifier, why it fails, and what it means for downstream tasks.**

Token-level drift evidence. Three failure-mode signatures. Public leaderboard. Hard claim boundaries.
