# ExactKV — X / Twitter Thread

> Claim-safe. Numbers from release artifacts. No "beats X", no speedup/VRAM claims.

---

**1/**
KV compression looks fine — until the first wrong token.

**ExactKV** crash-tests compressed KV: when drift starts, why, and whether downstream survives.

---

**2/**
Compressed KV drafts tokens → full-KV verifier checks each one → log first divergence.

Not average quality. **The first wrong token.**

---

**3/**
**8,132 GPU cells. exactkv_failures = 0.**

Llama-3.1-8B + Mistral-7B · LongBench, BFCL, MBPP, RULER, HumanEval.

---

**4/**
Same `int4_sim`, same model — drift depends on task:

Code **6%** → BFCL **11%** → LongBench **90%**

H2O-style eviction (75% kept): **100%** on LongBench.

---

**5/**
Same task, longer generation = more drift.

BFCL `int4_sim`: **9%** (16 tok) → **62%** (256 tok). int8 ≈ flat.

---

**6/**
Three failure modes (1,103 divergent cells):

int8 = near-tie noise · int4 = distribution shift · H2O = attention destruction

Same verifier fixes all three.

---

**7/**
Despite 50% token drift: **106/106** valid BFCL tool calls preserved.

---

**8/**
v3.0: `int6_sim` + `int4_per_vec_sim` → **0%** on code/tool tasks (both models).

First upstream adapter: **SnapKV** (kvpress), 87.5% MBPP drift on smoke. Verifier still holds.

---

**9/**
Caveats: research eval only · not VeriCache · not prod serving · drift ≠ official benchmark scores.

---

**10/**
Report + repro in repo. Link in reply.

**ExactKV tells you when compressed KV stops matching full-KV greedy — and why.**
