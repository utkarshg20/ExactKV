# Phase 17 Demo Script

Companion: [`PHASE_17_CLAIM_SAFE_DEMO.md`](PHASE_17_CLAIM_SAFE_DEMO.md)

---

## 30-second version

Everyone is racing to shrink KV caches. ExactKV tells you when they start lying.

KV compression should not be trusted — it should be crash-tested. Outcome benchmarks tell you if the answer scored well; ExactKV tells you if compression changed the model's path before the answer looked fine.

We built offline attention drift probes, a diagnostic tolerance policy, and guarded shadow observers that never touch token commit. In tested panels, shadow did not change generated tokens.

We are **not** claiming speedups, memory savings, or VeriCache serving reproduction.

---

## 2-minute version

**Hook:** Everyone is racing to shrink KV caches. ExactKV tells you when they start lying.

**Problem:** Teams ship smaller KV caches for throughput and memory wins, but lossy compression can drift attention silently. KV compression should not be trusted. It should be crash-tested.

**Benchmark gap:** Outcome benchmarks can tell you whether the answer scored well. ExactKV tells you whether KV compression changed the model's path before the answer looked fine.

**What we built (Phase 16):**
1. Offline streaming-attention diagnostics on Qwen2.5 — drift across layers and prefix lengths.
2. A tolerance policy that labels local alignment vs free-running accumulation — diagnostic only, not an exactness guarantee.
3. Generation-shadow observers that replay fixed sequences **after** generation — never during token commit.
4. An opt-in live round observer that records post-commit snapshots with baseline parity in tested panels.
5. A guarded decode-time shadow dry-run — shadow inside the observer callback, still diagnostic-only.

**Safety:** In tested Phase 16 panels, guarded shadow did not change generated tokens; exactkv_failures remained zero.

**What we do not claim:** Speed, throughput, latency, active GPU memory savings, production serving, or VeriCache reproduction.

---

## 5-minute version

Use the 2-minute script, then add:

**Attention drift card:** Walk through Qwen-family divergence panel (Exp 073) and numerics audit (070). Show that streaming vs materialized attention can diverge at boundary tolerances — that's the "lying" signal.

**Tolerance policy card:** Explain `local_alignment_pass_free_running_accumulation` as a **policy label**, not proof of exactness. Top-k agreement is supplementary only.

**Generation shadow card:** Show decode-prefix ladder (079) — shadow at each prefix step, post-hoc. Contrast with "integrated serving" — we are not there.

**Live observer card:** Exp 082 — baseline vs observer 16/16 token match; snapshots feed post-hoc shadow.

**Guarded decode-time card:** Exp 084 — 32 cells, 32/32 parity, 53 callback shadows, decode-time matches post-hoc. Emphasize: default runtime unchanged; observer optional.

**Claim freeze card:** Read allowed vs forbidden claims from Phase 16 closeout. Invite questions.

---

**Phase 17B:** Add optional 30-second line: "We validated the guarded shadow path on Qwen2.5 base and instruct — panel-scoped, not a benchmark."

**Phase 17C:** Add optional line: "We extended that panel to longer deterministic prompts at ~128–512 tokens — context-length-scoped, not long-context production support."

---

## Technical walkthrough version

1. **Architecture:** ExactKV = compressed KV proposes drafts; full-KV verifier commits. Phase 16 adds **parallel diagnostic shadow** tracks that never feed back.
2. **Tensor layer (066):** Reference int8 streaming attention feasibility.
3. **HF probes (067–073):** Single-layer → multi-layer → full-depth divergence.
4. **Policy (074):** `AttentionTolerancePolicy` on shadow cells.
5. **External shadow (076–080):** Post-hoc replay modes; round-log boundaries.
6. **Live observer (081–082):** `LiveRoundObserver` + `ExactKVGenerator.round_observer` opt-in hook.
7. **Guarded decode-time (083–084):** `GuardedDecodeTimeShadowObserver` in callback; parity gates.
8. **Artifacts:** `exactkv/demo/phase17_claim_safe_demo.py`, Exp 086 report JSON.
9. **Explicit non-goals:** No vLLM, LMCache, CUDA/Triton kernels in this story.

---

## Q&A

| Question | Answer |
|----------|--------|
| Are you claiming speedups? | **No.** |
| Are you claiming memory savings? | **No** active GPU or production memory claim yet. |
| Did you reproduce VeriCache serving? | **No.** |
| Is streaming attention used for token commit? | **No.** Diagnostic observer paths only. |
| What did Phase 16 prove? | Guarded diagnostic shadow infrastructure works in tested panels without changing generated tokens. |
| What remains? | CUDA/Triton, vLLM/LMCache, active memory validation, broader model/context validation. |
| Does top-k agreement prove exactness? | **No.** Supplementary only. |
| Is ExactKV production-ready? | **No.** Correctness-first crash-test lab. |
