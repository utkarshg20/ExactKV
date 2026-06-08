# PRIVATE — Future Post Notes (Experiment 003)

> 🔒 **PRIVATE DRAFT — DO NOT POST YET.**
> This file contains *draft* social/announcement copy for **later**, after the
> project is more complete. It is not for immediate posting. Nothing here is a
> public launch. Revisit and rewrite before any real post. Treat every draft
> below as a starting point, not final copy.

---

## 0. Context for future-me

These drafts are about **Experiment 003** (the v0.4.0 asymmetric K/V sweep).
Core honest framing:

* ExactKV is **inspired by VeriCache** (Yao et al., arXiv:2605.17613). I did not
  invent the draft-then-verify algorithm.
* The interesting empirical finding: **keys are more fragile than values** under
  compression, measured by ExactKV acceptance behaviour, with **0 ExactKV
  failures** across 612 runs.
* No speedup. No real INT4 memory savings. Not production-ready. Not final.

---

## 1. Short future X post draft

> Built ExactKV — a VeriCache-inspired KV-cache *verification* framework.
>
> In a 612-run asymmetric K/V sweep: 0 verification failures, and a clear signal
> that keys are much more fragile than values under compression.
>
> Full-precision keys + INT8 values → 98.8% of drafted tokens accepted.
> 4-bit keys + INT8 values → 56.2%.
>
> (Research framework, simulated compressors, no speed claims.)

---

## 2. Longer future X thread draft

**1/**
I've been building ExactKV: a correctness-first KV-cache *verification* framework,
inspired by the VeriCache paper. The idea: let a lossy KV compressor draft tokens,
then verify every token against full-KV decoding so the final output is identical
to normal decoding.

**2/**
ExactKV's job isn't to be fast. It's to answer a precise question: if you compress
the KV cache, *how often does the compressed draft already match what full-KV
would have produced* — and when it doesn't, can verification correct it?

**3/**
v0.4.0 adds asymmetric K/V compression: compress keys and values at *different*
bit-widths. Keys and values play different roles in attention, so compressing
them symmetrically may be a mistake.

**4/**
Experiment 003: 612 runs (34 prompts × 9 compressors × 2 draft lengths) on
Qwen2.5-0.5B.
- 0 ExactKV failures
- 386 lossy divergences, all corrected by verification
- mean acceptance 0.739

**5/**
The finding: keys are far more fragile than values.
- k_full_v8 (full keys, INT8 values): 0.988 acceptance
- int8 (INT8/INT8): 0.953
- k4_v8_sim (4-bit-sim keys, INT8 values): 0.562
- int4_sim (4-bit-sim both): 0.553

**6/**
So aggressive *value* compression was tolerable; aggressive *key* compression was
not. That's a concrete, measurable reason to treat K and V asymmetrically — and
acceptance behaviour caught it where a raw reconstruction-error (MSE) number
might not.

**7/**
Caveats, stated plainly: sub-INT8 compressors here are *simulated* (values stored
in int8 containers, no real bit-packing). No speedup claims. No production
readiness. ExactKV is a research/eval framework inspired by VeriCache, not a
serving system.

**8/**
Next: workspace-aware memory accounting (honest stored-vs-working-vs-scratch
bytes) and planning for real compressor backends so simulated and real policies
can be compared fairly. More later.

---

## 3. Future LinkedIn draft

> **ExactKV: evaluating KV-cache compression by acceptance behaviour, not just MSE**
>
> Over the last while I've been building ExactKV, a correctness-first KV-cache
> verification framework inspired by the VeriCache paper. It lets a lossy KV-cache
> compressor draft tokens and then verifies each token against full-KV decoding,
> so the final output stays identical to standard decoding.
>
> In the latest milestone (v0.4.0) I added asymmetric K/V compression — keys and
> values compressed at different bit-widths — and ran a 612-run sweep. Two things
> stood out:
>
> 1. **Zero verification failures across all 612 runs.** Every divergence
>    introduced by lossy compression was caught and corrected.
> 2. **Keys are much more fragile than values.** Full-precision keys with INT8
>    values accepted 98.8% of drafted tokens, while 4-bit (simulated) keys with
>    INT8 values dropped to 56.2%.
>
> The practical lesson: evaluating KV compression by *acceptance behaviour under
> verification* surfaces failure modes that an averaged reconstruction-error
> metric can hide.
>
> This is research, not a product. The sub-INT8 compressors are simulated (no
> real bit-packing), and I make no speed or production-readiness claims. Inspired
> by VeriCache — I did not invent draft-then-verify.

---

## 4. Future technical GitHub README blurb draft

> **ExactKV** is a compressor-agnostic KV-cache verification runtime and benchmark
> suite, inspired by the VeriCache paper. It runs a lossy KV compressor to draft
> tokens, verifies each token against full-KV greedy decoding, and corrects any
> divergence — so output token IDs are identical to full-KV decoding.
>
> ExactKV measures **exactness, acceptance, divergence, rejection, and correction
> behaviour** — not performance. In a 612-run asymmetric K/V sweep it recorded 0
> verification failures and showed that keys are markedly more fragile than values
> under compression (full-precision keys + INT8 values: 0.988 acceptance; 4-bit
> simulated keys + INT8 values: 0.562). Sub-INT8 compressors are simulated; no
> real bit-packing, no speedup claims, no production-readiness claims.

---

## 5. "Do not post yet" checklist

Before posting any of the above, confirm:

- [ ] Project is at a stage I'm comfortable making public.
- [ ] At least one real backend adapter exists, OR the post is explicitly framed
      as simulated-only research.
- [ ] Workspace-aware memory accounting (V5) is done or the post avoids any
      memory-savings implication.
- [ ] Numbers in the post are re-verified against the latest committed report.
- [ ] VeriCache attribution is present and correct (Yao et al., arXiv:2605.17613).
- [ ] Repo README, LICENSE, and docs are clean and self-consistent.
- [ ] No private notes, TODOs, or scratch files are exposed.
- [ ] A second read confirms no accidental speed/throughput/production wording.
- [ ] Decide on the right venue and timing deliberately — not impulsively.

---

## 6. "Do not say" section

Never claim, in any future post:

* ❌ "Production-ready" / "ready for production serving".
* ❌ "Faster" / any speedup, throughput, latency, tokens/sec, or runtime claim.
* ❌ "Real INT4 memory savings" or any real packed-memory claim for `_sim`
  compressors (they use int8 containers).
* ❌ "I invented VeriCache" / "I invented lossless KV compression" / "novel
  algorithm". ExactKV is *inspired by* VeriCache.
* ❌ "Final" / "complete" / "solved". It is a work in progress.
* ❌ Any benchmark comparison vs vLLM, LMCache, or named backends — none are
  implemented.
* ❌ Average effective bit width framed as a real memory measurement.
