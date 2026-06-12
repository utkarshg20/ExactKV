# Experiment 028: Span Verification Design

**Status:** V13 Phase 1 — **design only**; no implementation, no cells run.
**Phase 2** will add implementation smoke under the same experiment number.

> This is a **design phase**, not an implementation.
> This does **not** claim speedup, throughput, latency, runtime, tokens/sec,
> active GPU memory savings, or production readiness.
> Span verification is only a path toward practical benchmarking **after**
> exactness is proven in Phase 2.
> ExactKV's exactness gate remains **mandatory**.

---

## 1. Purpose

Close V13 Phase 1 by documenting the span verification algorithm, invariants,
edge cases, API plan, and test strategy before any code changes.

## 2. Deliverable

Full design: [`SPAN_VERIFICATION_DESIGN.md`](SPAN_VERIFICATION_DESIGN.md).

## 3. Algorithm summary

1. Draft `k` tokens on compressed KV (unchanged).
2. Set **`v_0 = full_state.next_token_id`** (cached; not from span-forward logits).
3. If `k ≥ 2`, one teacher-forced full-KV forward over `[d_0, …, d_{k−1}]`.
4. For **`i ≥ 1`**: **`v_i = argmax(out.logits[:, i − 1, :])`**; ignore
   `out.logits[:, k − 1, :]` (bonus disabled).
5. Compare each `d_i` to `v_i`; accept longest prefix; correct at first mismatch.
6. Commit + recompress (unchanged generator semantics).

See [`SPAN_VERIFICATION_DESIGN.md`](SPAN_VERIFICATION_DESIGN.md) §4 for the HF causal LM logits-shift table.

## 4. Phase 2 test requirement (logits shift)

Golden off-by-one test on NoOp / full-greedy draft:

- `verify_span` → 100% acceptance when draft matches greedy chain.
- `verifier_tokens[0] == full_state.next_token_id`.
- When `len(draft_tokens) > 1`: `verifier_tokens[1] == argmax(logits[:, 0, :])` after span forward.

## 5. Exactness expectation (Phase 2)

Span mode must produce **identical** `output_ids` to sequential mode and
`generate_full_greedy` on the Exp 029 grid — `exactkv_failures == 0`.

## 6. What this experiment proves (Phase 1)

- Design completeness for Phase 2 implementation.
- Backward compatibility plan (sequential default).

## 7. What this does not prove

- Speed improvement — Phase 3 only, diagnostic.
- Implementation correctness — Phase 2.

## 8. Next step

Phase 2: implement `VerificationEngine.verify_span` and generator flag per design doc.
