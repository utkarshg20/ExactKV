# Span Verification Design

**Status:** V13 Phase 1 — **design only**; no implementation.
**Experiment:** 028 (design portion).
**Supersedes:** informal D21 notes; builds on V1 sequential verification.

> This is a **design phase**, not an implementation.
> This does **not** claim speedup, throughput, latency, runtime, tokens/sec,
> active GPU memory savings, or production readiness.
> Span verification is only a path toward practical benchmarking **after**
> exactness is proven in Phase 2.
> ExactKV's exactness gate (`exactkv_failures == 0`) remains **mandatory**.

Companion experiment memo: [`EXPERIMENT_028_SPAN_VERIFICATION_DESIGN.md`](EXPERIMENT_028_SPAN_VERIFICATION_DESIGN.md).

---

## 1. Purpose

Define **span verification** — a verifier algorithm that compares an entire draft
span against full-KV greedy predictions in **one** (or fixed small number of)
full-model forward pass(es), producing the **same acceptance semantics** as today's
sequential verifier while potentially reducing verifier call count before Phase 2
implementation and Phase 3 diagnostic timing.

---

## 2. Why span verification is needed

Experiment 027 and [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md) identified
parallel/span verification as **P0** because:

| Issue | Sequential today |
|---|---|
| Verifier cost | Up to **`(draft_len − 1)`** full forwards per round when all draft tokens match |
| Practicality | Correctness-first V1 loop is not optimized for wall-clock |
| Benchmark path | Phase 3 timing harness needs a span arm **after** exactness is proven |

Span verification does **not** assume speedup — it enables a fair comparison arm.
Whether wall-clock improves depends on implementation, hardware, and draft length;
that is measured in **Phase 3**, not assumed here.

---

## 3. Current sequential verification behavior

Implemented in `exactkv/verification/engine.py` as `verify_sequential`.

**Round start state:**

- `full_state.past_key_values` — authoritative KV covering prompt + all committed tokens.
- `full_state.next_token_id` — greedy prediction for the **next** token (cached from last commit forward).

**Algorithm (per draft token):**

1. `v_i ← next_token_id` at `i=0`; later `v_i` from prior forward output.
2. Compare `d_i` (draft) with `v_i` (verifier).
3. On mismatch at `i`: stop; `verifier_tokens = [v_0, …, v_i]`.
4. On match with more draft remaining: forward **`d_i`** through a **deep-copied** temp KV; read `v_{i+1}`.
5. Return `compute_acceptance(draft_tokens, verifier_tokens)`.

**Properties preserved today:**

- Authoritative `full_state` is **never mutated** (temp KV is deep-copied).
- `ExactKVGenerator._commit` advances authoritative KV; `update_after_commit` realigns compressed draft KV.
- Acceptance semantics in `compute_acceptance` are **pure** and shared.

**Cost model:** `O(draft_len)` full forwards in the worst case (all match).

---

## 4. Proposed span verification algorithm

### Round flow (unchanged generator skeleton)

1. **Draft** `k` tokens on compressed/materialized draft KV (`_draft`, unchanged).
2. **Verify span** on authoritative full KV (new `verify_span`).
3. **Accept** longest matching prefix via existing `compute_acceptance`.
4. **Reject** at first mismatch; **correct** with verifier token at mismatch.
5. **Commit** accepted prefix + correction (if any) to authoritative full KV (`_commit`, unchanged semantics).
6. **Realign** compressed draft KV from new full state (`update_after_commit`, unchanged).

### Span verify core idea: teacher-forced full-KV forward

Given draft tokens `[d_0, …, d_{k−1}]` and authoritative prefix KV `K`:

1. Deep-copy `K` to `temp_kv` (same safety as sequential).
2. Set **`v_0 = full_state.next_token_id`** — the greedy prediction for the next token
   **before** any draft token is fed (same source as sequential step 0; **not** from
   span-forward logits).
3. If `k == 1`, skip forward; `verifier_tokens = [v_0]` only.
4. If `k ≥ 2`, build `input_ids = [[d_0, d_1, …, d_{k−1}]]` and run **one forward:**
   `out = runtime.forward(input_ids, past_key_values=temp_kv)`.
5. For each `i ∈ [1, k−1]`:
   - **`v_i = argmax(out.logits[:, i − 1, :])`** — greedy prediction **after** the model
     has processed `d_0…d_{i−1}` (equivalent to sequential’s post-`d_{i−1}` forward).
6. **Do not use** `out.logits[:, k − 1, :]` for acceptance — that position is the
   **bonus** prediction after the full draft span; bonus-token acceptance remains
   **disabled** (V1 decision).
7. Walk `i = 0…k−1`: if `d_i == v_i`, continue; else stop and set
   `verifier_tokens = [v_0, …, v_i]` (same stop rule as sequential).
8. If all match: `verifier_tokens = [v_0, …, v_{k−1}]`.
9. Discard `temp_kv`; assert authoritative `full_state` unchanged.
10. Return `compute_acceptance(draft_tokens, verifier_tokens)`.

#### HF causal LM logits shift (critical)

In standard Hugging Face causal LM semantics, `out.logits[:, j, :]` at input position
`j` is the distribution for the token **following** the prefix ending at that position.
After feeding `input_ids = [d_0, …, d_{k−1}]` with existing `past_key_values`:

| Logits index | Predicts token after… | Maps to verifier |
|---|---|---|
| (cached) | committed prefix only | **`v_0`** via `full_state.next_token_id` |
| `[:, 0, :]` | `d_0` | **`v_1`**, not `v_0` |
| `[:, i−1, :]` for `i ≥ 1` | `d_0…d_{i−1}` | **`v_i`** |
| `[:, k−1, :]` | full draft span | **Bonus** (ignored in Phase 2) |

**Wrong mapping (do not implement):** `v_i = argmax(out.logits[:, i, :])` — this shifts
all verifier tokens by one and breaks parity with `verify_sequential`.

**Why this matches sequential:** Sequential verification reads `v_0` from cache, then
feeds each matched `d_i` before reading `v_{i+1}`. The teacher-forced span forward
materializes the same `v_1…v_{k−1}` in one pass via the `i−1` logits index. On
mismatch at `i`, only `v_0…v_i` matter for acceptance — logits at indices `≥ i`
(including the bonus slot) are ignored, matching sequential early stop.

### Acceptance / commit (unchanged semantics)

- `accepted_tokens` = longest prefix where `d_i == v_i`.
- `correction_token` = `v_i` at first mismatch, else `None`.
- `rejected_tokens` = `draft_tokens[mismatch_idx:]` (includes mismatched draft token).
- **Commit list** = `accepted_tokens` + optional `[correction_token]`, then EOS truncate.

---

## 5. Exactness invariant

| Invariant | Requirement |
|---|---|
| **Greedy equivalence** | Final `output_ids` must equal `generate_full_greedy` for same prompt, model, compressor, `max_new_tokens`, `draft_len`. |
| **Rejected never committed** | Tokens in `rejected_tokens` never appear in authoritative `generated_ids`. |
| **Authoritative verifier** | Only full-KV predictions (not draft KV) determine accept/reject/correct. |
| **Correction is verifier token** | On mismatch, committed token at fault position is `v_i`, not `d_i`. |
| **Hard gate** | `exactkv_failures == 0`; per cell `exactkv_token_match == True`. |

Span verification is valid **only if** it produces bit-identical outputs to sequential
verification for all cells in the exactness grid (Exp 029).

---

## 6. Cache-state invariant

| Invariant | Requirement |
|---|---|
| **Authoritative advance** | After commit, `full_state` equals what `_commit` would produce feeding the same committed token list — identical to full greedy prefix extension. |
| **Alignment** | After every round: `full_state.seq_len == compressed_state.logical_seq_len`. |
| **No stale draft cache** | After mismatch, compressed state is rebuilt from **new** full state via `update_after_commit`, not left at pre-mismatch draft length. |
| **Verify isolation** | Span forward uses **deep-copied** temp KV only; `full_state.past_key_values` length and `next_token_id` unchanged post-verify. |
| **EOS commit rule** | No forward pass **after** committed EOS (matches `_commit` and `generate_full_greedy`). |

---

## 7. Edge cases

| Case | Behavior |
|---|---|
| **All tokens accepted** | `all_matched=True`; commit full draft span; no correction. |
| **Mismatch at first token** | `accepted=[]`; `correction=v_0`; all draft tokens rejected. |
| **Mismatch in middle** | `accepted=draft[:i]`; `correction=v_i`; `rejected=draft[i:]`. |
| **EOS inside draft span** | If `d_i == EOS` and matches `v_i`, accept through EOS; `_truncate_at_eos` stops generation; no forward after EOS in commit. |
| **Verifier predicts EOS before draft ends** | At first `d_i != v_i` where `v_i == EOS`: accept prefix, correct to EOS, reject remaining draft tokens. |
| **Draft emits EOS early** | `_draft` stops appending; shorter draft list verified normally. |
| **Empty span** | `draft_tokens=[]` → trivial `compute_acceptance([], [])`; generator should not commit empty (existing safety). |
| **`max_new_tokens` boundary** | `n = min(draft_len, remaining)` unchanged; span length ≤ remaining budget. |
| **Cache length mismatch** | Pre-verify assert `kv_seq_len(full_state.past_key_values) == full_state.seq_len`; post-verify assert unchanged. |
| **Tokenizer mismatch** | **N/A** — span operates on token IDs inside HF runtime; no cross-tokenizer path. |
| **`draft_len == 1`** | Only `v_0 = next_token_id`; no span forward required; must match sequential. |
| **NoOp / int8 / lossy compressors** | Same verify path; draft tokens may diverge; acceptance handles mismatch. |

---

## 8. Algorithm pseudocode

```
function verify_span(full_state, draft_tokens):
    if draft_tokens is empty:
        return compute_acceptance([], [])

    assert kv_len(full_state.past_key_values) == full_state.seq_len
    kv_len_before = kv_len(full_state.past_key_values)
    next_before = full_state.next_token_id

    temp_kv = deepcopy(full_state.past_key_values)
    k = len(draft_tokens)

    # v_0 from cache — NOT from logits[:, 0, :]
    verifier_tokens = [full_state.next_token_id]

    if k >= 2:
        input_ids = tensor([draft_tokens])   # shape [1, k]
        out = runtime.forward(input_ids, past_key_values=temp_kv)
        for i in range(1, k):
            v_i = argmax(out.logits[:, i - 1, :])
            verifier_tokens.append(v_i)
        # out.logits[:, k - 1, :] is bonus after full span — ignore (bonus disabled)

    # Compare draft vs verifier; truncate verifier_tokens on mismatch
    for i, d_i in enumerate(draft_tokens):
        if d_i != verifier_tokens[i]:
            verifier_tokens = verifier_tokens[: i + 1]
            break

    assert verifier_tokens[0] == full_state.next_token_id
    assert kv_len(full_state.past_key_values) == kv_len_before
    assert full_state.next_token_id == next_before

    return compute_acceptance(draft_tokens, verifier_tokens)


function generate_round_span_mode(full_state, compressed, draft_len, max_new_tokens):
    remaining = max_new_tokens - len(generated_so_far)
    n = min(draft_len, remaining)
    draft = draft_on_compressed_kv(compressed, n)
    acceptance = verify_span(full_state, draft.token_ids)

    committed = acceptance.accepted_tokens
    if acceptance.correction_token is not None:
        committed.append(acceptance.correction_token)

    committed, eos_found = truncate_at_eos(committed)
    full_state = commit(full_state, committed)
    compressed = compressor.update_after_commit(compressed, full_state)
    assert full_state.seq_len == compressed.logical_seq_len
    return full_state, compressed, committed, acceptance, eos_found
```

---

## 9. Data structures

| Structure | Role | New? |
|---|---|---|
| `draft_tokens: list[int]` | Proposed span from `_draft` | Existing (`DraftResult.token_ids`) |
| `verifier_tokens: list[int]` | Full-KV greedy predictions per position | Existing (`AcceptanceResult.verifier_tokens`) |
| `accepted_tokens` | Longest matching prefix | Existing |
| `correction_token: int \| None` | Verifier token at mismatch | Existing |
| `rejected_tokens` | Suffix from mismatch | Existing |
| `AcceptanceResult` | Pure accept/reject bundle | **Reuse unchanged** |
| `VerificationTrace` | Per-round audit | Existing; optional `verification_method: str` later if schema approved |
| Commit record | `committed` token list per round | Existing via trace + counters |
| `total_accepted / total_rejected / total_corrections` | Bookkeeping | Existing in `ExactKVResult` |

**No new core datatypes required for Phase 2 MVP** — span verify returns the same
`AcceptanceResult` as sequential.

---

## 10. Proposed API changes (Phase 2 — not implemented in Phase 1)

### `VerificationEngine`

```python
@torch.no_grad()
def verify_span(
    self,
    full_state: FullKVState,
    draft_tokens: list[int],
) -> AcceptanceResult:
    """Verify draft span in one teacher-forced full-KV forward (design target)."""
```

- `verify_sequential` **remains**; default path unchanged.

### `ExactKVGenerator`

Option A (preferred):

```python
def __init__(..., verification_method: Literal["sequential", "span"] = "sequential"):
```

Option B: separate `ExactKVSpanGenerator` subclass — only if flag pollution is undesirable.

`_verify_draft_tokens` dispatches:

```python
if self.verification_method == "span":
    return self.engine.verify_span(full_state, draft_tokens)
return self.engine.verify_sequential(full_state, draft_tokens)
```

### Result metadata (optional, additive only)

If schema approval allows later:

- `ExactKVResult.verification_method: str`
- `VerificationTrace.verification_method: str`

Phase 2 can prove exactness **without** report schema changes by testing only.

---

## 11. Backward compatibility

| Rule | Detail |
|---|---|
| **Default** | `verification_method="sequential"` — zero behavior change for existing callers. |
| **CLI / sweeps** | Unchanged until explicit opt-in flag in Phase 2. |
| **`compute_acceptance`** | Unchanged — span and sequential share acceptance logic. |
| **Reports** | No schema change in Phase 2 unless additive fields approved. |

---

## 12. Testing plan

### Phase 2 unit tests (no model)

- Reuse and extend `tests/test_acceptance_logic.py` — **no change required** if span produces identical `verifier_tokens` lists.

### Phase 2 engine parity tests (`tests/test_verification_engine.py` lineage)

| Test | Assert |
|---|---|
| `verify_span` vs `verify_sequential` same `AcceptanceResult` | All-match, first mismatch, middle mismatch, single token |
| Authoritative state unchanged | KV len, `next_token_id` |
| **Golden off-by-one / logits-shift test** | See below |
| `v_0 == full_state.next_token_id` | Always from cache, never from `logits[:, 0, :]` |

#### Golden off-by-one test (Phase 2 required)

On a tiny real-model panel (NoOp / pass-through draft equals full greedy):

1. **`verify_span` achieves 100% acceptance** when draft tokens match full-model greedy chain.
2. **`verifier_tokens[0] == full_state.next_token_id`** (cached, pre-forward).
3. When `len(draft_tokens) > 1`, after one span forward:
   **`verifier_tokens[1] == argmax(out.logits[:, 0, :])`** — confirms `v_1` comes from
   logits index `0` (prediction after `d_0`), not index `1`.
4. Parity: same `AcceptanceResult` as `verify_sequential` on identical drafts.

This test catches the incorrect mapping `v_i = argmax(logits[:, i, :])` before Exp 029.

### Phase 2 integration tests

| Test | Assert |
|---|---|
| Tiny prompt + NoOp | Span mode `output_ids == generate_full_greedy` |
| Intentionally wrong draft | Same partial accept as sequential |
| EOS in span | Generation stops; no post-EOS forward |
| Cache alignment each round | `full_state.seq_len == compressed.logical_seq_len` |

### Experiment 029 exactness grid

- Prompt panel × compressors (`noop`, `int8`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim`).
- Compare sequential vs span: **`exactkv_failures == 0`** for both; outputs identical.
- Bookkeeping: `total_accepted + total_rejected` reconciles per round.

### Existing test files to extend (Phase 2)

- `tests/test_verification_engine.py` — span parity
- `tests/test_acceptance_logic.py` — unchanged
- `tests/test_backend_adapter_poc.py` — span mode pass-through exactness
- `tests/test_serving_cache_lifecycle.py` — alignment invariants with span mode if generator flag added

---

## 13. Benchmarking implications

| Statement | Status |
|---|---|
| Span **may** reduce full-model forward count per verify round | Design hypothesis — from  `O(k)` to `O(1)` forwards for verify step |
| Draft + commit + recompress costs unchanged | Still dominate in some regimes |
| **This design does not prove speedup** | No timing in Phase 1 |
| Phase 3 harness | Compare full greedy, lossy-only, sequential ExactKV, span ExactKV on **same cells** after Exp 029 |
| Headline timing claims | **Forbidden** until Phase 3 methodology + Phase 9 approval |

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| **Forward semantics** | Teacher-forced span forward must match step-wise logits; parity tests vs sequential |
| **DynamicCache mutation** | Deep-copy before span forward; assert authoritative KV unchanged |
| **EOS handling** | Reuse `_truncate_at_eos` and `_commit` EOS rule; dedicated EOS tests |
| **Mismatch repair** | Reuse `compute_acceptance`; no new commit logic |
| **Recompression overhead** | Unchanged V1 strategy; memory/speed impact measured separately in Phases 3–4 |
| **Long span OOM** | Phase 2: cap at existing `draft_len`; chunked span verify deferred |
| **Logits position off-by-one** | Use §4 HF shift table: `v_0` from cache; `v_i` for `i≥1` from `logits[:, i−1, :]`; golden off-by-one test in Phase 2 |
| **Bonus tokens** | Remain **disabled** in Phase 2; ignore `logits[:, k−1, :]` |

---

## 15. Implementation plan for Phase 2

| Step | Deliverable |
|---|---|
| 1 | Add `VerificationEngine.verify_span` per §4 |
| 2 | Add `verification_method` flag to `ExactKVGenerator` (default `sequential`) |
| 3 | Wire `_verify_draft_tokens` dispatch |
| 4 | Parity tests: span vs sequential on `test_verification_engine.py` panel |
| 5 | Integration: span mode exactness vs `generate_full_greedy` (NoOp, int8, sim compressors) |
| 6 | Experiment 028 smoke report — small cell count, `exactkv_failures == 0` |
| 7 | Experiment 029 grid — full parity sequential vs span outputs |
| 8 | Document in `EXPERIMENT_028` / `EXPERIMENT_029` reports |

**Out of Phase 2 scope:** timing harness, schema changes, default flag flip, bonus tokens, sampling.

---

## 16. What this design proves

- A concrete span verification algorithm **compatible** with existing acceptance and commit semantics.
- A path to reduce verifier forward count **without** changing exactness rules.
- Clear invariants, edge cases, API surface, and test plan for Phase 2.
- Backward compatibility: sequential remains default until span exactness is proven.

---

## 17. What this design does not prove

- That span verification is **faster** — timing is Phase 3.
- That ExactKV saves GPU memory — Phase 4.
- That outputs match full greedy **before implementation** — requires Phase 2 tests.
- Production serving readiness — unchanged no-go from Exp 017.
- Model accuracy improvement — forbidden claim category.

---

## Related

- [`V13_SCOPE_STATEMENT.md`](V13_SCOPE_STATEMENT.md) — Phase 1 deliverable
- [`EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md`](EXPERIMENT_027_PERFORMANCE_MEMORY_TRUTH_BOUNDARY.md) — why sequential blocks speed headlines
- [`PRACTICALITY_GAP_ANALYSIS.md`](PRACTICALITY_GAP_ANALYSIS.md) — D21 / P0 gap
- `exactkv/verification/engine.py` — current sequential implementation
- `exactkv/verification/acceptance.py` — `compute_acceptance`
- `exactkv/runtime/exactkv_generator.py` — draft-verify-commit loop
