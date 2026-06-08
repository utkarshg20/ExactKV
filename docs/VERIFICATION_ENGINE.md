# 13_VERIFICATION_ENGINE.md

# ExactKV Verification Engine

## Purpose of this document

This document defines the verification engine for ExactKV.

The verification engine is the most important correctness component in the entire project.

If this component is wrong, ExactKV is wrong.

## Core responsibility

The verification engine takes tokens drafted using compressed KV and verifies them against full-KV decoding.

It decides:

- Which drafted tokens are accepted
- Which drafted tokens are rejected
- What correction token should be committed
- Whether a bonus verifier token can be accepted
- Whether the final output remains identical to full-KV decoding

## Conceptual model

Compressed KV is treated as a speculative drafter.

Full KV is treated as the verifier and source of truth.

```text
Compressed KV says:
A B C D E

Full KV says:
A B C X ...

Result:
Accept A B C
Reject D E
Commit X
```

## Exactness guarantee

Under greedy deterministic decoding:

```python
exactkv_output_ids == full_kv_output_ids
```

This is the primary correctness guarantee.

The verification engine is responsible for enforcing this.

## Inputs

The verification engine receives:

```python
full_state: FullKVState
draft_tokens: list[int]
model_runtime: ModelRuntime
```

Optional:

- current last token
- maximum verification length
- model-specific cache metadata
- debug flags

## Outputs

The verification engine returns:

```python
AcceptanceResult
```

## AcceptanceResult structure

```python
@dataclass
class AcceptanceResult:
    accepted_tokens: list[int]
    rejected_tokens: list[int]
    correction_token: int | None
    bonus_token: int | None
    first_mismatch_index: int | None
    all_matched: bool
    verifier_tokens: list[int]
    accepted_count: int
    drafted_count: int
```

## Field definitions

### accepted_tokens

Drafted tokens that matched full-KV predictions and can be committed.

### rejected_tokens

Drafted tokens after the first mismatch. These must not be committed.

### correction_token

The full-KV token at the first mismatch.

If there is a mismatch, this token must be committed after the accepted prefix.

### bonus_token

If all drafted tokens match, full-KV verification produces one additional next token after the draft. This bonus token can be committed.

Phase 1 may skip bonus token acceptance for simplicity.

### first_mismatch_index

The index of the first drafted token that does not match full-KV prediction.

Uses zero-based indexing.

If all tokens match, this is `None`.

### all_matched

Boolean indicating whether the full draft matched.

### verifier_tokens

The full-KV predicted tokens for each verified position.

### accepted_count

Number of drafted tokens accepted.

### drafted_count

Number of drafted tokens proposed.

## Verification algorithm

## Simple sequential verification

Phase 1 should use sequential verification first because it is easier to implement correctly.

### Algorithm

```text
Input:
    full_state
    draft_tokens = [t1, t2, ..., tx]

Initialize:
    accepted = []
    verifier_tokens = []

For i in range(x):
    Run full-KV next-token prediction.
    Let v_i be the full-KV greedy token.
    Append v_i to verifier_tokens.

    If v_i == draft_tokens[i]:
        Accept draft_tokens[i].
        Update temporary full-KV state with draft_tokens[i].
        Continue.
    Else:
        Reject draft_tokens[i:].
        correction_token = v_i.
        Return result.

If all draft tokens matched:
    Optionally generate bonus token.
    Return all matched result.
```

### Pros

- Simple
- Easier to debug
- Easier to compare against full generation
- Does not require parallel verification complexity

### Cons

- Slow
- Does not reproduce VeriCache's efficient multi-token verification
- May not show speedup

This is acceptable for Phase 1.

## Parallel verification

The VeriCache paper verifies a draft span in one forward pass.

This is more efficient and should be implemented later.

### Concept

Given draft tokens:

```text
t1, t2, ..., tx
```

The verifier runs one forward pass over the drafted positions conditioned on full KV and previous draft tokens.

It obtains:

```text
t1*, t2*, ..., tx*, t(x+1)*
```

where:

- `t1*` is the full-KV prediction before draft token 1
- `t2*` is the full-KV prediction after accepting draft token 1
- ...
- `t(x+1)*` is the bonus token after all drafted tokens

### Pros

- More efficient
- Closer to VeriCache
- Better speedup potential

### Cons

- Harder to implement correctly with Hugging Face cache APIs
- More sensitive to attention masks and position IDs
- More complex to debug

### Roadmap

- V1: sequential verification
- V3 or V5: parallel verification

## Acceptance rule

The acceptance rule is strict.

For each draft token:

```python
draft_tokens[i] == verifier_tokens[i]
```

If true, the token may be accepted.

If false:

- Stop accepting draft tokens.
- Reject token `i` and all later draft tokens.
- Commit the verifier token at position `i` as correction.
- Resume drafting from the corrected state.

## Bonus token rule

If all drafted tokens match, the verifier produces a bonus token.

In classic speculative decoding and VeriCache, this bonus can be accepted.

### V1 decision

Bonus token acceptance may be disabled initially.

Why:

- Simpler correctness accounting
- Easier trace interpretation
- Avoids cache-update complexity

### Later decision

Enable bonus token to improve efficiency.

## State update rules

After verification, the generator must update:

1. FullKVState
2. CompressedKVState
3. Output token list
4. Metrics trace

## Case 1: mismatch occurs

Draft:

```text
A B C D
```

Verifier:

```text
A B X
```

Commit:

```text
A B X
```

Actions:

- Add `A B` to output.
- Add correction token `X` to output.
- Discard `C D`.
- Update full KV with `A B X`.
- Update compressed KV to represent `A B X`.

## Case 2: all draft tokens match

Draft:

```text
A B C D
```

Verifier:

```text
A B C D E
```

Commit if bonus disabled:

```text
A B C D
```

Commit if bonus enabled:

```text
A B C D E
```

Actions:

- Add accepted tokens to output.
- Optionally add bonus token.
- Update both caches.

## Important invariant: do not commit unverified tokens

A drafted token must not be appended to final output until verified.

This includes tokens that seem likely or high-confidence.

## Temporary full state

Sequential verification can use a temporary full state.

Why:

During verification, we may need to test draft tokens step by step. If a mismatch occurs, we should not corrupt the authoritative state with rejected tokens.

Implementation options:

### Option A: clone full state

Clone full KV before verification and advance the clone.

Pros:

- Safe

Cons:

- Expensive

### Option B: verify and rebuild

Use full state to generate correction and then rebuild/update after accepted tokens.

Pros:

- Simpler state management maybe

Cons:

- Inefficient

### Option C: commit as we go but rollback

Pros:

- Efficient if implemented carefully

Cons:

- Dangerous in V1

### V1 decision

Prefer safety over efficiency.

Use a simple approach even if it recomputes more.

## Determinism requirements

ExactKV verification assumes deterministic generation.

Set:

```python
model.eval()
torch.no_grad()
do_sample = False
num_beams = 1
temperature = None or 0
```

Also consider:

```python
torch.manual_seed(0)
```

But seed alone does not guarantee deterministic GPU kernels.

## Token comparison

Compare token IDs.

Do not compare decoded strings.

```python
if int(draft_token) == int(verifier_token):
    accept
```

## Handling EOS

If verifier predicts EOS:

- EOS is a normal token.
- If draft token matches EOS, accept EOS and stop.
- If draft token differs and verifier token is EOS, commit EOS and stop.
- If draft predicts EOS but verifier does not, reject EOS and continue with verifier correction.

## Handling maximum length

The generator must not exceed `max_new_tokens`.

If accepted tokens plus correction exceed the maximum, truncate carefully.

Recommended:

- Only draft up to remaining token budget.
- Verification should not commit beyond budget.

## Trace format

Each verification round should emit a trace.

```python
@dataclass
class VerificationTrace:
    round_idx: int
    draft_len_requested: int
    drafted_count: int
    accepted_count: int
    rejected_count: int
    first_mismatch_index: int | None
    all_matched: bool
    drafted_tokens: list[int]
    verifier_tokens: list[int]
    accepted_tokens: list[int]
    rejected_tokens: list[int]
    correction_token: int | None
    bonus_token: int | None
```

## Metrics derived from traces

### Acceptance rate

```python
accepted_drafted_tokens / total_drafted_tokens
```

### Average accepted length

```python
mean(accepted_count per round)
```

### Rejection rate

```python
rejected_drafted_tokens / total_drafted_tokens
```

### Mismatch rate

```python
rounds_with_mismatch / total_rounds
```

### First mismatch distribution

Histogram of mismatch positions.

## Testing the verification engine

## Unit test 1: all tokens match

Draft tokens equal verifier tokens.

Expected:

- all_matched = True
- accepted_count = drafted_count
- rejected_tokens = []
- correction_token = None

## Unit test 2: first token mismatch

Draft:

```text
A B C
```

Verifier:

```text
X
```

Expected:

- accepted_tokens = []
- rejected_tokens = [A, B, C]
- correction_token = X
- first_mismatch_index = 0

## Unit test 3: middle mismatch

Draft:

```text
A B C D
```

Verifier:

```text
A B X
```

Expected:

- accepted_tokens = [A, B]
- rejected_tokens = [C, D]
- correction_token = X
- first_mismatch_index = 2

## Unit test 4: EOS handling

Verify that EOS stops generation correctly.

## Unit test 5: exact output equality

Run full generation and ExactKV generation on same prompt.

Expected:

```python
full_ids == exactkv_ids
```

## Debugging tools

Verification should support a debug mode that prints or logs:

- Draft tokens as IDs and strings
- Verifier tokens as IDs and strings
- Accepted tokens
- Correction token
- Output prefix after round

## Common failure modes

### Failure 1: off-by-one verification

The verifier compares each draft token to the wrong full-KV position.

Symptom:

- Very low acceptance rate
- ExactKV output may diverge

Mitigation:

- Test with no-op compressor that returns exact full KV. Acceptance should be 100%.

### Failure 2: incorrect position IDs

Transformer position IDs may be wrong during verification.

Symptom:

- Full baseline differs from step-by-step generation
- Acceptance unexpectedly low

Mitigation:

- Compare custom full-KV generation against `model.generate`.

### Failure 3: cache mutation during failed draft

Rejected draft tokens accidentally remain in cache.

Symptom:

- Later tokens diverge from full baseline

Mitigation:

- Use temporary states.
- Add cache length checks.

### Failure 4: tokenizer mismatch

Different tokenization or prompt formatting causes output mismatch.

Mitigation:

- Use one tokenizer path.
- Store input IDs explicitly.

### Failure 5: nondeterministic kernels

Different runs produce slightly different outputs.

Mitigation:

- Use deterministic settings where possible.
- Start with CPU or stable small GPU tests if needed.
- Compare within same process.

## V1 implementation recommendation

For V1, implement the simplest correct engine:

1. Generate full baseline using custom greedy loop.
2. Generate ExactKV using same custom greedy loop for full verification.
3. Use simple sequential verification.
4. Recompress from full KV after each verification round if necessary.
5. Compare output token IDs.

Do not try to optimize.

## V2 improvement

After V1 works:

- Enable bonus token.
- Avoid recompression where possible.
- Add INT4.
- Improve state updates.
- Add metrics.

## V5 improvement

Later:

- Implement parallel verification.
- Offload full KV.
- Overlap draft and full KV transfer.
- Add scheduler logic.

## Final verification principle

The verification engine is the trust boundary.

Compressed KV can be wrong.

The verifier cannot be wrong.
