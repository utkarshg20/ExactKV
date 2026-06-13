# Experiment 037: LongBench-Style Score-Preserving Drift Demo

**Phase:** V13 Phase 10A  
**Status:** PASS — strong candidate found (`lb_md_001` × `int4_sim`)  
**Artifact:** `reports/experiment_037_longbench_style_drift_candidates.json` (gitignored)

> **This is a LongBench-style demonstration, not an official LongBench evaluation.**  
> Outcome benchmarks and ExactKV answer different questions.

---

## 1. Purpose

Find a realistic case where:

- Full-KV and lossy-KV outputs both pass a **transparent task heuristic** (outcome stays green).
- Lossy compressed KV **changed the model's token path** versus full KV.
- ExactKV **rejects** the lossy draft, **commits** the full-KV correction, and final output **exactly matches** full KV with `exactkv_failures == 0`.

This complements the pharmacy structured-output demo (Exp 034b): same crash-test thesis, different task shape (multi-doc QA / outcome scoring).

---

## 2. Search setup

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Device | CPU `float32` (bounded overnight search; RunPod SSH unavailable from dev host) |
| Compressors | `int4_sim`, `k8_v4_sim`, `k8_v4_boundary4_v8_sim` |
| `draft_len` | 4, 8 |
| `max_new_tokens` | 64 |
| Cells searched | **60** (10 prompts × 3 compressors × 2 draft lengths) |
| Score-preserving candidates | **28** (all lanes pass heuristic + ExactKV exact match) |
| Strong candidates (score ≥ 200) | **12** |

Command:

```bash
python3 scripts/search_longbench_style_drift_demo.py --device cpu --dtype float32
```

---

## 3. Prompt categories searched

1. `customer_success_summary` — SSO blocker / renewal risk / Maya follow-up  
2. `meeting_summary` — billing migration / Priya launch owner / September 10  
3. `support_policy_qa` — 30-day refund window  
4. `refund_policy_qa` — purchase-date refund window  
5. `multi_doc_qa` — Friday follow-up owner  
6. `retrieval_copy_fact` — copy owner from ticket  
7. `long_context_ops_summary` — operational weekly log  

Prompts include padded long-context filler for runtime-bounded “long context” feel.

---

## 4. Best candidate

**Selected:** `lb_md_001` × `int4_sim`, `draft_len=4`  
**Category:** `multi_doc_qa`  
**Candidate score:** 320.0  

**Why selected over `lb_pol_001`:** The policy-QA top cell had punctuation-only correction (` Pro` → `.`) and a noisy full-KV multiple-choice tail. `lb_md_001` has a **human-readable semantic opening drift** (`billing` vs `answer`) while both lanes still contain the reference answer **Maya**.

---

## 5. Full KV output

```text
The answer is: Maya
```

---

## 6. Lossy compressed KV output

```text
The billing migration checkpoint is assigned to Maya.
The billing migration checkpoint is assigned to Maya.
…
```

---

## 7. ExactKV output

```text
The answer is: Maya
```

---

## 8. Task heuristic

**Mode:** QA reference answer (`Maya`)

| Lane | Pass | Matched |
|---|---|---|
| Full KV | yes | `Maya` |
| Lossy KV | yes | `Maya` |
| ExactKV | yes | `Maya` |

`outcome_score_changed`: **no** (heuristic)  
`behavior_drifted`: **yes** (token paths differ)

This is **not** official ROUGE or LongBench scoring — a transparent contains-reference check only.

---

## 9. First divergence

- **Index:** 1 (second generated token)  
- **Rejected draft token:** `billing`  
- **Verifier correction:** `answer`  
- **Lossy draft fragment:** `The billing`  
- **Draft tokens:** `The` · `billing` · `migration` · `checkpoint`

---

## 10. Why this is LongBench-style, not official LongBench

- Task shape mirrors long-context QA / summarization panels (multi-document context, reference answer).
- **No** official LongBench dataset split, metric pipeline, or leaderboard submission was run.
- We do **not** claim LongBench failed or is flawed — we show that **outcome-style scoring** and **behavior-level exactness** can diverge.

---

## 11. What it proves

- A lossy KV cache can produce an **answer-compatible** response while changing **the exact words** the same greedy model would have generated under full KV.
- ExactKV detects that drift token-by-token, rejects the lossy draft, and restores full-KV output with `exactkv_failures == 0`.

---

## 12. What it does not prove

- No speedup, throughput, latency, tokens/sec, or active GPU memory savings.  
- No production serving readiness or model accuracy improvement.  
- No claim that LongBench scores are wrong — only that they measure a **different question** than ExactKV.

---

## 13. Limitations

- Single small model (`Qwen2.5-0.5B`), CPU search, 60-cell budget.  
- Heuristic “pass” is not a human rubric or official benchmark score.  
- Lossy lane repeats boilerplate after the opening drift.  
- RunPod GPU rerun not completed (SSH key unavailable from automation host).

---

## 14. Next steps

- Re-run search on RunPod L40S (`--device cuda --dtype float16`) to confirm candidate on GPU.  
- Expand prompt set within bounded budget; prefer semantic opening drift over punctuation-only corrections.  
- Record terminal demo: `docs/EXACTKV_TERMINAL_LONGBENCH_DRIFT.md`.

**Terminal demo (secondary):**

```bash
python3 scripts/exactkv_terminal_longbench_drift.py
python3 scripts/exactkv_terminal_longbench_drift.py --speed fast
python3 scripts/exactkv_terminal_longbench_drift.py --no-delay --plain
```

Primary public demo remains the pharmacy crash test (`scripts/exactkv_terminal_crash_test.py`).
