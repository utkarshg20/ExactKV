# ExactKV — X / Twitter

> Claim-safe. Numbers from release artifacts. No "beats X", no speedup/VRAM claims.

---

## Main tweet

Everyone reports average quality on compressed KV caches.

ExactKV reports the **first wrong token** — draft from compressed KV, verify against full-KV greedy, log where they split.

Crash-test framework + public leaderboard.

---

## Thread — follow-up 1

8,132 GPU cells. `exactkv_failures = 0`.

Llama-3.1-8B + Mistral-7B · LongBench, BFCL, MBPP, RULER, HumanEval.

Same int4 compressor: **6%** drift on code → **90%** on reading. Longer generation = **7×** more drift on tool-calling. Three distinct failure modes — same verifier catches all of them.

Despite 50% token drift: **106/106** valid BFCL tool calls preserved.

---

## Thread — follow-up 2

Research eval only — not production serving, not VeriCache, not throughput/VRAM claims. External panels measure drift, not official benchmark scores.

Technical report + repro in repo:
[paper link]
[site/leaderboard link]
