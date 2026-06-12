# ExactKV Crash-Test Demo — Storyboard

_Generated 2026-06-12T19:07:41.725694+00:00_

**Source trace:** Exp 034 JSON: experiment_034_killer_correction_demo.json
**Scenario:** weather tool JSON (Exp 034 tj_002)
**prompt_id:** `tj_002` · **compressor:** `int4_sim`
**Rejected:** `}}` → **Correction:** `metric`

**Restaurant ordering search:** not found in existing reports; using verified Exp 034 weather trace.

**On-screen phrases:** Compressed KV drafts · Full KV verifies · Wrong token rejected · EXACT MATCH

## Voiceover / caption script

```
Everyone is racing to shrink KV caches.

ExactKV tells you when they start lying.

I wanted to know when they start lying.

Here is a structured-output prompt. One wrong token can change the action.

Full KV gives the trusted output.

Now watch lossy compressed KV.

The response looks plausible, but the compressed cache changes the next token.

ExactKV runs this differently.

Compressed KV drafts. Full KV verifies.

The moment the compressed cache disagrees, ExactKV rejects the draft token and commits the full-KV correction.

Final output matches full KV exactly.

In V13, span verification passed a 600-cell exactness grid with 0 sequential failures, 0 span failures, and 0 parity failures.

Llama-3.1-8B also passed a small suite with 0 failures and mean acceptance around 0.945.

It is not fast yet. Full greedy is still faster today, and active GPU memory savings are not claimed.

But the verifier is working.

KV compression should not be trusted.

It should be crash-tested.
```

## Scenes

| Time | Scene | Screen | Caption / VO | Source | Visual |
| --- | --- | --- | --- | --- | --- |
| 0–5s | Cold open | Scene 1 | Everyone is racing to shrink KV caches. | tagline | Black/dark, large type |
| 5–15s | Prompt | Scene 2 | Structured output is where one token matters. | Exp 034 JSON: experiment_034_killer_correction_demo.json | Editor style prompt |
| 15–30s | Full KV | Scene 3 | Full KV gives the trusted answer. | Exp 034 JSON: experiment_034_killer_correction_demo.json | Highlight "metric" green |
| 30–45s | Lossy KV | Scene 4 | Compressed cache changes the next token. | Exp 034 JSON: experiment_034_killer_correction_demo.json | Highlight }} red |
| 45–60s | First divergence | Scene 5 | First divergence: token 1. | Exp 034 JSON: experiment_034_killer_correction_demo.json | Token diff zoom |
| 60–80s | ExactKV trace | Scene 6 | Wrong token rejected. Verifier token committed. | Exp 034 JSON: experiment_034_killer_correction_demo.json | Red cross → green commit |
| 80–95s | Exact match | Scene 7 | Final output matches full KV exactly. | Exp 034 JSON: experiment_034_killer_correction_demo.json | EXACT MATCH badge |
| 95–110s | V13 proof cards | Scene 8 | 600-cell span grid; Llama 8B; SnapKV smoke. | Exp 029/033/032b docs | Proof cards |
| 110–115s | Honest status | Scene 9 | Not faster yet. Correctness first. | Exp 030/031 docs | Amber honesty card |
| 115–120s | Final title | Scene 10 | KV compression should be crash-tested. | tagline | Title card |
