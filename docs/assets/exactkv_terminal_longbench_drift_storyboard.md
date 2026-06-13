# Terminal storyboard — LongBench-style drift demo (Exp 037)

**Script:** `scripts/exactkv_terminal_longbench_drift.py`  
**Pacing:** cinematic ~90–120s · fast ~20–30s · `--no-delay` instant

---

## Beat 1 — Cold open (0:00–0:12)

```text
The answer can still score.
The compressed cache can still drift.
```

Pause on both lines.

---

## Beat 2 — Task panel (0:12–0:28)

Panel: **LONGBENCH-STYLE MULTI-DOC QA**

Show three context documents + question: who owns the Friday follow-up?

---

## Beat 3 — Outcome panel (0:28–0:42)

```text
Full KV:      acceptable
Lossy KV:     acceptable
ExactKV:      acceptable
```

Pause — outcome stayed green.

---

## Beat 4 — Behavior panel (0:42–0:55)

```text
Full KV path:  The answer is: Maya
Lossy KV path: The billing migration checkpoint is assigned to Maya. …
```

Explain token path = exact words the model would have generated.

---

## Beat 5 — Drift (0:55–1:10)

Animate lossy draft: `The billing` …

```text
DRIFT DETECTED
Outcome stayed green.
The exact words changed.
```

---

## Beat 6 — ExactKV decision (1:10–1:25)

```text
reject lossy draft
  rejected token: 'billing'
commit full-KV correction
  verifier token: 'answer'
```

---

## Beat 7 — Scoreboard (1:25–1:40)

```text
Task score changed:              no
Compressed KV behavior changed:  yes
ExactKV failures:                0
Final output match:              true
Rejected token committed:        false
```

---

## Beat 8 — Close (1:40–2:00)

```text
Outcome benchmarks ask whether the answer scored well.
ExactKV asks whether compression changed the model's behavior.

KV compression should not be trusted.
It should be crash-tested.
```

---

## Recording notes

- Use `--speed cinematic` for launch reels; `--speed fast` for README GIFs.  
- Do not claim official LongBench evaluation.  
- Pharmacy demo remains primary unless this trace is revalidated on GPU with broader search.
