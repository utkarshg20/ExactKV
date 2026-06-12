# ExactKV Terminal Crash-Test — Live Storyboard

**Program:** `scripts/exactkv_terminal_crash_test.py`  
**Source trace:** Exp 034b `pharm_001` × `k8_v4_sim` (`drop` → `pickup`)  
**Fallback:** Exp 034 `tj_002` × `int4_sim` (`}}` → `metric`)

Record with: `asciinema`, QuickTime, or OBS. Target length: **75–120s** (cinematic), **20–30s** (fast).

---

## Scene map

| Time (cinematic) | Section | On screen | Pause / beat |
|---|---|---|---|
| 0–8s | Cold open | Header box + tagline | Hold tagline 2s |
| 8–18s | Prompt | Pharmacy tool-call JSON panel | Let viewer read `pickup` field |
| 18–28s | Drafter | `k8_v4_sim` panel + empty progress bar | “proposing tokens…” |
| 28–35s | Verifier | Full FP KV checking | Brief wait |
| 35–55s | Token stream | `true,"drop` types out; freeze on `drop` | **Dramatic pause** |
| 55–65s | Drift | `DRIFT DETECTED` + dropoff vs pickup message | **Long pause** |
| 65–75s | Decision | REJECTED `drop` / COMMITTED `pickup` | Red/green highlight |
| 75–90s | Comparison | Full / Lossy / ExactKV lanes | Side-by-side read |
| 90–100s | Scoreboard | failures 0, match TRUE | Badge moment |
| 100–110s | Proof strip | V13 cells + honesty lines | No speed/VRAM claims |
| 110–120s | Close | “KV compression should not be trusted…” | Fade to end |

---

## Recording commands

```bash
# Cinematic (recommended for launch clip)
python3 scripts/exactkv_terminal_crash_test.py --speed cinematic

# asciinema
asciinema rec -c "python3 scripts/exactkv_terminal_crash_test.py --speed cinematic" \
  docs/assets/exactkv_terminal_crash_test.cast

# CI / instant
python3 scripts/exactkv_terminal_crash_test.py --no-delay --plain
```

---

## Visual notes

- Monospace font, 16–20 pt, dark background.
- Terminal width ≥ **120 columns**.
- Red = wrong draft token; green = verifier correction; yellow/amber = drift warning.
- No raw token IDs on screen — text only.

---

## Claims boundary

Terminal replay of verified search output. **Not** a timing or memory benchmark.
