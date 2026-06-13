# ExactKV Terminal LongBench-Style Drift Demo

**Experiment:** 037 (V13 Phase 10A)  
**Script:** `scripts/exactkv_terminal_longbench_drift.py`  
**Role:** **Secondary** public demo — complements the pharmacy crash test (Exp 034b).

> This is a **LongBench-style demonstration**, not an official LongBench evaluation.  
> Outcome benchmarks and ExactKV answer different questions.

---

## 1. How to run

```bash
python3 scripts/exactkv_terminal_longbench_drift.py
python3 scripts/exactkv_terminal_longbench_drift.py --speed cinematic
python3 scripts/exactkv_terminal_longbench_drift.py --speed fast
python3 scripts/exactkv_terminal_longbench_drift.py --no-delay --plain   # tests / CI
```

Load trace from search report:

```bash
python3 scripts/exactkv_terminal_longbench_drift.py \
  --source-json reports/experiment_037_longbench_style_drift_candidates.json
```

**Replay only** — no model inference, no fake outputs.

---

## 2. How to record

1. Terminal width ≥ 80 columns; dark background recommended.  
2. Cinematic pacing (~90–120s):

   ```bash
   python3 scripts/exactkv_terminal_longbench_drift.py --speed cinematic
   ```

3. Fast cut (~20–30s): `--speed fast`  
4. Capture with `asciinema`, OBS window capture, or `script` + `cat` replay.  
5. Do **not** present the MP4 pharmacy demo as interchangeable — this demo targets **outcome-green / path-drift** narrative.

---

## 3. Source trace

| Field | Value |
|---|---|
| Prompt | `lb_md_001` — multi-doc QA, Friday follow-up owner |
| Model | `Qwen/Qwen2.5-0.5B` |
| Compressor | `int4_sim` |
| `draft_len` | 4 |
| Rejected | `billing` |
| Correction | `answer` |
| Full KV | `The answer is: Maya` |
| Lossy KV | `The billing migration checkpoint is assigned to Maya.` |
| ExactKV | `The answer is: Maya` |
| `exactkv_failures` | 0 |

Fixture embedded in script; overridden by `selected_demo` in Exp 037 JSON when score ≥ 200.

---

## 4. Why this complements the pharmacy demo

| | Pharmacy (034b) | LongBench-style (037) |
|---|---|---|
| Task | Structured JSON tool call | Multi-doc QA / reference answer |
| Outcome heuristic | Tool-field semantics | Reference answer contains `Maya` |
| Drift shape | `drop` vs `pickup` | `billing` vs `answer` opening |
| Audience hook | “API lied” | “The answer still scores” |

Both show: **compressed KV can drift while ExactKV restores full-KV behavior.**

---

## 5. Allowed claims

- LongBench-**style** task demonstration (not official LongBench).  
- Outcome heuristic stayed green on full, lossy, and ExactKV lanes for this trace.  
- Token paths differed between full KV and lossy KV.  
- ExactKV compares compressed-KV behavior against full-KV behavior token-by-token.  
- `exactkv_failures == 0` and final ExactKV output matches full KV for this trace.  
- ExactKV rejects the lossy draft and commits the full-KV correction.

---

## 6. Forbidden claims

- Official LongBench score or ranking.  
- LongBench is bad, failed, or flawed.  
- Speedup, throughput, latency, tokens/sec, active GPU memory savings.  
- Production serving or model accuracy improvement.  
- Universal guarantee that all LongBench tasks behave this way.

---

## Storyboard

See [`assets/exactkv_terminal_longbench_drift_storyboard.md`](assets/exactkv_terminal_longbench_drift_storyboard.md).
