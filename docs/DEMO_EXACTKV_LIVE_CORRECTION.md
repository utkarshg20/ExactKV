# ExactKV Live Correction Demo (V13 Phase 7b)

**Status:** Ready — terminal replay of verified Exp 034 trace.

> This is a **live correctness demo**, not a benchmark.
> The numbers and tokens are from the **Exp 034 trace**, not a timing or memory result.
> This does **not** claim speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, production serving, or model accuracy improvement.
> ExactKV preserves full-greedy output while using lossy KV only as a draft.
> Full-KV verifier remains authoritative. Rejected draft tokens are never committed.

---

## 1. Purpose

Show a recordable terminal demo — like public AI infra demos — where lossy compressed KV proposes a wrong token, ExactKV rejects it, the full-KV verifier commits the correction, and final output matches full greedy.

## 2. How it differs from Exp 034 Markdown report

| Exp 034 report | Live demo (7b) |
|---|---|
| Static Markdown trace | Animated terminal UI |
| Search methodology + tables | Single focused replay for video/asciinema |
| Full experiment metadata | Public-facing “crash test” narrative |

Exp 034 remains the source of truth for the trace. This demo does not replace it.

## 3. How to run

```bash
python3 scripts/demo_exactkv_live_correction.py
```

Options:

| Flag | Effect |
|---|---|
| `--no-delay` | Skip `time.sleep()` animation (tests/CI) |
| `--plain` | Disable ANSI colors |
| `--record-script` | Write/update recording guide at `docs/assets/demo_exactkv_live_correction_script.md` |
| `--trace-json PATH` | Load trace from Exp 034 JSON (default: `reports/experiment_034_killer_correction_demo.json`) |

If the JSON is missing, the script falls back to an embedded fixture copied from the Exp 034 selected demo.

## 4. How to record

See [`assets/demo_exactkv_live_correction_script.md`](assets/demo_exactkv_live_correction_script.md).

**asciinema:**

```bash
asciinema rec -c "python3 scripts/demo_exactkv_live_correction.py" \
  docs/assets/demo_exactkv_live_correction.cast
```

**Tests / CI (no animation):**

```bash
python3 scripts/demo_exactkv_live_correction.py --no-delay --plain
```

Other options: terminalizer, OBS, QuickTime screen recording — details in the recording script asset.

## 5. What the viewer should see

1. Header: `EXACTKV LIVE KV CRASH TEST`
2. Weather JSON tool-call prompt (`tj_002`)
3. Full-KV verifier expects `"metric"`
4. Lossy draft animates to `"}}}` with `}}` highlighted red
5. `REJECT draft token: }}` (red) → `COMMIT verifier token: metric` (green)
6. Comparison table: Full KV / Lossy / ExactKV outputs
7. Status: failures `0`, rejected token not committed, final match `true`
8. Tagline: *Everyone is racing to shrink KV caches. ExactKV tells you when they start lying.*

## 6. Source trace from Exp 034

| Field | Value |
|---|---|
| prompt_id | `tj_002` |
| suite | `tool_json` |
| model | `Qwen/Qwen2.5-0.5B` |
| compressor | `int4_sim` |
| rejected token | `}}` (id 3417) |
| correction token | `metric` (id 15903) |
| exactkv_failures | 0 |
| final output match | true |

Primary source: [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md) and `reports/experiment_034_killer_correction_demo.json` (gitignored).

## 7. Claims allowed

- Lossy KV can draft wrong tokens on this verified trace.
- ExactKV rejects the draft and commits the verifier correction.
- Final ExactKV output matches full greedy on this cell.
- This is a replay of a real Exp 034 observation — no inference during the demo.

## 8. Claims not allowed

- No speedup, throughput, latency, runtime, tokens/sec, or active GPU memory savings.
- No production serving readiness or model accuracy improvement.
- No implication that every compressor or prompt behaves this way.

---

**Related:** [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md) · [`assets/experiment_034_correction_trace.md`](assets/experiment_034_correction_trace.md)
