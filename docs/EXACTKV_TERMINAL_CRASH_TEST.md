# ExactKV Terminal Crash-Test Demo

**Status:** Complete — terminal-native public demo (V13 Phase 8e)

> This is a **terminal-native public demo**, not a benchmark.
> It replays a **real Exp 034b semantic correction trace** (`pharm_001` × `k8_v4_sim`: `drop` → `pickup`).
> No speedup, throughput, latency, runtime, tokens/sec, active GPU memory savings, production serving, or model accuracy improvement claim is made.
> ExactKV preserves full-greedy output while using lossy KV only as a draft.
> Full-KV verifier remains authoritative. Rejected draft tokens are never committed.

---

## 1. Purpose

Provide a **live, screen-recordable terminal dashboard** that shows how lossy compressed KV can change structured tool-call output — and how ExactKV rejects the wrong draft token, commits the verifier correction, and still matches full greedy exactly.

This is the **primary public demo artifact**. Run it in Terminal, record with QuickTime/OBS/asciinema.

## 2. Why this is different from the MP4 renderer

| | Terminal demo | MP4 renderer (`render_exactkv_crash_test_video.py`) |
|---|---|---|
| **Artifact** | Live Python program you run and record | Pre-rendered animation |
| **Feel** | Live infra dashboard (panels, progress, pauses) | Cinematic slides |
| **Inference** | None in default replay mode | None |
| **Primary use** | README hero, conference booth, social clips | Optional rendered share asset |

The MP4 remains an optional rendered artifact — not the main demo.

## 3. Why this beats `'}}' → 'metric'`

Exp 034 selected `tj_002` where lossy KV proposed `'}}'` and the verifier corrected to `'metric'`. That trace is **real** but **tokenization-weird**: viewers need BPE intuition to understand why `}}` is wrong.

Exp 034b found a **semantic** correction on a pharmacy tool-call JSON:

| | Lossy draft | Verifier correction |
|---|---|---|
| Token | `drop` | `pickup` |
| Meaning | tries `dropoff` fulfillment | commits `pickup_date` path |

A nontechnical viewer immediately sees: **compressed KV tried dropoff instead of pickup**.

If no semantic winner existed, the demo would honestly fall back to Exp 034 `tj_002`.

## 4. How to run live

```bash
python3 scripts/exactkv_terminal_crash_test.py
```

| Flag | Effect |
|---|---|
| `--speed fast` | ~20–30 second pacing |
| `--speed cinematic` | ~90–120 second pacing |
| `--no-delay --plain` | Instant, no color (tests/CI) |
| `--source-json PATH` | Load trace from Exp 034b or Exp 034 JSON |

Examples:

```bash
python3 scripts/exactkv_terminal_crash_test.py --speed cinematic
python3 scripts/exactkv_terminal_crash_test.py --source-json reports/experiment_034_killer_correction_demo.json
```

**Recommended terminal:** 120×36 or larger, monospace 16–20 pt.

## 5. How to record

### asciinema

```bash
asciinema rec -c "python3 scripts/exactkv_terminal_crash_test.py --speed cinematic" \
  docs/assets/exactkv_terminal_crash_test.cast
```

### QuickTime (macOS)

1. Terminal → full screen, dark background, large monospace font.
2. QuickTime → File → New Screen Recording.
3. Run `python3 scripts/exactkv_terminal_crash_test.py --speed cinematic`.

### OBS

1. Add Window Capture (Terminal) or Display Capture.
2. 1920×1080 canvas; crop to terminal frame.
3. Start recording, then run the demo command.

## 6. What viewers should see

1. **Header** — EXACTKV CRASH TEST tagline.
2. **Prompt panel** — pharmacy JSON tool call with `pickup` field.
3. **Drafter + verifier panels** — lossy `int4_sim`/`k8_v4_sim` draft vs full FP KV.
4. **Token stream** — lossy output types until `drop` appears → **DRIFT DETECTED**.
5. **Decision panel** — `drop` **REJECTED**, `pickup` **COMMITTED** (red/green).
6. **Three-lane comparison** — Full KV / Lossy only / ExactKV.
7. **Scoreboard** — failures 0, exact match TRUE.
8. **V13 proof strip** — honest timing/memory disclaimers.
9. **Closing** — “KV compression should not be trusted. It should be crash-tested.”

## 7. Source trace

| Field | Value |
|---|---|
| Search | [`EXPERIMENT_034B_SEMANTIC_CORRECTION_SEARCH.md`](EXPERIMENT_034B_SEMANTIC_CORRECTION_SEARCH.md) |
| prompt_id | `pharm_001` |
| suite | `crafted_pharmacy` |
| compressor | `k8_v4_sim` |
| draft_len | 4 |
| rejected | `drop` |
| correction | `pickup` |
| exactkv_failures | 0 |
| final output match | true |

Embedded fixture in `scripts/exactkv_terminal_crash_test.py` when JSON reports are absent.

## 8. Allowed claims

- ExactKV preserved full-greedy output on this trace (`exactkv_failures == 0`).
- Lossy compressed KV proposed a semantically wrong token; ExactKV rejected it.
- Full-KV verifier remained authoritative.
- This is a correctness replay demo, not a performance benchmark.

## 9. Forbidden claims

- Speedup, throughput, latency, runtime improvement, tokens/sec.
- Active GPU memory savings or VRAM reduction.
- Production serving readiness.
- Model accuracy improvement.
- Shard / SpectralQuant ExactKV results before integration.

## 10. Troubleshooting

| Issue | Fix |
|---|---|
| Colors garbled | Use `--plain` or a true-color terminal |
| Layout wraps | Widen terminal (≥120 cols) |
| Too slow | `--speed fast` or `--no-delay` |
| Wrong trace | Pass `--source-json` explicitly |

**Related:** [`EXACTKV_CRASH_TEST_VIDEO.md`](EXACTKV_CRASH_TEST_VIDEO.md) (optional MP4) · [`DEMO_EXACTKV_LIVE_CORRECTION.md`](DEMO_EXACTKV_LIVE_CORRECTION.md) (Phase 7b simpler replay)
