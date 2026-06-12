# ExactKV Crash-Test Video (V13 Phase 8c)

**Status:** Watchable artifact generated from verified trace.

> This is a **cinematic correctness demo**, not a benchmark.
> Trace tokens are from **Exp 034 JSON: experiment_034_killer_correction_demo.json** — not invented.
> No speedup, throughput, latency, tokens/sec, active GPU memory savings, production serving, or model accuracy improvement claim is made.

---

## 1. Purpose

90–120 second launch-quality video showing lossy KV proposing a wrong token, ExactKV rejecting it, committing the verifier correction, and matching full greedy — a crash-test narrative, not a debug table.

## 2. Why static cards were not enough

Phase 8b PNG cards are useful for README and threads, but a **watchable video** is required for launch posts and social distribution.

## 3. Source trace

| Field | Value |
| --- | --- |
| Scenario | weather tool JSON (Exp 034 tj_002) |
| prompt_id | `tj_002` |
| compressor | `int4_sim` |
| Rejected token | `}}` |
| Correction token | `metric` |
| exactkv_failures | 0 |
| final match | true |

**Restaurant ordering trace:** not found in existing reports without new model search. **Fallback:** verified Exp 034 weather tool JSON (`tj_002` × `int4_sim`).

## 4. How to render

```bash
# Full quality (~105s, 1920×1080)
python3 scripts/render_exactkv_crash_test_video.py

# Quick preview
python3 scripts/render_exactkv_crash_test_video.py --fast

# Frames + storyboard only
python3 scripts/render_exactkv_crash_test_video.py --fast --no-video
```

Options: `--source-json PATH`, `--fps N`, `--width W`, `--height H`

## 5. How to view

Open `docs/assets/exactkv_crash_test_demo.mp4` or `exactkv_crash_test_demo.html` in a browser.

## 6. How to record/share

- Upload MP4 to X/LinkedIn/YouTube
- Embed HTML player page for docs site
- GIF: `docs/assets/exactkv_crash_test_demo.gif` for lightweight sharing

## 7. Generated artifacts

- `docs/assets/exactkv_crash_test_storyboard.md`
- `docs/assets/exactkv_crash_test_frames/`
- `docs/assets/exactkv_crash_test_demo.html`

## 8. Allowed claims

- Lossy KV drafted wrong token on this verified trace; ExactKV corrected.
- `exactkv_failures == 0` on shown V13 panels (Exp 029/033/032b cited in video).
- Exp 030/031 honesty framing (slower, no VRAM savings).

## 9. Forbidden claims

- Speedup, throughput, latency, tokens/sec, VRAM savings.
- Production serving or model accuracy improvement.
- Shard/SpectralQuant as ExactKV results.

## 10. Next steps

**Proceed to Phase 9** launch package with this video linked from README.

---

**Related:** [`EXPERIMENT_034_KILLER_CORRECTION_DEMO.md`](EXPERIMENT_034_KILLER_CORRECTION_DEMO.md) · [`DEMO_EXACTKV_LIVE_CORRECTION.md`](DEMO_EXACTKV_LIVE_CORRECTION.md) · [`PUBLIC_VISUAL_PACKAGE.md`](PUBLIC_VISUAL_PACKAGE.md)
