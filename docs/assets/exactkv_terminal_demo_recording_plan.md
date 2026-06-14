# ExactKV Terminal Demo Recording Plan (Phase 9C)

**Goal:** Capture a watchable terminal demo without faking an MP4 in validation scripts.

**Recommended output:** `docs/assets/exactkv_crash_test_demo.mp4` (encode from terminal capture or use existing rendered video pipeline).

Companion: [`../EXACTKV_CRASH_TEST_VIDEO.md`](../EXACTKV_CRASH_TEST_VIDEO.md) · [`../../scripts/record_terminal_demo.sh`](../../scripts/record_terminal_demo.sh)

---

## 1. Recommended terminal size

| Setting | Value |
|---|---|
| Columns | **100–120** (`export COLUMNS=110`) |
| Rows | **32–40** |
| Font | Monospace, 14–18 pt (readable in 1080p crop) |
| Theme | Dark background, high contrast (green/yellow/red semantic colors show well) |

Test layout before recording:

```bash
python3 scripts/exactkv_terminal_crash_test.py --no-delay --plain | head -40
```

---

## 2. Command to run

### Primary (crash-test narrative)

```bash
python3 scripts/exactkv_terminal_crash_test.py --speed cinematic --plain
```

### Fast rehearsal (no recording)

```bash
python3 scripts/exactkv_terminal_crash_test.py --no-delay --plain
```

### Optional second beat (LongBench-style drift)

```bash
python3 scripts/exactkv_terminal_longbench_drift.py --speed cinematic --plain
```

### Helper script (transcript / asciinema — not MP4)

```bash
bash scripts/record_terminal_demo.sh
```

---

## 3. Suggested pacing mode

| Mode | Flag | Use |
|---|---|---|
| **Cinematic** | `--speed cinematic` | **Recording** — ~90–120s with pauses on drift/reject/commit |
| Fast | `--speed fast` | Rehearsal |
| Instant | `--no-delay` | CI / smoke only |

---

## 4. Exact lines that should appear

### Crash-test demo (minimum must-see)

```text
EXACTKV CRASH TEST
Everyone is racing to shrink KV caches.
ExactKV tells you when they start lying.
DRIFT DETECTED
REJECTED
COMMITTED
exactkv_failures: 0
KV compression should not be trusted.
It should be crash-tested.
```

Semantic variant (Exp 034b pharm trace) should show **drop** → **pickup** correction when that trace is selected.

### LongBench-style demo (optional clip)

```text
Outcome benchmarks ask whether the answer scored well.
ExactKV asks whether compression changed the model's behavior.
exactkv_failures: 0
```

---

## 5. What makes the demo good

- Viewer sees **lossy draft propose wrong token** before final output looks fine.
- **DRIFT DETECTED** pause is visible (cinematic speed).
- **REJECTED** (red) vs **COMMITTED** (green) contrast is readable.
- Closing tagline lands: *"KV compression should not be trusted. It should be crash-tested."*
- `exactkv_failures: 0` visible on scoreboard strip.
- No speed/memory/serving claims on screen.

---

## 6. What would make the demo fail

| Problem | Symptom |
|---|---|
| Terminal too narrow | Box drawing wraps / truncates |
| `--no-delay` used for final MP4 | No dramatic pause; looks like debug spam |
| Wrong trace / invented tokens | Mismatch with Exp 034/034b JSON sources |
| Missing install | `ModuleNotFoundError: exactkv` |
| Overclaim narration | Any speedup / VRAM savings / production-ready voice-over |
| Fake MP4 in validation | Empty or static file passed off as recording |

---

## 7. Recording methods (pick one)

| Method | Notes |
|---|---|
| **asciinema** | `bash scripts/record_terminal_demo.sh` → `.cast` file; convert with agg/svg if needed |
| **`script` transcript** | Fallback `.typescript` log; not video but auditable |
| **Screen capture** | OBS / QuickTime on terminal window — human step, not automated in CI |
| **Rendered video** | `python3 scripts/render_exactkv_crash_test_video.py` for polished MP4 from verified trace |

Phase 9C does **not** require a new MP4 — only that replay commands pass and this plan exists.

---

## 8. Post-recording checklist

- [ ] MP4 plays in browser / social preview
- [ ] Drift + reject + commit legible at 1080p
- [ ] No forbidden claims in on-screen text or captions
- [ ] File saved as `docs/assets/exactkv_crash_test_demo.mp4`
- [ ] Optional: update [`EXACTKV_CRASH_TEST_VIDEO.md`](../EXACTKV_CRASH_TEST_VIDEO.md) with capture date
