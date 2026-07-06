# ExactKV Live Demo — Recording Plan

**Goal:** One continuous in-place terminal replay — characters stream into the same three-path table, drift is caught, verifier rejects/commits, output matches full precision.

**Script:** `scripts/exactkv_live_demo.py` (default `--mode stream`)

---

## Record this

```bash
export COLUMNS=110
python3 scripts/exactkv_live_demo.py --speed launch      # ~2–3 min, 2 drifts + long pauses
python3 scripts/exactkv_live_demo.py --speed social      # ~90s tighter cut
```

~2–3 min at `--speed launch` · ~90s at `--speed social` · 2 drifts with pauses

### Flow (single act)

1. **Intro splash** — EXACTKV logo + “CRASH TEST · LIVE VERIFIER REPLAY”
2. Prompt + HUD (decode bar · drift counter · verifier status)
3. Three-path table streams with cursor on active row
4. **Drift 1** — flashing red alert banner → VERIFIER ACTION card (REJECT/COMMIT)
5. Resume stream → **Drift 2** — same beat
6. **Finale** — “WHAT WOULD HAVE SHIPPED” side-by-side + green EXACTKV MATCH banner

**Hold frames:** divergence alert, VERIFIER ACTION card, ship comparison, victory banner.

---

## Optional: case carousel (old style)

```bash
python3 scripts/exactkv_live_demo.py --mode cases --speed cinematic
```

---

## Rehearsal

```bash
python3 scripts/exactkv_live_demo.py --speed fast
python3 scripts/exactkv_live_demo.py --no-delay --plain | less
```

Save MP4 as `docs/assets/exactkv_live_demo.mp4`.
