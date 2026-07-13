# ExactKV Live Demo — Recording Plan

**Hero launch cut (canonical):** see [`launch/demo_hero_10.md`](../../launch/demo_hero_10.md)

**Script:** `scripts/exactkv_live_demo.py`

---

## Record this (hero — default for launch)

```bash
export COLUMNS=110
python3 scripts/exactkv_live_demo.py --speed hero
```

~20–28s terminal · **one** semantic drift (`dropoff` → `pickup`) · scale punch 6%→90%

Save MP4 as `docs/assets/exactkv_hero_terminal.mp4`. Optional Sora cold-open + end card in edit.

### Hero flow

1. Short splash — EXACTKV + crash-test line
2. Pharmacy-style tool JSON streams (three-path table)
3. **Drift** — lossy writes `dropoff`, full KV wants `pickup`
4. VERIFIER ACTION — REJECT / COMMIT
5. WITHOUT vs WITH ExactKV
6. **Scale punch** — same 4× compressor: code ~6% · reading ~90% · 8,132 cells

---

## Longer modes (optional, not hero)

```bash
python3 scripts/exactkv_live_demo.py --speed launch      # ~2–3 min, 4 weather drifts
python3 scripts/exactkv_live_demo.py --speed social      # tighter weather cut
python3 scripts/exactkv_live_demo.py --mode cases --speed cinematic
```

---

## Rehearsal

```bash
python3 scripts/exactkv_live_demo.py --speed hero --no-delay --plain | less
python3 scripts/exactkv_live_demo.py --speed fast
```
