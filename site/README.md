# ExactKV Landing Page

A fully static, dependency-free landing page for ExactKV. It works by opening
`index.html` directly or by serving the folder.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Single-page site (hero, results, leaderboard, demos, methodology, lineage, prior art, limitations, reproduce, links). |
| `styles.css` | Dark, modern styling. |
| `main.js` | Optional enhancement (nav highlight, footer year). The page is fully functional without JS. |
| `content_manifest.json` | The structured content (sections, leaderboard rows, demo cards) the page renders. |
| `claim_safe_copy.json` | Approved copy + required caveats + forbidden-term list. |

## Run locally

```bash
# Option A: open directly
open site/index.html

# Option B: serve (recommended for relative links to repo docs)
cd /path/to/ExactKV
python3 -m http.server 8000
# then visit http://localhost:8000/site/
```

## Content provenance

Every number on the page traces to on-disk artifacts:

- Headline metrics & leaderboard → `reports/scale_7b/raw.json`, `reports/public_release/leaderboard_final.json`
- Kernel microbenchmark → `reports/phaseF_kernel_benchmark.json`
- Claim boundaries → `docs/CLAIM_BOUNDARIES.md`

## Claim safety

The site is validated by:

```bash
python3 scripts/check_site_claims.py
pytest tests/test_site_artifacts.py -q
```

These check that the hero, leaderboard, and required caveats are present, and
that no forbidden positive claims (production-ready, first-ever, end-to-end
speedup, active GPU memory savings, real SpectralQuant/Shard, beats X) appear.

> Style/layout took loose inspiration from strong ML-systems launch pages
> (e.g. the Shard launch page) for **layout only** — no claims were copied.
