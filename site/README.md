# ExactKV Site

Single-page narrative site (Shard-inspired blog layout). Open `index.html` directly or serve the folder.

**Public artifact:** [ExactKV research release](https://github.com/utkarshg20/ExactKV/releases/tag/v-release) (git tag `v-release`).

**Public URL (after GitHub Pages deploy):** https://utkarshg20.github.io/ExactKV/

## Preview

```bash
cd /path/to/ExactKV
bash scripts/sync_site_data.sh   # leaderboard + case studies + figure assets
python3 -m http.server 8000
# → http://localhost:8000/site/
```

Or serve from `site/` directly:

```bash
cd site && python3 -m http.server 8000
# → http://localhost:8000/
```

## Deploy (GitHub Pages)

1. In GitHub repo **Settings → Pages**, set **Source** to **GitHub Actions**.
2. Push to `main`, workflow `.github/workflows/deploy-site.yml` runs `sync_site_data.sh` and publishes `site/`.
3. Live at https://utkarshg20.github.io/ExactKV/

Manual deploy trigger: **Actions → Deploy site → Run workflow**.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Long-form story: executive summary, drift findings, figures, case gallery, leaderboard, reproduce. |
| `styles.css` | Dark prose layout + viz bars + tables + hero CTAs. |
| `main.js` | Loads leaderboard + case studies from `data/`. Nav scroll highlight. |
| `data/leaderboard.json` | Copy of public leaderboard (sync via `scripts/sync_site_data.sh`). |
| `data/case_studies.json` | Curated divergence snippets from headline GPU panels. |
| `assets/*.png` | Self-contained figure copies for GitHub Pages. |
| `content_manifest.json` | Structured site metadata (used by tests). |
| `claim_safe_copy.json` | Approved copy + caveats. |

## Claim safety

```bash
python3 scripts/check_site_claims.py
python3 scripts/curate_site_case_studies.py
pytest tests/test_site_artifacts.py -q
```
