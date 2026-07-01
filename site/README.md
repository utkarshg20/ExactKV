# ExactKV Site

Single-page narrative site (Shard-inspired blog layout). Open `index.html` directly or serve the folder.

## Preview

```bash
cd /path/to/ExactKV
python3 -m http.server 8000
# → http://localhost:8000/site/
```

Or: `open site/index.html`

## Files

| File | Purpose |
|------|---------|
| `index.html` | Long-form story: TL;DR, drift findings, tables, leaderboard, reproduce. |
| `styles.css` | Dark prose layout + viz bars + tables. |
| `main.js` | Footer year (optional). |
| `content_manifest.json` | Structured leaderboard/demo metadata (used by tests). |
| `claim_safe_copy.json` | Approved copy + caveats. |

## Claim safety

```bash
python3 scripts/check_site_claims.py
pytest tests/test_site_artifacts.py -q
```
