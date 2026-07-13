# Wave-3 release checklist

Use this after RunPod backfill reports **576/576** and `rebuild_wave3_panels.py` exits 0.

---

## 1. Lock artifacts

- [ ] `python3 scripts/rebuild_wave3_panels.py --dir reports/external_panels/faithful/wave3 --write`
- [ ] `python3 scripts/integrate_faithful_panel_results.py --dir reports/external_panels/faithful/wave3 --write`
- [ ] Confirm **576/576 ok** and **exactkv_failures: 0** in `summary.md`
- [ ] Remove `reports/external_panels/faithful/wave3/` from `.gitignore` and commit wave-3 JSON + summaries
- [ ] `python3 scripts/validate_external_panel_artifacts.py` (if applicable to faithful panels)

---

## 2. Paper

- [ ] Add **§6.17.2 Wave-3 faithful panel** (int8 + TurboQuant, full grid, both models)
- [ ] Tables: per-family divergence + acceptance; overall rollup
- [ ] Update abstract limitations row (wave-3 complete, not pending)
- [ ] `bash scripts/build_paper_pdf.sh` — rebuild PDF
- [ ] Bump report version note in `docs/VERSION_LINEAGE.md`

---

## 3. Site

- [ ] `bash scripts/sync_site_data.sh` — leaderboard + case studies
- [ ] Add wave-3 row/section to `site/index.html` (appendix framing, not headline)
- [ ] `python3 scripts/check_site_claims.py` — must pass
- [ ] `pytest tests/test_site_artifacts.py -q`

---

## 4. README / RELEASE

- [ ] Update `README.md` faithful appendix line (wave-1 + wave-2 + **wave-3 576**)
- [ ] Update `RELEASE.md` Phase D3 / appendix paragraph
- [ ] Regenerate `reports/public_release/` if wave-3 merges into public leaderboard JSON

---

## 5. Launch copy (local only — `launch/` is gitignored)

- [ ] `launch/x_thread.md` — wave-3 LongBench TurboQuant story
- [ ] `launch/linkedin_post.md` — same numbers, claim-safe caveats
- [ ] `launch/short_announcement.md` — appendix bullet
- [ ] Record hero demo: `export COLUMNS=110 && python3 scripts/exactkv_live_demo.py --speed hero` (see `launch/demo_hero_10.md`)

---

## 6. Validators (must pass before tag)

```bash
python3 scripts/check_no_secrets.py
python3 scripts/audit_public_claims.py
python3 scripts/check_site_claims.py
python3 scripts/validate_final_release_package.py
pytest tests/test_final_release_package.py tests/test_site_artifacts.py tests/test_exactkv_live_demo.py -q
```

---

## 7. GitHub release

- [ ] Commit wave-3 artifacts + paper/site/README updates
- [ ] Tag (e.g. `v-release-wave3` or bump existing research preview tag)
- [ ] GitHub release notes: wave-3 appendix complete, link to `reports/external_panels/faithful/wave3/summary.md`
- [ ] Push Pages / verify `https://utkarshg20.github.io/ExactKV/`

---

## Do not

- Pull from RunPod mid-backfill (risk clobbering Llama 144/144)
- Re-run `run_wave3_longbench_backfill.sh` on Llama
- Publish wave-3 headline claims before 576/576
- Commit `launch/` or `release_synthesis/` (gitignored, local only)
