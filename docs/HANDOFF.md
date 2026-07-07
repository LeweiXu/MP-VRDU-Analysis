# Session handoff — 2026-07-06 (afternoon)

Continuation of the earlier 2026-07-06 session. bf16 judging is done and the
tables are built; the open thread is **G2 / Table 3 (InternVL)**, which failed on
a missing dependency and is now being smoke-tested. Everything below is
**uncommitted on `main`**.

## THE LIVE THING: G2 InternVL smoke test (job 1010783)

**Why:** the full G2 run (job 1010307) reported `success` but wrote **zero
predictions** - every InternVL cell died with `ImportError: ... requires timm`.
`timm` was never a declared dependency. It's now in `requirements.txt`
(`timm==1.0.20`) and installed in the Kaya env (you did this).

**Gotcha we hit:** a plain smoke run does NOT test InternVL. `G2Family.model_specs`
returns `()` in smoke mode by design ("smoke has one family, Table 3 reuses G1"),
so the first smoke submit (job 1010756) just did side work and exited in 40s
without loading InternVL. The real test is a **capped full run**, which loads
InternVL but only on a few questions:

```
python -m cli.generate --generation G2_family --full --questions 5 \
    --visual-resolution low --run-tag g2-smoke
```

That's job **1010783** (2xV100, 1h), submitted and RUNNING.

### Watch it
```bash
# blocking: waits, pulls results+logs, tails the log
envs/mpvrdu/bin/python -m kaya.kaya watch 1010783
# or just poll status
ssh kaya 'sacct -j 1010783 -o JobID,State,Elapsed,ExitCode'
# pull whenever
envs/mpvrdu/bin/python -m kaya.kaya pull
```

### What a real pass looks like
After it finishes, pull and check:
```bash
wc -l results/cache/g2-smoke/full/G2_family/predictions.jsonl   # expect ~20 (5 q x 4 rungs, minus any >10-page drop)
cat  results/cache/g2-smoke/full/G2_family/generate_status.json # status: success
grep -c "specs=(side-only)" logs/generate_1010783.out           # want 0 (InternVL must actually run)
grep -ci timm logs/generate_1010783.err                         # want 0
```
Elapsed should be **minutes** (InternVL load takes a few), not 40s. The log should
show InternVL loading and per-cell `done` lines. Note: the false-success bug is
fixed now (see below), so if InternVL fails to load or every cell fails, the job
exits non-zero / writes `status: failed` instead of lying. So `COMPLETED` + a
non-empty `predictions.jsonl` = a genuine pass.

### If the smoke passes -> run the full G2
```bash
envs/mpvrdu/bin/python -m kaya.kaya submit --gres=gpu:v100:2 --mem=24G --time=06:00:00 --no-wait \
  cli/generate.py -- --generation G2_family --full --visual-resolution low --run-tag bf16-lowres --continue-on-error
# then pull, judge (needs a working Gemini key), build:
envs/mpvrdu/bin/python -m kaya.kaya pull
python -m cli.judge --generation G2_family --full --run-tag bf16-lowres --continue-on-error
python -m cli.build --full --run-tag bf16-lowres
```
Once G2 is judged, **table3 (family replication) unblocks** and builds.

### If it fails
Read `logs/generate_1010783.err` / `.out`. Suspects: timm still not in the env
(re-run `kaya run scripts/setup_env.py`), another InternVL dep, or a 2-GPU
sharding OOM. Integration point is `models/internvl.py`.

## bf16 results are done

All bf16 judging finished (G1 1232, G3 836, G5 616 cells; judge said "3 scored, 0
failed"). Tables built under `results/tables/full-bf16-lowres/`:

- Built: table1, table2, table4, table5, table6, table7.
- Skipped: table3 (waiting on G2 judge), table8 (blocked, no G4 scale task).
- **New:** `all_tables_summarised.md` - paper-style compact tables (accuracy % with
  95% CI, frontier in bold), alongside the raw `all_tables.md` and the CSVs.

Headline (table1): frontiers are T / TL / TL (text or text+layout is sufficient at
low res; vision never wins the frontier). table6 (matched vs cross retrieval) now
reports in_between + visual_heavy, and shows text-retrieval (cross) is about as
good as vision-retrieval (matched).

## Code changed this session (all uncommitted)

- `experiments/corpus.py` - drop questions with >10 evidence pages
  (`MAX_EVIDENCE_PAGES`, 7 in the full corpus incl. the 24-page OOM case), applied
  after per-bin sampling so it doesn't perturb the cache.
- `pipeline/judge.py` - two-key Gemini fallback (`GEMINI_API_KEY` ->
  `GEMINI_API_KEY_SECONDARY`), daily-quota 429 fast-fails to switch keys instead
  of burning retries.
- `experiments/driver.py` - a generate task whose every cell is skipped now
  **raises** (writes `status: failed`) instead of false-success.
- `experiments/reporting.py` - **table gate**: a table CSV is written only when
  all its source tasks' generate+judge finished (success status + non-empty
  output); unfinished/blocked tables are skipped and stale CSVs removed. table8
  blocked (no G4). table6 now sources G1+G5 (it needs G1's oracle rows to pick
  vision bins). Writes `all_tables_summarised.md`.
- `experiments/tables.py` - table6 inclusion loosened from "vision is the
  frontier" to "vision materially helps" (best TLV/V beats best T/TL by the
  sufficiency margin, and the bin has data). Paper-style summariser
  (`render_paper_tables_markdown`).
- `requirements.txt` - `timm==1.0.20`.
- Tests updated/added; suite is **86 passed, 1 skipped**.

## Kaya jobs

- `1010783 g2-smoke` - RUNNING, the InternVL smoke (this handoff's focus).
- `1010304 4bit-resume` - RUNNING (~4.5h elapsed), the separate 4-bit full run.
- `1010307 bf16-g2` - COMPLETED but FAILED (0 predictions, missing timm); superseded.
- `1010308 4bit-g2` - CANCELLED earlier so bf16-g2 would run next.
- `1010756 generate` - the no-op smoke (side-only); ignore.

## Judge quota note

Supervisor's key is Tier-1 (billed, ~10k/day) - it briefly dropped to free tier
when the card expired, now restored. Personal free key is only 20/day (kept as a
fallback). The two-key fallback in `pipeline/judge.py` covers both. See
`memory/mpvrdu-judge-quota.md`.

## Open items

1. Watch job 1010783; if it passes, run the full G2 (commands above) -> table3.
2. **Commit this session's work** (it's all uncommitted).
3. 4-bit full run (`1010304`) is a separate track still resuming.
