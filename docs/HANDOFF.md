# Session handoff — 2026-07-06

End of a long session. Read top to bottom. Durable details are in
`docs/AGENT_GUIDE.md`; how to run is in `docs/USER_GUIDE.md` (local) and
`kaya/KAYA_USER_GUIDE.md` (cluster). **Everything is uncommitted on `main`** — a
large diff (two structural refactors + an OOM fix); review before committing.

## What changed this session

1. **Role-split refactor (the big one).** `experiments/` is now organized by
   *generation task*, not by paper table, and split into a library + thin CLI
   wrappers:
   - `experiments/G1_sufficiency.py` … `G6_classifier.py` — one `GenerationTask`
     per file. **Adding an experiment = drop in a new `G*_*.py` + one line in
     `experiments/registry.py`.** (A `G4` scale task is intentionally not built.)
   - `experiments/base.py` (ABC + cell factories), `registry.py`, `driver.py`
     (the generate+judge engine), `tables.py` (pure builders), `reporting.py`
     (table→source-task routing), `paths.py` (cache layout).
   - Runnable entry points moved to `cli/`: **`cli/generate.py`** (GPU, carries
     the `# kaya:` directives — this is what a cluster submits), **`cli/judge.py`**,
     **`cli/build.py`**. The old `cli/experiments.py` / `cli/build_tables.py` and
     the per-table `T*.py` are gone.
   - 83 tests pass; README, USER_GUIDE, implementation_plan, AGENT_GUIDE, and the
     KAYA guide were all swept to the new commands/layout.

2. **OOM fix: per-cell skip.** A long tail of oracle cells have many gold pages
   (9/10/**24**), and their O(seq²) attention OOMs a V100 *even at
   `--visual-resolution low`*. `experiments/driver.py::generate` now takes
   `skip_failed_cells` (wired to `--continue-on-error`): a failing cell is logged,
   the GPU freed, and the run continues instead of aborting the task. Validated on
   the cluster — bf16 G1 skipped exactly 2/1236 cells (question `mmlongbench:000855`)
   and finished `success`. Details: `docs/AGENT_GUIDE.md` → "Third OOM".

3. **InternVL prestaged.** G2 (family/Table 3) had failed offline because
   `OpenGVLab/InternVL3-8B` wasn't on Kaya. It's now staged (15G under `.cache/`).

## The two full runs (why the OOM/timeout happened, and current state)

Both runs were submitted `--generation all --full`, one 4-bit, one bf16 at
`--visual-resolution low`, 2×V100, 12h, distinct `--run-tag`. The first attempt:
`bf16-lowres` (#1009635) COMPLETED in 10h but G1 OOM'd (partial) and G2 failed
offline; `4bit-current` (#1009634) TIMED OUT at 12h (4-bit is *slower* per cell,
and it ran at full resolution). After the fix + prestage I resubmitted.

**Current cache state (pulled locally, under `results/cache/<tag>/full/<task>/`):**

| task | bf16-lowres | 4bit-current |
|---|---|---|
| G1_sufficiency | **done** (1234/1236, 2 skipped) | resuming now (job 1010304) |
| G2_family (InternVL) | queued (job 1010307) | queued (job 1010308) |
| G3_dataset | done (836) | resuming now (job 1010304) |
| G5_retrieval | done (618 + retrieval side) | pending in 1010304 |
| G6_classifier | done (39 docs) | pending in 1010304 |

**Jobs on Kaya right now (the "3 jobs" you saw):**
- `1010304 4bit-resume` — RUNNING, finishing 4-bit G1/G3/G5/G6 (~10h left).
- `1010307 bf16-g2` — queued, bf16 InternVL (Table 3).
- `1010308 4bit-g2` — queued, 4-bit InternVL (Table 3).
(`1010303 bf16-g1-resume` already COMPLETED — that's why bf16 G1 is done.)

## THE BLOCKER: judging is out of quota (do this first)

**The Gemini free tier is exhausted** (10k requests/day; earlier judging used it
up). It resets ~19h from ~11:05 on 2026-07-06. `.env` has **no `OPENAI_API_KEY`**,
so there is no paid fallback. So no real Table CSVs could be produced this
session. Saved to memory (`mpvrdu-judge-quota`).

To unblock **now**: add `OPENAI_API_KEY=...` to `.env` and judge with
`--judge gpt-4o-mini`. Otherwise wait for the quota reset. Judging is resumable
(`results.jsonl` keyed by cache_key), so re-running only scores new cells.

## Next steps (in order)

1. **Wait for the 3 jobs to finish** (monitor: `python scripts/kaya_status.py` or
   `ssh kaya 'sacct -j 1010304,1010307,1010308 -o JobID,State,Elapsed,ExitCode'`),
   then `kaya.kaya pull`.
2. **Judge + build each run** (needs a working judge key — see blocker):
   ```bash
   set -a; . ./.env; set +a
   # bf16-lowres:
   python -m cli.judge --generation all --full --run-tag bf16-lowres --continue-on-error
   python -m cli.build --full --run-tag bf16-lowres
   # 4bit-current (spec has the -4bit suffix, so judge MUST pass --quantization):
   python -m cli.judge --generation all --full --quantization 4bit --run-tag 4bit-current --continue-on-error
   python -m cli.build --full --run-tag 4bit-current
   ```
   Tables land in `results/tables/full-<tag>/` (8 CSVs + `all_tables.md`).
3. **Compare** the two Table-1 frontiers (4-bit vs bf16-lowres) and the F1 gate:
   ```bash
   python -m cli.gates frontier --table results/tables/full-bf16-lowres/table1_headline.csv \
       --json-output results/gates/F1_bf16.json
   ```

## Deferred / notes

- The 2 skipped many-page cells per run are a documented gap, not a bug. A true
  total-vision-token cap (shrink `max_pixels`/drop pages to a budget) is still
  unimplemented; per-cell skip is the mitigation.
- `4bit-current` may time out again (4-bit is slow); with per-cell skip +
  `--continue-on-error` it caches progress and can be resubmitted to resume.
- Gates F2 (judge–human κ) and F3 (classifier pilot) still pending.
- The partial results from the *previous* session are archived under `temp/`
  (old `T1_headline` cache names, not the new G-task layout).
