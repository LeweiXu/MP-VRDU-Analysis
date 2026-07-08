# Phase 4 autonomous build runbook

This is the single source of truth for the unattended overnight build. A session
cron fires every 5 minutes and follows this file. Full permission was granted by
the user on the night of 2026-07-08; the PC is left running. When this file and
memory disagree with an older doc, this file wins for the autonomous run.

## How it works

- Stage progress ground truth: `envs/mpvrdu/bin/python -m pytest -q`.
- Per-file reuse map + stage detail: `docs/phase4_plan.md`.
- Decisions log: append one line per stage to `docs/DECISIONS.md`.

## Remaining stages (dependency order)

- **Stage 5** - models + pipeline + orchestrator + engine lifecycle. No direct
  red tests; validate at import/registry level only (no GPU here). Rename
  `local_vlm -> qwen3vl`, drop `_truncate_context`, add prefill/decode + vram
  capture, lift judges/conditioner/reasoner, port orchestrator (ResultRow from
  schema, keys from engine/paths), lift the driver lifecycle half + systemic abort.
- **Stage 6** - tasks + registry + yaml. Greens `test_imports_registry` (2) +
  `test_yaml_spec` (2). Four `G[num]_[name]` tasks, registry lists exactly them,
  `parse_spec` rejects a `machine` field.
- **Stage 7** - scoring + reporting. Greens `test_io_fixtures` (2). Port
  `metrics/ -> scoring/`, salvage kappa/frontier from `gates/`, rewrite table
  routing to v4 names + build-time routing; add the v4 jsonl reader
  (`experiments.engine.driver.read_rows`) + `reporting.build.group_rows`.
- **Stage 8** - ops entry points + reworked scripts. No failing tests; keep the
  docstrings test green.

## Rules (hold every fire)

- Lift/adapt from `old/` exactly where `docs/phase4_plan.md` says. Compliant
  docstrings: current function only, 1-3 sentences, no roadmap/pivot/RQ/table
  refs, no `v3`/`v4`, no em-dashes, casual tone.
- Frozen interfaces (schema contracts, ModelInput, Reasoner/Judge/Retriever ABCs,
  cell keys, two-cache design) stay stable; changes are recorded, not silent.
- Revert any change that regresses a previously-green test; only advance when the
  stage's target tests pass AND nothing regressed.
- **Commit after each completed stage** on the current branch:
  `git add -A && git commit -m "stage <num>"`. Plain message only, no signature,
  no Claude co-author trailer. Never `git push`. Never delete `old/`. Never guess
  on science.

## Failsafes (after the 3:35am usage reset, and always)

1. **Re-establish true state first.** Each fire: read this file, then run pytest.
   Do not assume in-memory state (the session may have restarted across the reset).
2. **No-progress guard.** Append one line per fire to `logs/phase4_auto.log`:
   `<iso-timestamp> red=<N> action=<what>`. Before acting, read the last 3 lines;
   if the red count has not decreased across the last 3 fires and no stage was
   completed, STOP: write a clear `BLOCKED` note in the Fire log below and take no
   further code action (let the next human look).
3. **One stage per fire.** Never run two heavy steps concurrently.
4. **Hard don'ts.** No `git push`, no deleting `old/`, no touching a real run's
   `results/`. (Committing per stage with `git commit -m "stage <num>"` is
   required, not forbidden.)
5. **Broken harness.** If pytest itself errors at collection or the env is broken,
   STOP and flag here; do not keep firing edits.
6. **Completion.** When all tests pass, first build **Stage 8** if it is missing
   (`ops/generate.py`, `ops/judge.py`, `ops/build.py` entry points + the driver
   generate/judge task-loop with the parse pre-pass, reasoner load/free, and
   systemic-abort; the reporting table builders as needed). Stage 8 and the driver
   loop have NO unit tests, so the smoke test is their acceptance: commit Stage 8
   as `git commit -m "stage 8"` once it is written. Then run the local smoke test
   below exactly once, guarded by the marker `results/phase4_smoke_done.txt`. On
   success write the marker, append a summary here + in DECISIONS.md, commit
   (`git commit -m "smoke test"`), then `CronList` + `CronDelete` this job and stop
   permanently.
7. **Budget.** After the reset it is fine to spend budget finishing stages and the
   smoke test, but once the smoke marker exists, stop for good.

## Post-completion local smoke test (user instruction, full permission)

Goal: prove the `.jsonl` result cells generate correctly end to end, locally,
across all four generation tasks, on a very small MMLongBench-Doc subset, using
**qwen3vl-2b 4-bit quantized** at **minimum visual resolution**. This is a
plumbing proof (do cells generate valid rows), not a science run.

1. **Local env.** Ensure a local GPU env exists (`envs/mpvrdu-local-gpu`, or build
   per `docs/requirements/README.md` local / sm_120). Install if missing (full
   permission to configure/install).
2. **Local prestage.** Stage qwen3-vl-2b + a tiny data slice locally, e.g.
   `python -m ops.kaya.kaya`-free direct run of `ops/scripts/prestage.py --local
   --smoke` or targeted `--model-id Qwen/Qwen3-VL-2B-Instruct`. MMLongBench is
   already under `.data/` locally; confirm.
3. **Run each generation task** (`G1_oracle_ladder`, `G2_retrieval`,
   `G3_hallucination`, `G4_classifier_pricing`) on a tiny subset
   (`corpus: {sampling: {limit: 2}}` or 2-3 questions) with:
   - reasoner spec `qwen3vl-2b-local`, `quantization="4bit"`
   - `visual_resolution="min"`  (local testing override; the global default stays
     `med` pending the resolution probe, do NOT change `DEPLOYMENT_RESOLUTION`).
4. **Verify** each task writes well-formed result cells: one jsonl row per cell,
   valid `ResultRow` JSON, telemetry populated, `status` set. A cell that fails
   (e.g. a TL/TLV parser cell if the local parser env is not wired) is fine as
   long as it records a `status=error/oom` row with `skipped_reason` (that proves
   the failure-row path). The bar is: cells GENERATE rows, ok or failed, not that
   answers are correct.
5. **Record** the outcome here + in `docs/DECISIONS.md`; write
   `results/phase4_smoke_done.txt` on success.

## Fire log

_(cron appends one line per fire: timestamp, red count, action taken)_
