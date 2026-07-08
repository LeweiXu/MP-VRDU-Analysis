# Session handoff - 2026-07-08

This session was a documentation + code refactor to make future experiments
easier to navigate. No pipeline behavior changed; everything is local working-tree
edits (nothing committed, nothing pushed to Kaya). Full test suite is green:
**99 passed, 1 skipped**.

## Cluster job (from the previous session, not touched here)

The full G1/G2/G5 rerun (`1012121`, run tag `yaml-g1-g2-g5-rerun`, spec
`specs/g1_g5_rerun.yaml`, 2x V100) was still running at the last handoff. This
session did **not** push to Kaya, so the remote job runs from its own synced
snapshot and is unaffected by these local edits. Job state was not re-checked this
session (watcher session `18442`, logs under `logs/g1g2g5-full_1012121.*`). After it
finishes, score and build:

```bash
python -m cli.judge --spec specs/g1_g5_rerun.yaml --judge <judge>
python -m cli.build --run-tag yaml-g1-g2-g5-rerun --bootstrap 1000
```

Note the judge CLI is now `--spec` (or `--full --run-tag`); the deprecated
`--generation` flag was removed (see below).

## Documentation changes

- **`docs/PROJECT_SPEC.md`** rewritten from the README (it was the stale original
  spec). Corrected: judge is Gemini-2.5-Flash default / GPT-4o-mini paid; retrievers
  are `bge-small-en-v1.5` + `colqwen2.5-v0.2`; CIs are document-level bootstrap;
  Table 4 is a held-out MMLongBench subset (not LongDocURL); 70 text-heavy docs (not
  54); plus OCR routing and per-bin sampling.
- **`docs/AGENT_GUIDE.md`** cut ~50% (dropped the build log and the blow-by-blow OOM
  saga; kept decisions, frozen interfaces, caching contract, and the models/data/
  tools/evaluation reference). Recorded the Phase-A checkpoint (shared side-artifact
  writers).
- **`docs/USER_GUIDE.md`** §1-10 refreshed to match PROJECT_SPEC (same fact fixes:
  70 docs, bge-small, Gemini default, document-level bootstrap, held-out Table 4, and
  the V100 hardware-limits risk replacing the resolved transformers-availability one).
- **Dead `implementation_plan.md` refs** cleaned out after the user deleted the file:
  the real breakage was `config.py::project_root` (root marker is now `README.md` +
  `config.py`); also fixed `CLAUDE.md` and `USER_GUIDE.md`. Same for the deleted
  `SINGLE_GPU_8B_FEASIBILITY.md` (two code comments repointed to `AGENT_GUIDE.md`).

## Code changes (refactor)

- **Deleted dead code:** `scripts/attn_probe.py`, `scripts/single_gpu_probe.py`
  (zero-reference probes), `models/api_vlm.py` (empty placeholder).
- **Side-artifact de-duplication:** new `experiments/side_artifacts.py`
  (`write_retrieval_eval` / `write_classifier_eval`). `G5`, `G6`, and the YAML task
  all call it now, so the two task systems can't drift. Emission order preserved, so
  existing `retrieval.jsonl` / `classifier.jsonl` caches stay byte-identical.
- **Removed the deprecated `--generation` CLI path** from `cli/generate.py` and
  `cli/judge.py` (YAML `--spec` is the only entry point now). `generate.py` lost its
  now-dead config flags; the driver's `run_generate`/`run_judge`/`judge` functions
  were kept (still test-covered engine internals).
- **Reorganized `experiments/` into top-level packages** (it now holds generation
  only):
  - `reporting/` = `tables/` (per-table builders) + `build.py` (table -> source-task
    routing).
  - `gates/` = `core.py` (Section-2 gate logic) + `viewer.py` (the shared cached-cell
    viewer, moved out of `experiments/inspect.py`) + `__main__.py`. The gate CLI is
    now **`python -m gates`** (was `python -m scripts.gates`).
  - `scripts/inspect_results.py` stays as the debug CLI, importing `gates.viewer`.
- **Split `reporting/tables.py`** (was 1004 lines) into a `reporting/tables/`
  package: one `T*_*.py` module per table (`T1_headline` .. `T8_scale`, mirroring the
  `G*` task naming), `_common.py` (shared helpers), `_markdown.py` (the two `.md`
  renderers), and `__init__.py` as the entry point that re-exports everything and
  owns the all-tables aggregation (`build_all_tables` / `write_all_tables`). The
  `"table1".."table8"` dict keys are unchanged, so `reporting.build`'s routing and the
  CSV filenames are untouched.

Frozen interfaces were not touched. `GenerationTask.run_side` (refactored in Phase A)
is not a frozen interface; the checkpoint is noted in `AGENT_GUIDE.md`.

## Verification

- `envs/mpvrdu/bin/python -m pytest` -> 99 passed, 1 skipped.
- Entry points import/run: `python -m gates --help`, `python -m cli.{generate,judge,build} --help`,
  `python -m scripts.inspect_results --help`, and `config.ROOT` still resolves.
- Phase-A behavior was verified via the test suite only. The **byte-identical
  side-artifact check needs a GPU** (retrievers/classifier), so it could not run on
  the local sm_120 box; worth running the smoke spec on Kaya when the queue frees up.

## Open items / next steps

- Nothing is committed. We are on `main`; the plan is to commit this session on a
  branch (the plan file is `~/.claude/plans/cheerful-toasting-lerdorf.md`).
- All planned refactor phases (A-E) are done; none remain outstanding.
- After job `1012121` finishes: `kaya.kaya pull`, then judge/build as above, then run
  the F1 frontier gate (`python -m gates frontier --full --run-tag yaml-g1-g2-g5-rerun`).
