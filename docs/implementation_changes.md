# Implementation Changes

Append to this file during every implementation run with changes that future
stages should know about: deviations from the written plan, migrations,
operational assumptions, and follow-up hazards.

## 2026-07-03 Stage 0

- Created the Stage 0 skeleton modules as docstring-only placeholders. No
  runtime interfaces or logic are frozen yet; those are Stage 2 and Stage 3
  work.
- Moved downloaded/local dataset and render artifacts from `data/` to `.data/`.
  The `data/` directory is now reserved for the importable Python package
  (`data/__init__.py`, `data/loader.py`, `data/render.py`).
- Updated `.gitignore`, Kaya environment variables, sync exclusions, README,
  and runbook docs to treat `.data/` as the root-relative dataset/render
  artifact directory.
- Added the rule to `docs/implementation_plan.md` that this file must be
  updated on every implementation run.
- Added pipeline-specific Kaya scripts under `scripts/kaya/`. The existing
  top-level `kaya/` directory remains a standalone reference kit and was not
  modified.

## 2026-07-03 Environment Install

- Created the local conda environment at `envs/mpvrdu` with Python 3.11,
  matching the Kaya setup script.
- Changed `requirements.txt` from `torch==2.7.1` to `torch==2.7.0` because
  `vllm==0.9.2` pins `torch==2.7.0`; the previous pair was not installable by
  pip's resolver.
