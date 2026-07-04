"""Resolve the question set an experiment runs on, for smoke or full.

Purpose:
    One place that turns an `ExperimentConfig` into the list of `Question`s to
    run, so every experiment is corpus-agnostic and identical between the smoke
    and full runs. Smoke → the frozen ~7-doc smoke corpus; full → the whole
    MMLongBench-Doc set (optionally capped for a gate pilot).

Pipeline role:
    `experiments/driver.py` calls `load_questions(config)` once and hands the
    same list to every experiment's `generation_cells` / `build`.

Arguments:
    None. Import-only module; callers use `load_questions(config, limit=...)`.
"""

from __future__ import annotations

from config import ExperimentConfig
from data.loader import load_mmlongbench
from experiments.smoke import load_smoke_questions
from schema import Question


def load_questions(config: ExperimentConfig, *, limit: int | None = None) -> list[Question]:
    """Return the questions for this run: frozen smoke corpus or the full set."""

    if config.smoke:
        questions = list(load_smoke_questions(config.paths.data_dir))
    else:
        questions = list(load_mmlongbench(data_dir=config.paths.data_dir))
    if limit is not None:
        questions = questions[: max(1, limit)]
    return questions
