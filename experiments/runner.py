"""Expand experiment configs into cached pipeline cells.

Purpose:
    Provides small, reusable cell-expansion helpers for MVP smoke stages before
    the full paper-table runner exists. Stage M4 uses the oracle ladder helper
    to run every `(question, representation)` cell through the same
    cache-resumable orchestrator path that later full sweeps will use.

Pipeline role:
    Keeps experiment expansion outside `pipeline.orchestrator`: this module
    chooses questions/conditions/rungs, while the orchestrator remains the
    single-cell A->B->C->D executor and cache owner. Section-2 stages will add
    broader condition/model/policy grids behind this same shape.

Arguments:
    None. This module is import-only; callers pass an `ExperimentConfig`, an
    iterable of `Question` objects, and optionally an injected `Orchestrator`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from config import ExperimentConfig
from pipeline.conditioner import OracleConditioner
from pipeline.orchestrator import Orchestrator, ResultRow, make_cache_key
from schema import Modality, Question


@dataclass(frozen=True)
class RunBatch:
    """Rows plus cache accounting from one expanded orchestrator pass."""

    rows: tuple[ResultRow, ...]
    cache_hits: int
    computed: int
    cache_path: Path
    cache_rows: int


def run_oracle_ladder(
    config: ExperimentConfig,
    questions: Iterable[Question],
    *,
    orchestrator: Orchestrator | None = None,
    representations: Sequence[Modality] | None = None,
) -> RunBatch:
    """Run oracle pages through every requested representation rung.

    Cache hits are counted before calling `run_cell()`. A second call with the
    same orchestrator/cache should therefore report `computed == 0`.
    """

    orchestrator = orchestrator or Orchestrator(config)
    conditioner = OracleConditioner()
    rungs = tuple(representations or config.representations)
    rows: list[ResultRow] = []
    cache_hits = 0

    for question in questions:
        for representation in rungs:
            key = make_cache_key(
                question,
                conditioner.name,
                representation,
                orchestrator.reasoner.spec,
                orchestrator.judge.spec,
                config.dpi,
            )
            if orchestrator.cache.get(key) is not None:
                cache_hits += 1
            rows.append(orchestrator.run_cell(question, conditioner, representation))

    return RunBatch(
        rows=tuple(rows),
        cache_hits=cache_hits,
        computed=len(rows) - cache_hits,
        cache_path=orchestrator.cache.path,
        cache_rows=len(orchestrator.cache),
    )
