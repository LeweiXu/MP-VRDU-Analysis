"""Experiment T1 — RQ1 headline: the sufficiency frontier by document type.

Purpose:
    The pivotal result (Table 1) and the source rows for T2/T3/T4/T5/T7/T8. Runs
    the reasoner over oracle (gold) pages across the full `T`/`TL`/`TLV`/`V`
    ladder, per Option-A bin, and marks the cheapest sufficient rung with its
    latency.

Pipeline role:
    A concrete `Experiment`. Its oracle-ladder cells are the only reasoner
    generation most tables need; the aggregation-only tables reuse these rows via
    `depends_on`. Reusable for smoke (2B, frozen corpus) and full (8B, all Qs).

Arguments:
    None. Import-only; the driver instantiates `Headline()` via the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from experiments.base import Cell, Experiment, Retrievers, bootstrap_resamples, oracle_ladder_cells
from experiments.tables import build_table1_headline
from pipeline.orchestrator import ResultRow


class Headline(Experiment):
    name = "T1_headline"
    tables = ("table1",)

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence, *, retrievers: Retrievers
    ) -> list[Cell]:
        return oracle_ladder_cells(config, questions)

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table1": build_table1_headline(
                rows,
                bins=config.bins,
                margin_points=config.sufficiency_margin,
                n_bootstrap=bootstrap_resamples(config),
            )
        }
