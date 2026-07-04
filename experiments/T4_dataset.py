"""Experiment T4 — RQ1 dataset replication: does the recipe hold on LongDocURL?

    Purpose:
    Re-runs the RQ1 headline on LongDocURL and reports whether frontiers match
    MMLongBench (Table 4). Smoke reuses T1's MMLongBench rows; full mode loads
    LongDocURL annotations through the frozen `Question` schema and generates an
    oracle ladder over that replication corpus.

Pipeline role:
    A concrete `Experiment` with `depends_on = ("T1_headline",)`. When a second
    dataset loader exists it becomes a generating experiment; today it is an
    aggregation-only relabel of the headline rows.

Arguments:
    None. Import-only; the driver instantiates `Dataset()` via the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from data.loader import load_longdocurl
from experiments.base import Cell, Experiment, Retrievers, bootstrap_resamples, oracle_ladder_cells
from experiments.tables import build_table4_dataset_replication
from pipeline.orchestrator import ResultRow


class Dataset(Experiment):
    name = "T4_dataset"
    tables = ("table4",)
    depends_on = ("T1_headline",)

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return () if config.smoke else (config.reasoner_spec,)

    def resolve_questions(self, config: ExperimentConfig, questions: Sequence) -> Sequence:
        return questions if config.smoke else load_longdocurl(config.paths.data_dir)

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence, *, retrievers: Retrievers
    ) -> list[Cell]:
        return oracle_ladder_cells(config, questions)

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table4": build_table4_dataset_replication(
                rows,
                bins=config.bins,
                margin_points=config.sufficiency_margin,
                n_bootstrap=bootstrap_resamples(config),
            )
        }
