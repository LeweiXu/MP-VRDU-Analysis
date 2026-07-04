"""Experiment T4 — RQ1 dataset replication: does the recipe hold on LongDocURL?

Purpose:
    Re-runs the RQ1 headline on a second dataset and reports whether the
    frontiers match MMLongBench (Table 4, doc-type layer only). In smoke (and
    until the LongDocURL loader lands) this reuses T1's MMLongBench rows and
    labels the dataset column; the full run adds a LongDocURL corpus.

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
from experiments.base import Experiment, bootstrap_resamples
from experiments.tables import build_table4_dataset_replication
from pipeline.orchestrator import ResultRow


class Dataset(Experiment):
    name = "T4_dataset"
    tables = ("table4",)
    depends_on = ("T1_headline",)

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table4": build_table4_dataset_replication(
                rows,
                dataset=config.dataset,
                bins=config.bins,
                margin_points=config.sufficiency_margin,
                n_bootstrap=bootstrap_resamples(config),
            )
        }
