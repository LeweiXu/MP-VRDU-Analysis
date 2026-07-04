"""Experiment T5 — RQ2 mechanism: evidence-composition mediation.

Purpose:
    Decomposes accuracy by evidence-source label (text/table/chart/figure/layout)
    to show that a bin's composition, not its name, drives its frontier (Table 5,
    the causal core of the paper). Aggregation-only: built from T1's judged rows.

Pipeline role:
    A concrete `Experiment` with `depends_on = ("T1_headline",)` and no
    generation cells.

Arguments:
    None. Import-only; the driver instantiates `Composition()` via the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from experiments.base import Experiment, bootstrap_resamples
from experiments.tables import build_table5_composition_mediation
from pipeline.orchestrator import ResultRow


class Composition(Experiment):
    name = "T5_composition"
    tables = ("table5",)
    depends_on = ("T1_headline",)

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table5": build_table5_composition_mediation(
                rows,
                bins=config.bins,
                margin_points=config.sufficiency_margin,
                n_bootstrap=bootstrap_resamples(config),
            )
        }
