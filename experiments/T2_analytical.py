"""Experiment T2 — RQ1 analytical slice: bin × question-type.

Purpose:
    Re-slices the T1 headline rows by question type (single/multi/none hop) per
    bin (Table 2). Aggregation-only: no new reasoner work — it builds entirely
    from T1's judged oracle-ladder rows.

Pipeline role:
    A concrete `Experiment` with `depends_on = ("T1_headline",)` and no
    generation cells, so its Kaya "job" is just a table rebuild.

Arguments:
    None. Import-only; the driver instantiates `Analytical()` via the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from experiments.base import Experiment, bootstrap_resamples
from experiments.tables import build_table2_analytical
from pipeline.orchestrator import ResultRow


class Analytical(Experiment):
    name = "T2_analytical"
    tables = ("table2",)
    depends_on = ("T1_headline",)

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table2": build_table2_analytical(
                rows, bins=config.bins, n_bootstrap=bootstrap_resamples(config)
            )
        }
