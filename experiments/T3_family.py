"""Experiment T3 — RQ1 family replication: does the recipe hold on another 8B?

Purpose:
    Re-runs the RQ1 headline on a second model family (InternVL3-8B) and reports
    whether each bin's frontier matches Qwen3-VL qualitatively (Table 3). In smoke
    there is only one model, so this reuses T1's rows and the table shows a single
    family; the full run adds the InternVL spec via `model_specs`.

Pipeline role:
    A concrete `Experiment`. `model_specs` drives the driver to run the oracle
    ladder under each extra family; `depends_on = ("T1_headline",)` brings the
    primary-family rows in so the table can compare.

Arguments:
    None. Import-only; the driver instantiates `Family()` via the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from experiments.base import Cell, Experiment, Retrievers, bootstrap_resamples, oracle_ladder_cells
from experiments.tables import build_table3_family_replication
from pipeline.orchestrator import ResultRow


class Family(Experiment):
    name = "T3_family"
    tables = ("table3",)
    depends_on = ("T1_headline",)

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        # Smoke has one model (reuse T1). Full adds the second family.
        return () if config.smoke else ("internvl3-8b-local",)

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence, *, retrievers: Retrievers
    ) -> list[Cell]:
        return oracle_ladder_cells(config, questions)

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table3": build_table3_family_replication(
                rows,
                bins=config.bins,
                margin_points=config.sufficiency_margin,
                n_bootstrap=bootstrap_resamples(config),
            )
        }
