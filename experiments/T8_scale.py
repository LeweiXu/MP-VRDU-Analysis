"""Experiment T8 — appendix: scale sanity across model sizes.

Purpose:
    Re-runs the RQ1 headline at other sizes (2B/32B) to check the recipe is
    qualitatively stable across scale (Table 8). In smoke there is one size, so
    this reuses T1's rows and shows a single size; the full run adds the extra
    sizes via `model_specs`.

Pipeline role:
    A concrete `Experiment`. `model_specs` drives the driver to run the oracle
    ladder under each extra size; `depends_on = ("T1_headline",)` brings the
    primary-size rows in for the comparison.

Arguments:
    None. Import-only; the driver instantiates `Scale()` via the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from experiments.base import Cell, Experiment, Retrievers, bootstrap_resamples, oracle_ladder_cells
from experiments.tables import build_table8_scale_sanity
from pipeline.orchestrator import ResultRow


class Scale(Experiment):
    name = "T8_scale"
    tables = ("table8",)
    depends_on = ("T1_headline",)

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        # Smoke has one size (reuse T1). Full adds the appendix sizes.
        return () if config.smoke else ("qwen3vl-2b-local", "qwen3vl-32b-local")

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence, *, retrievers: Retrievers
    ) -> list[Cell]:
        return oracle_ladder_cells(config, questions)

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table8": build_table8_scale_sanity(
                rows,
                bins=config.bins,
                margin_points=config.sufficiency_margin,
                n_bootstrap=bootstrap_resamples(config),
            )
        }
