"""Experiment T4 — RQ1 replication on a held-out MMLongBench document subset.

Purpose:
    Re-runs the RQ1 headline on a *different* set of MMLongBench documents and
    reports whether the per-domain frontiers replicate (Table 4). For text_heavy
    and in_between it draws ~100 questions from documents NOT used by T1 (a
    disjoint held-out subset); visual_heavy is too thin to hold out (101 Q / 15
    docs) so it reuses T1's questions. SlideVQA is the planned visual-heavy
    replication and is out of scope here. (This replaces the earlier LongDocURL
    plan, which had no way to map LongDocURL categories onto the three domains.)

Pipeline role:
    A concrete `Experiment` with `depends_on = ("T1_headline",)`. In full mode it
    generates an oracle ladder over the held-out subset; smoke reuses T1's rows.

Arguments:
    None. Import-only; the driver instantiates `Dataset()` via the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from data.loader import load_mmlongbench
from experiments.base import Cell, Experiment, Retrievers, bootstrap_resamples, oracle_ladder_cells
from experiments.corpus import sample_table4_replication
from experiments.tables import build_table4_dataset_replication
from pipeline.orchestrator import ResultRow


class Dataset(Experiment):
    name = "T4_dataset"
    tables = ("table4",)
    depends_on = ("T1_headline",)

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return () if config.smoke else (config.reasoner_spec,)

    def resolve_questions(self, config: ExperimentConfig, questions: Sequence) -> Sequence:
        if config.smoke:
            return questions
        # Held-out MMLongBench subset: disjoint docs for text/in-between, T1's
        # questions reused for the thin visual_heavy bin.
        all_questions = list(load_mmlongbench(data_dir=config.paths.data_dir))
        return sample_table4_replication(
            all_questions,
            config.per_bin_sample or 100,
            bins=config.bins,
            seed=config.sample_seed,
        )

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
