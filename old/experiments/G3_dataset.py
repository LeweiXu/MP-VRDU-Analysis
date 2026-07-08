"""G3 dataset: the oracle ladder on a held-out MMLongBench subset.

Purpose:
    Re-runs the sufficiency ladder on documents NOT used by G1, for the two big
    bins (text_heavy, in_between), so Table 4 can report whether the per-domain
    recipe replicates on a disjoint document set. visual_heavy is too thin to hold
    out and is out of scope for now (no reuse, no SlideVQA yet).

Pipeline role:
    One `GenerationTask`; its held-out rows build Table 4. Smoke reuses G1 (no
    generation).

Arguments:
    None. Import-only; the registry instantiates `G3Dataset()`.
"""

from __future__ import annotations

from collections.abc import Sequence

from experiments.base import Cell, GenerationTask, Retrievers, oracle_ladder_cells
from experiments.corpus import sample_table4_replication
from schema import Question


class G3Dataset(GenerationTask):
    name = "G3_dataset"

    def model_specs(self, config) -> tuple[str, ...]:
        return () if config.smoke else (config.reasoner_spec,)

    def resolve_questions(self, config, questions) -> Sequence[Question]:
        if config.smoke:
            return questions
        from data.loader import load_mmlongbench

        all_questions = list(load_mmlongbench(data_dir=config.paths.data_dir))
        held_out_bins = tuple(b for b in config.bins if b != "visual_heavy")
        return sample_table4_replication(
            all_questions,
            config.per_bin_sample or 100,
            bins=held_out_bins,
            seed=config.sample_seed,
            reuse_bins=(),
        )

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return oracle_ladder_cells(config, questions)
