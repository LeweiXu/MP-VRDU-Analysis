"""G1 sufficiency: the oracle ladder for the primary reasoner.

Purpose:
    The core generation: oracle (gold) pages across the full T/TL/TLV/V
    representation ladder, run with the config's primary reasoner (8B). Its judged
    rows are the source for tables 1, 2, 5, and 7 (routing policy accuracy).

Pipeline role:
    One `GenerationTask` (`experiments/registry.py` registers it). Reusable for
    smoke (2B) and full (8B); the config's `reasoner_spec` selects the model.

Arguments:
    None. Import-only; the registry instantiates `G1Sufficiency()`.
"""

from __future__ import annotations

from experiments.base import Cell, GenerationTask, Retrievers, oracle_ladder_cells


class G1Sufficiency(GenerationTask):
    name = "G1_sufficiency"

    def model_specs(self, config) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return oracle_ladder_cells(config, questions)
