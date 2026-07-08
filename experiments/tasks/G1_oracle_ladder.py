"""Oracle pages across the T/TL/TLV/V ladder; the base grid the reasoning sweeps reuse."""

from __future__ import annotations

from experiments.tasks.base import Cell, GenerationTask, Retrievers, oracle_ladder_cells


class G1OracleLadder(GenerationTask):
    name = "G1_oracle_ladder"

    def model_specs(self, config) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return oracle_ladder_cells(config, questions)
