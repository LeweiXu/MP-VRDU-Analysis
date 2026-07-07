"""G2 family: the oracle ladder on a second model family.

Purpose:
    Re-runs the sufficiency ladder on InternVL3-8B so Table 3 can report whether
    each bin's frontier matches the primary Qwen3-VL family qualitatively. In
    smoke there is one family, so this generates nothing and Table 3 reuses G1.

Pipeline role:
    One `GenerationTask`. Its InternVL rows plus G1's primary-family rows build
    Table 3 (see `reporting.tables` routing).

Arguments:
    None. Import-only; the registry instantiates `G2Family()`.
"""

from __future__ import annotations

from experiments.base import Cell, GenerationTask, Retrievers, oracle_ladder_cells


class G2Family(GenerationTask):
    name = "G2_family"

    def model_specs(self, config) -> tuple[str, ...]:
        # Smoke has one family (Table 3 reuses G1). Full adds the second family.
        return () if config.smoke else ("internvl3-8b-local",)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return oracle_ladder_cells(config, questions)
