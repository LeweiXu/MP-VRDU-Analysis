"""Retrieved pages across TLV/V by retrieval method and k."""

from __future__ import annotations

from pathlib import Path

from experiments.engine.side_artifacts import write_retrieval_eval
from experiments.tasks.base import Cell, GenerationTask, Retrievers, matched_cross_sweep_cells


class G2Retrieval(GenerationTask):
    name = "G2_retrieval"
    side_artifact = "retrieval.jsonl"

    def _k_values(self, config) -> tuple[int, ...]:
        return tuple(config.k_values) if config.k_values else (1,)

    def model_specs(self, config) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return matched_cross_sweep_cells(questions, retrievers=retrievers, ks=self._k_values(config))

    def run_side(self, config, questions, side_dir: Path) -> None:
        """Log page R/P/F1 for both retrievers across the k-sweep."""

        k_values = self._k_values(config)
        pairs = [("vision", k) for k in k_values] + [("text", k) for k in k_values]
        write_retrieval_eval(config, questions, pairs, side_dir, filename=self.side_artifact)
