"""G5 retrieval: matched vs cross retrieval cells + retrieval R/P/F1.

Purpose:
    Tests whether retrieval must use the same modality as reasoning (Table 6):
    matched = vision-retrieval + vision-reasoning; cross = text-retrieval +
    vision-reasoning, both under real retrieval at a vision-bearing rung. It also
    logs page retrieval R/P/F1 for both retrievers as a side artifact.

Pipeline role:
    One `GenerationTask` with reasoner cells (the driver passes real retrievers in
    generate, guards in judge) plus `run_side` retrieval diagnostics. Builds
    Table 6.

Arguments:
    None. Import-only; the registry instantiates `G5Retrieval()`.
"""

from __future__ import annotations

from pathlib import Path

from experiments.base import Cell, GenerationTask, Retrievers, matched_cross_sweep_cells
from experiments.side_artifacts import write_retrieval_eval


class G5Retrieval(GenerationTask):
    name = "G5_retrieval"
    side_artifact = "retrieval.jsonl"

    def _k_values(self, config) -> tuple[int, ...]:
        return tuple(config.k_values) if config.k_values else (1,)

    def model_specs(self, config) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        return matched_cross_sweep_cells(questions, retrievers=retrievers, ks=self._k_values(config))

    def run_side(self, config, questions, side_dir: Path) -> None:
        """Log page R/P/F1 for both retrievers (evidence-modality diagnostic).

        Vision-then-text, k-ascending: the same emission order as before the shared
        writer, so existing retrieval.jsonl caches stay byte-identical.
        """

        k_values = self._k_values(config)
        pairs = [("vision", k) for k in k_values] + [("text", k) for k in k_values]
        write_retrieval_eval(config, questions, pairs, side_dir, filename=self.side_artifact)
