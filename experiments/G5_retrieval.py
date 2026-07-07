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

import json
from pathlib import Path

from experiments.base import Cell, GenerationTask, Retrievers, matched_cross_sweep_cells


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
        """Log page R/P/F1 for both retrievers (evidence-modality diagnostic)."""

        from dataclasses import asdict

        from covariates.retriever import BM25BGERetriever, ColQwenRetriever, MemoizedRetriever
        from data.loader import resolve_pdf
        from data.render import pdf_page_count
        from metrics.retrieval import score_retrieval

        k_values = self._k_values(config)
        text = MemoizedRetriever(
            BM25BGERetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        )
        vision = MemoizedRetriever(
            ColQwenRetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        )
        side_dir.mkdir(parents=True, exist_ok=True)
        with (side_dir / self.side_artifact).open("w") as handle:
            for question in questions:
                page_count = pdf_page_count(resolve_pdf(question.doc_id, config.paths.data_dir))
                for modality, retriever in (("vision", vision), ("text", text)):
                    for k in k_values:
                        ranked = retriever.retrieve(question, page_count, k)
                        record = asdict(
                            score_retrieval(question, ranked, retriever=retriever.name, modality=modality, k=k)
                        )
                        for key, value in list(record.items()):
                            if isinstance(value, tuple):
                                record[key] = list(value)
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
