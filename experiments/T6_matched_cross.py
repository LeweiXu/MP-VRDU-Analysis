"""Experiment T6 — RQ2 mechanism: matched vs cross retrieval.

Purpose:
    Tests whether retrieval must use the same modality as reasoning (Table 6):
    matched = vision-retrieval + vision-reasoning; cross = text-retrieval +
    vision-reasoning, both under real retrieval. This is the one experiment whose
    generation needs the retrievers.

Pipeline role:
    A concrete `Experiment`. Its generation cells are `RetrievedTopK` conditions
    at a vision-bearing rung; the driver passes real retrievers in the generate
    phase and guard retrievers in the judge phase (every cell must be a
    prediction-cache hit). It also logs retrieval R/P/F1 as a side artifact.

Arguments:
    None. Import-only; the driver instantiates `MatchedCross()` via the registry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from experiments.base import Cell, Experiment, Retrievers, bootstrap_resamples, matched_cross_cells
from experiments.tables import build_table6_matched_vs_cross
from metrics.retrieval import score_retrieval
from pipeline.orchestrator import ResultRow


def _top_k(config: ExperimentConfig) -> int:
    return int(config.k_values[0] if config.k_values else 1)


class MatchedCross(Experiment):
    name = "T6_matched_cross"
    tables = ("table6",)
    depends_on = ("T1_headline",)

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence, *, retrievers: Retrievers
    ) -> list[Cell]:
        return matched_cross_cells(questions, retrievers=retrievers, k=_top_k(config))

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def run_side(self, config: ExperimentConfig, questions: Sequence, side_dir: Path) -> None:
        """Log page R/P/F1 for both retrievers (evidence-modality diagnostic)."""

        from covariates.retriever import BM25BGERetriever, ColQwenRetriever, MemoizedRetriever

        top_k = _top_k(config)
        text = MemoizedRetriever(
            BM25BGERetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        )
        vision = MemoizedRetriever(
            ColQwenRetriever(data_dir=config.paths.data_dir, cache_dir=config.paths.cache_dir, dpi=config.dpi)
        )
        from data.render import pdf_page_count
        from data.loader import resolve_pdf

        side_dir.mkdir(parents=True, exist_ok=True)
        with (side_dir / "retrieval.jsonl").open("w") as handle:
            for question in questions:
                page_count = pdf_page_count(resolve_pdf(question.doc_id, config.paths.data_dir))
                for modality, retriever in (("vision", vision), ("text", text)):
                    ranked = retriever.retrieve(question, page_count, top_k)
                    record = asdict(
                        score_retrieval(question, ranked, retriever=retriever.name, modality=modality, k=top_k)
                    )
                    for key, value in list(record.items()):
                        if isinstance(value, tuple):
                            record[key] = list(value)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        return {
            "table6": build_table6_matched_vs_cross(
                rows,
                bins=config.bins,
                margin_points=config.sufficiency_margin,
                retrieval_records=_load_retrieval_records(side_dir / "retrieval.jsonl"),
                n_bootstrap=bootstrap_resamples(config),
            )
        }


def _load_retrieval_records(path: Path) -> list[dict]:
    """Load T6 retrieval side records, returning empty list if absent."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
