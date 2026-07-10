"""Unanswerable questions over similarity-retrieved pages under varied prompts,
plus the one-shot document classifier priced as a side-artifact."""

from __future__ import annotations

from pathlib import Path

from config import G3_PROMPT_MODES
from experiments.corpus.resolve import filter_by_pool, sample_per_doc_type
from experiments.engine.side_artifacts import write_classifier_eval
from experiments.tasks.base import Cell, GenerationTask, Retrievers
from pipeline.conditioner import SimilarityTopK

# The abstention study has no oracle arm (zero gold pages), so it feeds a small
# similarity-retrieved page set. Correct behaviour is abstention.
SIMILARITY_K = 3
REPRESENTATION = "TLV"


class G3Hallucination(GenerationTask):
    name = "G3_hallucination"
    side_artifact = "classifier.jsonl"

    def model_specs(self, config) -> tuple[str, ...]:
        return self._reasoner_specs(config)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        # Same similarity pages under each prompt condition (no guidance / generic /
        # abstention-targeted). The prompt mode is baked into the conditioner name so
        # each mode is its own cached cell, and rides on Cell.prompt_mode so the
        # reasoner applies the matching instruction.
        retriever_name = getattr(retrievers.text, "name", "text")
        base = f"similarity_{retriever_name}_k{SIMILARITY_K}"
        cells: list[Cell] = []
        for mode in (tuple(config.prompt_modes) or G3_PROMPT_MODES):
            conditioner = SimilarityTopK(retrievers.text, k=SIMILARITY_K, name=f"{base}_prompt-{mode}")
            cells += [Cell(question, conditioner, REPRESENTATION, prompt_mode=mode) for question in questions]
        return cells

    def run_side(self, config, questions, side_dir: Path, *, limit: int | None = None) -> None:
        """Price the document classifier over G1's answerable doc set (once).

        Routing reads the classifier's predicted bin per document, and it only ever
        routes G1's documents, so the classifier prices the same answerable pool +
        per_doc_type sample G1 runs on, not G3's unanswerable cells. It runs only
        when a classifier model is configured; otherwise routing reports the
        gold-bin ceiling with no classifier price.
        """

        if not config.classifier_spec:
            return
        docs = filter_by_pool(questions, "answerable")
        docs = sample_per_doc_type(docs, config.per_doc_type_sample, config.sample_seed)
        if limit is not None:
            docs = docs[:limit]
        write_classifier_eval(config, docs, side_dir, filename=self.side_artifact)
