"""Unanswerable questions over similarity-retrieved pages under varied prompts."""

from __future__ import annotations

from experiments.tasks.base import Cell, GenerationTask, Retrievers
from pipeline.conditioner import SimilarityTopK

# The abstention study has no oracle arm (zero gold pages), so it feeds a small
# similarity-retrieved page set. Correct behaviour is abstention.
SIMILARITY_K = 3
REPRESENTATION = "TLV"


class G3Hallucination(GenerationTask):
    name = "G3_hallucination"

    def model_specs(self, config) -> tuple[str, ...]:
        return (config.reasoner_spec,)

    def generation_cells(self, config, questions, *, retrievers: Retrievers) -> list[Cell]:
        # Similarity pages from the text retriever at a fixed small k. The prompt
        # sweep (no prompt / generic / hallucination-targeted) rides on a prompt
        # variant that is threaded through the reasoner; that interface is added
        # with the driver prompt-mode wiring.
        conditioner = SimilarityTopK(retrievers.text, k=SIMILARITY_K)
        return [Cell(question, conditioner, REPRESENTATION) for question in questions]
