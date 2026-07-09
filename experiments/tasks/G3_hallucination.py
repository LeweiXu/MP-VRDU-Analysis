"""Unanswerable questions over similarity-retrieved pages under varied prompts."""

from __future__ import annotations

from config import G3_PROMPT_MODES
from experiments.tasks.base import Cell, GenerationTask, Retrievers
from pipeline.conditioner import SimilarityTopK

# The abstention study has no oracle arm (zero gold pages), so it feeds a small
# similarity-retrieved page set. Correct behaviour is abstention.
SIMILARITY_K = 3
REPRESENTATION = "TLV"


class G3Hallucination(GenerationTask):
    name = "G3_hallucination"

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
        for mode in G3_PROMPT_MODES:
            conditioner = SimilarityTopK(retrievers.text, k=SIMILARITY_K, name=f"{base}_prompt-{mode}")
            cells += [Cell(question, conditioner, REPRESENTATION, prompt_mode=mode) for question in questions]
        return cells
