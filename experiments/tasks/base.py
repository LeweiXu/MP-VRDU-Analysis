"""GenerationTask ABC and shared cell factories."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config import DEFAULT_PROMPT_MODE, ExperimentConfig
from experiments.corpus.resolve import filter_by_pool, pool_for_task
from pipeline.conditioner import InputConditioner, OracleConditioner, RetrievedTopK
from retrievers import Retriever
from schema import Modality, Question


@dataclass(frozen=True)
class Cell:
    """One unit of reasoner work: a question under a condition and a rung.

    `prompt_mode` picks the reasoner's instruction preamble (the hallucination
    task sweeps it); answerable tasks leave it at the default.
    """

    question: Question
    conditioner: InputConditioner
    representation: Modality
    prompt_mode: str = DEFAULT_PROMPT_MODE


@dataclass(frozen=True)
class Retrievers:
    """The two retrievers a generation pass may need (built lazily by the driver).

    The generate phase passes real retrievers; the judge phase passes guards that
    raise if called (every retrieved cell must be a prediction-cache hit).
    """

    text: Retriever
    vision: Retriever


class GenerationTask(ABC):
    """One coarse generation pass: specs + cells + optional side work."""

    #: Stable short name; also the cache subdirectory (e.g. "G1_oracle_ladder").
    name: str = "generation"
    #: Side-artifact filename this task writes into its cache dir, if any.
    side_artifact: str | None = None

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        """Reasoner specs this task generates. Empty for side-only tasks."""

        return ()

    def resolve_questions(self, config: ExperimentConfig, questions: Sequence[Question]) -> Sequence[Question]:
        """The corpus this task runs on, bound to its answerable/unanswerable pool."""

        return filter_by_pool(questions, pool_for_task(self.name))

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence[Question], *, retrievers: Retrievers
    ) -> list[Cell]:
        """Reasoner cells to run per spec. Empty for side-only tasks."""

        return []

    def run_side(self, config: ExperimentConfig, questions: Sequence[Question], side_dir: Path) -> None:
        """Optional extra GPU work (retrieval diagnostics, classifier)."""

        return None


def oracle_ladder_cells(config: ExperimentConfig, questions: Sequence[Question]) -> list[Cell]:
    """Oracle pages x the full representation ladder (the sufficiency cells)."""

    oracle = OracleConditioner()
    return [Cell(question, oracle, rung) for question in questions for rung in config.representations]


def matched_cross_sweep_cells(
    questions: Sequence[Question],
    *,
    retrievers: Retrievers,
    ks: Sequence[int],
    representation: Modality = "TLV",
) -> list[Cell]:
    """Matched (vision-retrieval) and cross (text-retrieval) cells swept over k.

    Question-major, k-minor. The `retrieved_{modality}_k{k}` conditioner name
    carries k, so each k lands in its own prediction-cache row.
    """

    conditioners = [
        (
            RetrievedTopK(retrievers.vision, k, name=f"retrieved_vision_k{k}"),
            RetrievedTopK(retrievers.text, k, name=f"retrieved_text_k{k}"),
        )
        for k in ks
    ]
    cells: list[Cell] = []
    for question in questions:
        for vision, text in conditioners:
            cells.append(Cell(question, vision, representation))
            cells.append(Cell(question, text, representation))
    return cells
