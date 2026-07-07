"""The `GenerationTask` contract and the reusable cell shapes.

Purpose:
    Defines the unit the pipeline generates: one coarse GPU pass (a set of
    reasoner specs over a set of conditioned cells, plus optional side work).
    Each task lives in its own `experiments/G*_*.py` file and subclasses
    `GenerationTask`; adding a generation experiment is just adding one such file
    and registering it in `experiments/registry.py`.

Pipeline role:
    A leaf contract module. `experiments/driver.py` runs a task (generate on GPU,
    judge locally); `experiments/tables.py` aggregates the judged rows into the
    paper tables. This file holds only the ABC and the two cell factories shared
    across tasks (the oracle ladder and the matched/cross retrieval cells).

Arguments:
    None. Import-only. Subclasses override `model_specs`, `generation_cells`,
    `run_side`, and `resolve_questions`.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config import ExperimentConfig
from covariates.retriever import Retriever
from pipeline.conditioner import InputConditioner, OracleConditioner, RetrievedTopK
from schema import Modality, Question


@dataclass(frozen=True)
class Cell:
    """One unit of reasoner work: a question under a condition and a rung."""

    question: Question
    conditioner: InputConditioner
    representation: Modality


@dataclass(frozen=True)
class Retrievers:
    """The two retrievers a generation pass may need (built lazily by the driver).

    The generate phase passes real retrievers; the judge phase passes guards
    that raise if called (every retrieved cell must be a prediction-cache hit).
    """

    text: Retriever
    vision: Retriever


class GenerationTask(ABC):
    """One coarse generation pass: specs + cells + optional side work."""

    #: Stable short name; also the cache subdirectory (e.g. "G1_sufficiency").
    name: str = "generation"
    #: Side-artifact filename this task writes into its cache dir, if any.
    side_artifact: str | None = None

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        """Reasoner specs this task generates. Empty for side-only tasks."""

        return ()

    def resolve_questions(
        self, config: ExperimentConfig, questions: Sequence[Question]
    ) -> Sequence[Question]:
        """The corpus this task runs on (defaults to the shared corpus)."""

        return questions

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence[Question], *, retrievers: Retrievers
    ) -> list[Cell]:
        """Reasoner cells to run per spec. Empty for side-only tasks."""

        return []

    def run_side(
        self, config: ExperimentConfig, questions: Sequence[Question], side_dir: Path
    ) -> None:
        """Optional extra GPU work (retrieval diagnostics, classifier)."""

        return None


def oracle_ladder_cells(config: ExperimentConfig, questions: Sequence[Question]) -> list[Cell]:
    """Oracle pages x the full representation ladder (the sufficiency cells)."""

    oracle = OracleConditioner()
    return [Cell(question, oracle, rung) for question in questions for rung in config.representations]


def matched_cross_cells(
    questions: Sequence[Question],
    *,
    retrievers: Retrievers,
    k: int,
    representation: Modality = "TLV",
) -> list[Cell]:
    """Matched (vision-retrieval) and cross (text-retrieval) vision-reasoning cells.

    Conditioner names are fixed strings so the prediction key is identical across
    the generate and judge phases regardless of which retriever object is passed.
    """

    vision = RetrievedTopK(retrievers.vision, k, name=f"retrieved_vision_k{k}")
    text = RetrievedTopK(retrievers.text, k, name=f"retrieved_text_k{k}")
    cells: list[Cell] = []
    for question in questions:
        cells.append(Cell(question, vision, representation))
        cells.append(Cell(question, text, representation))
    return cells


def matched_cross_sweep_cells(
    questions: Sequence[Question],
    *,
    retrievers: Retrievers,
    ks: Sequence[int],
    representation: Modality = "TLV",
) -> list[Cell]:
    """Matched/cross cells swept over several top-k values.

    Question-major, k-minor: for each question, every k contributes a matched
    (vision-retrieval) then cross (text-retrieval) cell, so one question's k-1..k-N
    cells sit together. The `retrieved_{modality}_k{k}` conditioner name carries k,
    so each k lands in its own prediction-cache row and Table 6 can separate them.
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
