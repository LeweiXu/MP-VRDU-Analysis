"""The `Experiment` contract and shared cell helpers.

Purpose:
    Defines the reusable unit the pipeline runs: one paper table (or table group)
    is one `Experiment`. Each experiment declares (a) which reasoner specs and
    conditioned cells it needs generated on a GPU, (b) any GPU side work (e.g. the
    doc-type classifier), and (c) how to build its CSV from cached, judged rows.
    The same experiment object serves the tiny smoke run and the full run — only
    the `ExperimentConfig` (model, corpus) differs — and each can run as its own
    Kaya job, so a change to one table is re-run in isolation.

Pipeline role:
    `experiments/T*_*.py` subclass `Experiment`; `experiments/registry.py` maps
    names to instances; `experiments/driver.py` runs them (generate on GPU, judge
    locally, build tables). This module only holds the contract and the cell
    factories shared across experiments (the oracle ladder and the matched/cross
    retrieval cells), so no table logic lives here.

Arguments:
    None. Import-only module. Subclasses override `model_specs`,
    `generation_cells`, `run_side`, and `build`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import ExperimentConfig
from covariates.retriever import Retriever
from pipeline.conditioner import InputConditioner, OracleConditioner, RetrievedTopK
from pipeline.orchestrator import ResultRow
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


class Experiment(ABC):
    """One paper table (or table group), runnable for smoke or full, per job."""

    #: Stable short name; also the cache subdirectory (e.g. "T1_headline").
    name: str = "experiment"
    #: The table keys this experiment emits (e.g. ("table1",)).
    tables: tuple[str, ...] = ()
    #: Other experiments whose judged rows this one also builds from.
    depends_on: tuple[str, ...] = ()

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        """Reasoner specs this experiment generates itself.

        Default: aggregation-only. Generating experiments override this with the
        config reasoner or a table-specific model list. Multi-model tables
        (family/scale) override this for the full run.
        """

        return ()

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence[Question], *, retrievers: Retrievers
    ) -> list[Cell]:
        """Cells to run per reasoner spec. Empty for aggregation-only tables."""

        return []

    def run_side(
        self, config: ExperimentConfig, questions: Sequence[Question], side_dir: Path
    ) -> None:
        """Optional extra GPU work (e.g. the classifier), writing side artifacts."""

        return None

    def resolve_questions(
        self, config: ExperimentConfig, questions: Sequence[Question]
    ) -> Sequence[Question]:
        """Return the corpus this experiment should run on."""

        return questions

    @abstractmethod
    def build(
        self, config: ExperimentConfig, rows: Sequence[ResultRow], side_dir: Path
    ) -> Mapping[str, pd.DataFrame]:
        """Build this experiment's table(s) from judged rows + side artifacts.

        `rows` is the union of this experiment's judged rows and every
        `depends_on` experiment's judged rows.
        """


def bootstrap_resamples(config: ExperimentConfig) -> int:
    """Document-level bootstrap resamples: fewer for smoke, 1000 for full."""

    return 200 if config.smoke else 1000


def oracle_ladder_cells(config: ExperimentConfig, questions: Sequence[Question]) -> list[Cell]:
    """Oracle pages × the full representation ladder (the T1 headline cells)."""

    oracle = OracleConditioner()
    return [
        Cell(question, oracle, rung)
        for question in questions
        for rung in config.representations
    ]


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
