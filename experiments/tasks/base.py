"""GenerationTask ABC and shared cell factories."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from config import DEFAULT_PROMPT_MODE, ExperimentConfig
from experiments.corpus.resolve import filter_by_pool, pool_for_task
from pipeline.conditioner import InputConditioner, JointTopK, OracleConditioner, RetrievedTopK
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
    """The retrievers a generation pass may need (built lazily by the driver).

    The generate phase passes real retrievers; the judge phase passes guards that
    raise if called (every retrieved cell must be a prediction-cache hit).
    `rankers` holds the page_set ranking sources by name (empty when the run
    declares no page_set).
    """

    text: Retriever
    vision: Retriever
    rankers: Mapping[str, Retriever] = field(default_factory=dict)


class GenerationTask(ABC):
    """One coarse generation pass: specs + cells + optional side work."""

    #: Stable short name; also the cache subdirectory (e.g. "G1_oracle_ladder").
    name: str = "generation"
    #: Side-artifact filename this task writes into its cache dir, if any.
    side_artifact: str | None = None

    def model_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        """Reasoner specs this task generates. Empty for side-only tasks."""

        return ()

    def _reasoner_specs(self, config: ExperimentConfig) -> tuple[str, ...]:
        """The reasoner sweep list if set, else the single reasoner_spec.

        The driver runs one generation pass per spec, freeing the GPU between
        them, so returning several specs here (via `config.reasoner_specs`) is the
        model-size / family sweep. Each spec's cells key on its own model spec, so
        the passes never collide in cache.
        """

        return tuple(config.reasoner_specs) or (config.reasoner_spec,)

    def resolve_questions(self, config: ExperimentConfig, questions: Sequence[Question]) -> Sequence[Question]:
        """The corpus this task runs on, bound to its answerable/unanswerable pool.

        When `config.per_doc_type_sample` is set, the pool is subset to about that
        many questions per native doc_type, drawn as whole documents.
        """

        pool = filter_by_pool(questions, pool_for_task(self.name))
        if config.per_doc_type_sample:
            from experiments.corpus.resolve import sample_per_doc_type
            pool = sample_per_doc_type(pool, config.per_doc_type_sample, config.sample_seed)
        return pool

    def generation_cells(
        self, config: ExperimentConfig, questions: Sequence[Question], *, retrievers: Retrievers
    ) -> list[Cell]:
        """Reasoner cells to run per spec. Empty for side-only tasks."""

        return []

    def run_side(
        self, config: ExperimentConfig, questions: Sequence[Question], side_dir: Path, *, limit: int | None = None
    ) -> None:
        """Optional extra GPU work (retrieval diagnostics, classifier).

        `questions` is the full corpus (not the task's pool), so a side writer that
        needs a different scope than the task's cells can resolve it itself. `limit`
        is the smoke cap: a writer applies it after resolving its own scope.
        """

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
    joint_ks: Sequence[int] = (1, 3, 5),
    representations: Sequence[Modality] = ("TLV",),
    include_joint: bool = True,
) -> list[Cell]:
    """Matched (vision-retrieval), cross (text-retrieval), and joint (free union)
    cells swept over k, at each representation rung.

    This is the reasoner-inference sweep: the chosen text + vision arms plus their
    post-hoc union (when `include_joint`), fed to the reasoner at every rung in
    `representations` (default TLV and V). The full six-method accuracy ladder lives
    in the retrieval side-artifact, not here. Question-major; the
    `retrieved_{modality}_k{k}` conditioner name carries the modality and k, so each
    (modality, k, rung) lands in its own prediction-cache row.
    """

    single = [
        (
            RetrievedTopK(retrievers.vision, k, name=f"retrieved_vision_k{k}"),
            RetrievedTopK(retrievers.text, k, name=f"retrieved_text_k{k}"),
        )
        for k in ks
    ]
    joint = [
        JointTopK(retrievers.text, retrievers.vision, k, name=f"retrieved_joint_k{k}")
        for k in joint_ks
    ] if include_joint else []
    cells: list[Cell] = []
    for question in questions:
        for rep in representations:
            for vision, text in single:
                cells.append(Cell(question, vision, rep))
                cells.append(Cell(question, text, rep))
            for cond in joint:
                cells.append(Cell(question, cond, rep))
    return cells
