"""Score model predictions against gold answers through a uniform interface.

Purpose:
    Defines Stage D of the pipeline. A `Judge` converts a `Prediction` and
    `Question` into a comparable `Score`, keeping answer evaluation independent
    of representation and reasoner backend.

Pipeline role:
    The orchestrator applies one judge implementation across all cells so table
    columns are commensurable. The current stub judge supports local tests; the
    GPT-4o-mini judge will land behind the same interface.

Arguments:
    None. This module is import-only; callers instantiate a `Judge` subclass and
    call `score(question, prediction)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from metrics.abstention import is_abstention
from schema import Prediction, Question, Score


class Judge(ABC):
    """Score a prediction against a question's gold answer."""

    spec: str = "judge"

    @abstractmethod
    def score(self, question: Question, prediction: Prediction) -> Score:
        """Return a `Score` for the prediction on this question."""


class StubJudge(Judge):
    """Heuristic placeholder judge used until the real judge arrives in Stage 7."""

    def __init__(self, spec: str = "stub") -> None:
        self.spec = spec

    def score(self, question: Question, prediction: Prediction) -> Score:
        abstained = is_abstention(prediction.text)
        gold = question.gold_answer.strip().casefold()
        pred = prediction.text.strip().casefold()
        if question.is_unanswerable:
            correct = abstained
        else:
            correct = bool(gold) and gold in pred and not abstained
        return Score(
            value=1.0 if correct else 0.0,
            correct=correct,
            abstained=abstained,
            judge_spec=self.spec,
        )
