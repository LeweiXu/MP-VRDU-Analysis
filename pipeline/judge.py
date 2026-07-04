"""Uniform judge interface for converting predictions into comparable scores.

Stage D of the pipeline. A `Judge` scores one `Prediction` against a `Question`'s
gold answer, applied identically across every condition so columns are
commensurable. In Stage 7 the real judge is just another `Reasoner` spec (from a
different family than the evaluated model) wrapped by this protocol; the
generate -> extract -> score logic and judge-human agreement gate live there.

Stage 3 ships only `StubJudge`, a cheap heuristic (substring / abstention match)
so the orchestrator emits well-typed `Score`s end to end.
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
