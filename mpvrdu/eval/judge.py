"""Judge interface: rule-based (local, deterministic) + LLM (real runs).

The judge decides per-question correctness. Local plumbing uses the
deterministic RuleBasedJudge (no API, reproducible). Real runs may use an LLM
judge — which MUST be fixed and declared in the methodology (context.md §6,§10),
since it is a hidden experimental variable otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..config import JudgeConfig
from .metrics import AnswerScore, score_answer


class Judge(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, question: str, pred: str, gold: str, answer_format: str,
              gold_is_unanswerable: Optional[bool] = None) -> AnswerScore:
        ...


class RuleBasedJudge(Judge):
    """Deterministic, API-free judge via the rule comparison in eval/metrics."""

    name = "rule"

    def score(self, question: str, pred: str, gold: str, answer_format: str,
              gold_is_unanswerable: Optional[bool] = None) -> AnswerScore:
        return score_answer(pred, gold, answer_format,
                            gold_is_unanswerable=gold_is_unanswerable)


class LLMJudge(Judge):
    """LLM-as-judge (extraction + comparison). Lazy; for real runs only.

    Declared-and-fixed model. Falls back to the rule comparison for the final
    answerability bookkeeping. Implementation stub: wire the real client in the
    stage that first needs scored Kaya runs.
    """

    name = "llm"

    def __init__(self, model_id: str):
        self.model_id = model_id

    def score(self, question: str, pred: str, gold: str, answer_format: str,
              gold_is_unanswerable: Optional[bool] = None) -> AnswerScore:
        raise NotImplementedError(
            "LLMJudge is not wired yet — use type: rule for local plumbing. "
            "Implement the fixed-model extraction client before Kaya scored runs."
        )


def build_judge(cfg: JudgeConfig) -> Judge:
    if cfg.type == "rule":
        return RuleBasedJudge()
    if cfg.type == "llm":
        if not cfg.model_id:
            raise ValueError("judge.type=llm requires judge.model_id")
        return LLMJudge(cfg.model_id)
    raise ValueError(f"unknown judge type {cfg.type!r}")
