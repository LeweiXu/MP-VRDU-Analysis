"""Metrics: per-question correctness, accuracy, F1, retrieval recall@k.

Faithful-ish to MMLongBench-Doc (context.md §11):
- Accuracy: mean per-question correctness, where correctness is the
  `answer_format`-aware rule comparison, with abstention handled explicitly
  (unanswerable questions are correct iff the model abstains).
- F1: binary F1 of the *answerability* decision (positive class = "answerable").
  This is how MMLongBench-Doc / VLMEvalKit summarise abstention quality:
  gold_answerable vs pred_answerable over all questions.

The rule comparison stands in for the official LLM-as-judge for local plumbing.
For real runs, swap in the LLM judge (eval/judge.py) and DECLARE it in the
methodology (context.md §6, §10: judge is a fixed, reproducible variable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .extract import compare, is_abstention


@dataclass
class AnswerScore:
    correct: bool
    pred_abstained: bool
    gold_answerable: bool


def score_answer(pred: str, gold: str, answer_format: str,
                 gold_is_unanswerable: Optional[bool] = None) -> AnswerScore:
    """Score one prediction. `gold_is_unanswerable` overrides format inference."""
    if gold_is_unanswerable is None:
        gold_is_unanswerable = (
            (answer_format or "").strip().lower() in {"none", "unanswerable"}
            or is_abstention(gold)
        )
    pred_abstained = is_abstention(pred)
    gold_answerable = not gold_is_unanswerable

    if gold_is_unanswerable:
        correct = pred_abstained
    elif pred_abstained:
        correct = False  # abstained on an answerable question
    else:
        correct = compare(pred, gold, answer_format)

    return AnswerScore(correct=correct, pred_abstained=pred_abstained,
                       gold_answerable=gold_answerable)


def _binary_f1(gold: list[bool], pred: list[bool]) -> float:
    """F1 with True as the positive class."""
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def aggregate(scores: Iterable[AnswerScore]) -> dict:
    """Aggregate per-question scores into accuracy + answerability F1."""
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "f1": 0.0}
    accuracy = sum(s.correct for s in scores) / n
    gold_ans = [s.gold_answerable for s in scores]
    pred_ans = [not s.pred_abstained for s in scores]
    return {
        "n": n,
        "accuracy": accuracy,
        "f1": _binary_f1(gold_ans, pred_ans),
        "n_answerable": sum(gold_ans),
        "n_unanswerable": n - sum(gold_ans),
    }


def recall_at_k(retrieved_pages: list[int], evidence_pages: list[int],
                k: Optional[int] = None) -> float:
    """Fraction of gold evidence pages present in the top-k retrieved pages.

    For unanswerable questions (no evidence) recall is defined as 1.0 (nothing
    to retrieve), so it doesn't drag down the retrieval metric.
    """
    if not evidence_pages:
        return 1.0
    top = retrieved_pages[:k] if k is not None else retrieved_pages
    hit = len(set(top) & set(evidence_pages))
    return hit / len(set(evidence_pages))
