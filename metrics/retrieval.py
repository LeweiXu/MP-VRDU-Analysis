"""Compute page-retrieval metrics against gold evidence pages.

Purpose:
    Scores retrieved page indices against MMLongBench's page-level evidence so
    the paper can separate "locating the right page" from "reasoning over the
    selected representation." The metrics are intentionally page-level because
    MMLongBench does not expose in-page evidence boxes.

Pipeline role:
    Stage M6 uses these helpers to evaluate text (`BM25+BGE`) and vision
    (`ColQwen`) retrievers before `RetrievedTopK` feeds the selected pages into
    the normal orchestrator path. Later full runs reuse the same rows for
    matched-vs-cross and evidence-modality slices.

Arguments:
    None. This module is import-only; callers pass `Question` objects, retrieved
    page tuples, and retriever names/modalities into `score_retrieval()`,
    `page_prf()`, or summary helpers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from schema import Question


@dataclass(frozen=True)
class PagePRF:
    """Precision/recall/F1 for one retrieved page set."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RetrievalEvalRow:
    """One retrieval metric row for a question/retriever/modality."""

    question_id: str
    doc_id: str
    doc_type: str
    evidence_sources: tuple[str, ...]
    retriever: str
    modality: str
    k: int
    retrieved_pages: tuple[int, ...]
    gold_pages: tuple[int, ...]
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RetrievalSummary:
    """Macro-averaged retrieval metrics for a row group."""

    n_rows: int
    precision: float
    recall: float
    f1: float


def page_prf(retrieved: Sequence[int], gold: Sequence[int]) -> PagePRF:
    """Return page precision/recall/F1 for retrieved vs gold page indices."""

    retrieved_set = set(int(page) for page in retrieved)
    gold_set = set(int(page) for page in gold)
    if not retrieved_set and not gold_set:
        return PagePRF(1.0, 1.0, 1.0)
    if not retrieved_set:
        return PagePRF(0.0, 0.0 if gold_set else 1.0, 0.0)
    true_positive = len(retrieved_set.intersection(gold_set))
    precision = true_positive / len(retrieved_set)
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return PagePRF(precision, recall, f1)


def score_retrieval(
    question: Question,
    retrieved_pages: Sequence[int],
    *,
    retriever: str,
    modality: str,
    k: int,
) -> RetrievalEvalRow:
    """Build one retrieval metric row for a question."""

    prf = page_prf(retrieved_pages, question.evidence_pages)
    return RetrievalEvalRow(
        question_id=question.id,
        doc_id=question.doc_id,
        doc_type=question.doc_type,
        evidence_sources=question.evidence_sources,
        retriever=retriever,
        modality=modality,
        k=int(k),
        retrieved_pages=tuple(int(page) for page in retrieved_pages),
        gold_pages=question.evidence_pages,
        precision=prf.precision,
        recall=prf.recall,
        f1=prf.f1,
    )


def retrieval_slice_keys(question: Question, retrieval_modality: str) -> tuple[str, ...]:
    """Return evidence-modality slice keys for one question/retrieval modality."""

    sources = question.evidence_sources or ("none",)
    return tuple(
        f"{retrieval_modality}:{str(source).strip().casefold().replace(' ', '_') or 'none'}"
        for source in sources
    )


def retrieval_summary(rows: Iterable[RetrievalEvalRow]) -> RetrievalSummary:
    """Return macro-averaged precision/recall/F1 over retrieval rows."""

    items = tuple(rows)
    if not items:
        return RetrievalSummary(0, 0.0, 0.0, 0.0)
    n = len(items)
    return RetrievalSummary(
        n_rows=n,
        precision=sum(row.precision for row in items) / n,
        recall=sum(row.recall for row in items) / n,
        f1=sum(row.f1 for row in items) / n,
    )


def retrieval_summary_by_modality(rows: Iterable[RetrievalEvalRow]) -> Mapping[str, RetrievalSummary]:
    """Group retrieval metrics by retriever modality (`text`, `vision`, ...)."""

    groups: dict[str, list[RetrievalEvalRow]] = defaultdict(list)
    for row in rows:
        groups[row.modality].append(row)
    return {key: retrieval_summary(value) for key, value in groups.items()}


def retrieval_summary_by_evidence_slice(
    rows: Iterable[RetrievalEvalRow],
) -> Mapping[str, RetrievalSummary]:
    """Group retrieval metrics by retrieval modality and evidence-source label."""

    groups: dict[str, list[RetrievalEvalRow]] = defaultdict(list)
    for row in rows:
        for source in row.evidence_sources or ("none",):
            key = f"{row.modality}:{str(source).strip().casefold().replace(' ', '_') or 'none'}"
            groups[key].append(row)
    return {key: retrieval_summary(value) for key, value in groups.items()}
