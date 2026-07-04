"""Input-conditioner interfaces for oracle, retrieved, full-document, and buried-oracle page sets.

Stage A of the pipeline. An `InputConditioner` decides *which pages reach the
model* for one question, returning a `PageSet` with its provenance. The four
conditions mirror the spec's input-conditions table plus the RQ3 burying sweep:

- `OracleConditioner`  -> exactly the gold evidence pages (the reasoning ceiling).
- `RetrievedTopK`      -> top-k pages from a real `Retriever` (Stage 8; stub now).
- `FullDoc`            -> every page (the feed-everything baseline).
- `BuriedOracle`       -> gold pages held present, padded with same-corpus
                          distractor pages (how much irrelevant context the model
                          tolerates).

`condition(question, page_count)` is the frozen signature. `page_count` is the
document's total page count, resolved once per question by the orchestrator, so
the full-document and burying conditions know the page range without every
conditioner re-opening the PDF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from covariates.retriever import Retriever
from schema import PageSet, Question


class InputConditioner(ABC):
    """Select the pages that reach the model for one question."""

    #: Stable short name used in cache keys and result rows.
    name: str = "conditioner"

    @abstractmethod
    def condition(self, question: Question, page_count: int) -> PageSet:
        """Return the `PageSet` fed to the representation for this question."""


class OracleConditioner(InputConditioner):
    """Feed exactly the gold evidence pages (the reasoning measurement)."""

    name = "oracle"

    def condition(self, question: Question, page_count: int) -> PageSet:
        pages = tuple(p for p in question.evidence_pages if 0 <= p < page_count)
        if not pages:
            # Native-unanswerable questions have no gold pages; fall back to the
            # first page so the pipeline still has something to render.
            pages = (0,) if page_count else ()
        return PageSet(pages, "oracle")


class RetrievedTopK(InputConditioner):
    """Feed the top-k pages returned by a real retriever (RQ7 / abstention)."""

    def __init__(self, retriever: Retriever, k: int) -> None:
        self.retriever = retriever
        self.k = int(k)
        self.name = f"retrieved_k{self.k}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        ranked = self.retriever.retrieve(question, page_count, self.k)
        pages = tuple(ranked)[: self.k]
        return PageSet(pages, "retrieved", note=f"k={self.k}")


class FullDoc(InputConditioner):
    """Feed every page (the feed-everything long-context baseline)."""

    name = "full"

    def condition(self, question: Question, page_count: int) -> PageSet:
        return PageSet.full(page_count)


class BuriedOracle(InputConditioner):
    """Feed gold pages plus N same-document distractor pages (RQ3 burying).

    Gold pages always stay present; distractors are the first `n_distractors`
    non-gold pages in document order (deterministic, so the cache key is stable).
    The full-document endpoint of the burying curve is `FullDoc`.
    """

    def __init__(self, n_distractors: int) -> None:
        self.n_distractors = int(n_distractors)
        self.name = f"buried_n{self.n_distractors}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        gold = tuple(p for p in question.evidence_pages if 0 <= p < page_count)
        distractors = tuple(
            p for p in range(page_count) if p not in set(gold)
        )[: self.n_distractors]
        return PageSet(gold + distractors, "buried", note=f"n={self.n_distractors}")
