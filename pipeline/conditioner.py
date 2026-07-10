"""Selects the pages fed to a cell: oracle, retrieved, or similarity."""

from __future__ import annotations

from abc import ABC, abstractmethod

from retrievers import Retriever
from schema import PageSet, Question


class InputConditioner(ABC):
    """Select the pages that reach the model for one question."""

    #: Stable short name used in cell keys and result rows.
    name: str = "conditioner"

    @abstractmethod
    def condition(self, question: Question, page_count: int) -> PageSet:
        """Return the `PageSet` fed to the representation for this question."""


class OracleConditioner(InputConditioner):
    """Feed exactly the gold evidence pages (the reasoning ceiling)."""

    name = "oracle"

    def condition(self, question: Question, page_count: int) -> PageSet:
        pages = tuple(p for p in question.evidence_pages if 0 <= p < page_count)
        if not pages:
            # Questions with no gold pages (native unanswerable) fall back to the
            # first page so the pipeline still has something to render.
            pages = (0,) if page_count else ()
        return PageSet(pages, "oracle")


class RetrievedTopK(InputConditioner):
    """Feed the top-k pages returned by a retriever."""

    def __init__(self, retriever: Retriever, k: int, name: str | None = None) -> None:
        self.retriever = retriever
        self.k = int(k)
        self.name = name or f"retrieved_{getattr(retriever, 'name', 'r')}_k{self.k}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        ranked = self.retriever.retrieve(question, page_count, self.k)
        return PageSet(tuple(ranked)[: self.k], "retrieved", note=f"k={self.k}")


class JointTopK(InputConditioner):
    """Feed the free deduplicated union of two retrievers' top-k page sets.

    Joint retrieval is post-hoc and free (pivot 4.1): it unions two already-ranked
    page sets, no new retrieval and no score fusion (union, not RRF). Each retriever
    is asked for its own top-k, then `union` dedups them keeping first-seen order,
    so the result is at most 2k pages.
    """

    def __init__(self, text: Retriever, vision: Retriever, k: int, name: str | None = None) -> None:
        self.text = text
        self.vision = vision
        self.k = int(k)
        self.name = name or f"retrieved_joint_k{self.k}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        from retrievers.joint import union

        text_pages = self.text.retrieve(question, page_count, self.k)
        vision_pages = self.vision.retrieve(question, page_count, self.k)
        merged = union(text_pages, vision_pages)
        return PageSet(tuple(merged), "retrieved", note=f"joint k={self.k}")


class SimilarityTopK(InputConditioner):
    """Feed a few similarity-retrieved pages for zero-gold-page questions.

    The hallucination study has no oracle arm (unanswerable questions have no
    gold pages), so the only coherent page selection is a small similarity-ranked
    set. Same mechanism as `RetrievedTopK` but tagged `similarity` provenance and
    a fixed small k.
    """

    def __init__(self, retriever: Retriever, k: int = 3, name: str | None = None) -> None:
        self.retriever = retriever
        self.k = int(k)
        self.name = name or f"similarity_{getattr(retriever, 'name', 'r')}_k{self.k}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        ranked = self.retriever.retrieve(question, page_count, self.k)
        return PageSet(tuple(ranked)[: self.k], "similarity", note=f"k={self.k}")


class FullDoc(InputConditioner):
    """Feed every page (the feed-everything long-context baseline)."""

    name = "full"

    def condition(self, question: Question, page_count: int) -> PageSet:
        return PageSet.full(page_count)
