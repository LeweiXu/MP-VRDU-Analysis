"""Define page-ranking retriever interfaces for text and vision covariates.

Purpose:
    A `Retriever` ranks document pages for a question so `RetrievedTopK` can feed
    realistic page sets instead of oracle evidence. Retrieval is measured as a
    covariate, not treated as the reasoner itself.

Pipeline role:
    Input conditioning calls `retrieve(question, page_count, k)` and retrieval
    metrics compare the returned page indices with gold evidence pages. Real
    BM25+BGE and ColQwen implementations will replace `StubRetriever` behind the
    same signature.

Arguments:
    None. This module is import-only; callers instantiate retriever classes and
    call `retrieve()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schema import Question


class Retriever(ABC):
    """Rank a document's pages for a question."""

    name: str = "retriever"

    @abstractmethod
    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        """Return page indices ranked most- to least-relevant (at most `k`)."""


class StubRetriever(Retriever):
    """Deterministic placeholder: return the first `k` pages in document order."""

    name = "stub"

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return tuple(range(min(int(k), page_count)))
