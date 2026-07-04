"""Retriever interfaces for text and vision page-ranking covariates.

A `Retriever` is a measured covariate, not an evaluated model: it ranks a
document's pages for a question so `RetrievedTopK` can feed a realistic page set,
and so `metrics/retrieval.py` can score page R/P/F1 vs gold (the RQ7
locate-vs-reason decomposition and the RQ2 retrieval-modality divergence).

Stage 3 ships only `StubRetriever` (returns pages in document order). The real
text (BM25 + BGE) and vision (ColPali/ColQwen) retrievers land in Stage 8 behind
this same `retrieve(question, page_count, k)` signature.
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
