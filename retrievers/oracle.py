"""Oracle page selection: return the gold evidence pages (the perfect-retrieval
ceiling), exposed as a Retriever so it goes through the same path as bm25/colqwen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrievers import Retriever
from schema import Question


class OracleRetriever(Retriever):
    """Perfect retrieval: gold evidence pages first, then the rest in page order.

    It reads the labels rather than ranking content, so it is not a real retriever,
    but fitting the interface lets oracle page selection and a perfect-retrieval
    reference row reuse the same machinery. Gold pages come first (in evidence
    order); remaining pages follow in document order so a large k still returns a
    full ranking.
    """

    name = "oracle"
    modality = "oracle"

    def __init__(self, *, data_dir: Path | None = None, cache_dir: Path | None = None,
                 dpi: int = 200, **_ignored: Any) -> None:
        # Accepts the standard retriever kwargs so the uniform factory can build it.
        self.dpi = int(dpi)
        self.last_query_s = 0.0
        self.index_build_s = 0.0

    def rank(self, question: Question, page_count: int) -> tuple[int, ...]:
        gold = [p for p in question.evidence_pages if 0 <= p < int(page_count)]
        seen = set(gold)
        rest = [p for p in range(int(page_count)) if p not in seen]
        return tuple(gold + rest)

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return self.rank(question, int(page_count))[: int(k)]
