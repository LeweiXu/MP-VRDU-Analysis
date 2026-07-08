"""Page retrievers arranged as cost rungs: the Retriever base, shared ranking
helpers, and memoization."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config import DEFAULT_PATHS
from data.loader import resolve_pdf
from data.render import render_pdf
from schema import Question


def tokenize(text: str) -> list[str]:
    """Return a compact lower-case token stream for lexical ranking."""

    return re.findall(r"[A-Za-z0-9]+", text.casefold())


def normalise_scores(scores: Sequence[float]) -> list[float]:
    """Scale finite scores to [0, 1], collapsing ties to zeros."""

    finite = [float(score) if math.isfinite(float(score)) else 0.0 for score in scores]
    if not finite:
        return []
    lo, hi = min(finite), max(finite)
    if hi <= lo:
        return [0.0 for _ in finite]
    return [(score - lo) / (hi - lo) for score in finite]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity for two numeric vectors."""

    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    norm_a = math.sqrt(sum(float(a[i]) ** 2 for i in range(n)))
    norm_b = math.sqrt(sum(float(b[i]) ** 2 for i in range(n)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def as_vectors(value: Any) -> list[list[float]]:
    """Convert common embedding outputs into a list of float vectors."""

    try:
        value = value.tolist()
    except Exception:
        pass
    if not isinstance(value, list):
        return []
    if value and all(isinstance(item, (int, float)) for item in value):
        return [[float(item) for item in value]]
    out: list[list[float]] = []
    for row in value:
        try:
            out.append([float(item) for item in row])
        except Exception:
            continue
    return out


def rank_pages(scores: Sequence[float], k: int) -> tuple[int, ...]:
    """Return page indices sorted by score desc, then page order asc, capped at k."""

    limit = max(0, int(k))
    ranked = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return tuple(ranked[:limit])


def simple_bm25_scores(query_tokens: Sequence[str], page_tokens: Sequence[Sequence[str]]) -> list[float]:
    """Small BM25 used when `rank_bm25` is unavailable."""

    if not page_tokens:
        return []
    query = [token for token in query_tokens if token]
    doc_count = len(page_tokens)
    avg_len = sum(len(tokens) for tokens in page_tokens) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for tokens in page_tokens:
        doc_freq.update(set(tokens))
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for tokens in page_tokens:
        counts = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for token in query:
            freq = counts[token]
            if freq == 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = freq + k1 * (1 - b + b * length / max(avg_len, 1e-9))
            score += idf * freq * (k1 + 1) / denom
        scores.append(score)
    return scores


def render_document_pages(question: Question, page_count: int, *, data_dir: Path, cache_dir: Path,
                          dpi: int, render_images: bool) -> list:
    """Render or extract all pages required for one document ranking."""

    if page_count <= 0:
        return []
    pdf = resolve_pdf(question.doc_id, data_dir)
    return render_pdf(pdf, tuple(range(page_count)), cache_dir=cache_dir, dpi=dpi,
                      render_images=render_images, extract_text=True)


class Retriever(ABC):
    """Rank a document's pages for a question."""

    name: str = "retriever"
    modality: str = ""

    @abstractmethod
    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        """Return page indices ranked most- to least-relevant (at most `k`)."""

    def unload(self) -> None:
        """Release any GPU-resident weights so the reasoner can load. No-op by default."""


class StubRetriever(Retriever):
    """Deterministic placeholder: return the first `k` pages in document order."""

    name = "stub"

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return tuple(range(min(int(k), page_count)))


class MemoizedRetriever(Retriever):
    """Cache rankings from another retriever by question, page count, and `k`."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.name = inner.name
        self.modality = getattr(inner, "modality", "")
        self._cache: dict[tuple[str, int, int], tuple[int, ...]] = {}

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        key = (question.id, int(page_count), int(k))
        if key not in self._cache:
            self._cache[key] = self.inner.retrieve(question, page_count, k)
        return self._cache[key]

    def unload(self) -> None:
        """Drop the inner retriever's model but keep the memoized rankings."""

        self.inner.unload()


DEFAULT_DATA_DIR = DEFAULT_PATHS.data_dir
DEFAULT_CACHE_DIR = DEFAULT_PATHS.cache_dir
