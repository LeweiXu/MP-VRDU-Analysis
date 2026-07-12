"""Page retrievers arranged as cost rungs: the Retriever base, shared ranking
helpers, and memoization."""

from __future__ import annotations

import json
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


# A document's per-page embedded text is query-independent, so it is parsed from
# the PDF once per (doc, page count) and reused across every question and k.
_PAGE_TEXT_CACHE: dict[tuple[str, int], list[str]] = {}


def document_page_texts(question: Question, page_count: int, *, data_dir: Path, cache_dir: Path,
                        dpi: int) -> list[str]:
    """Return each page's embedded text for a document, cached per document.

    The text layer does not depend on the question or `k`, so extracting it once
    per document (rather than per question x k) is what removes the repeated PDF
    re-parse the lexical and dense text retrievers used to pay.
    """

    key = (question.doc_id, int(page_count))
    cached = _PAGE_TEXT_CACHE.get(key)
    if cached is not None:
        return cached
    pages = render_document_pages(question, page_count, data_dir=data_dir, cache_dir=cache_dir,
                                  dpi=dpi, render_images=False)
    texts = [page.text for page in pages]
    _PAGE_TEXT_CACHE[key] = texts
    return texts


class Retriever(ABC):
    """Rank a document's pages for a question."""

    name: str = "retriever"
    modality: str = ""

    @abstractmethod
    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        """Return page indices ranked most- to least-relevant (at most `k`)."""

    def rank(self, question: Question, page_count: int) -> tuple[int, ...]:
        """Return every page ranked, most- to least-relevant (k-independent).

        Ranking a document does not depend on `k` (only the final slice does), so
        a k-sweep computes the ranking once here and slices per k. The default
        derives it from `retrieve`; concrete retrievers override to compute the
        query embedding/scores a single time per document.
        """

        return self.retrieve(question, page_count, int(page_count))

    def unload(self) -> None:
        """Release any GPU-resident weights so the reasoner can load. No-op by default."""


class StubRetriever(Retriever):
    """Deterministic placeholder: return the first `k` pages in document order."""

    name = "stub"

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return tuple(range(min(int(k), page_count)))


class RetrievalMemoMiss(RuntimeError):
    """A reuse-only memoized retriever was asked for a question its memo lacks.

    Raised so the inference cell records this as its failure reason (and rides on)
    instead of the retriever silently re-ranking on a `--skip-retrieval` pass.
    """


class MemoizedRetriever(Retriever):
    """Cache a retriever's full ranking per (question, page count), slice per `k`.

    A k-sweep asks for the same document ranked at k in {1,3,5,7,10}; the ranking
    is `k`-independent, so it is computed once and every k is a slice. With a
    `persist_dir` the full rankings are also written to disk keyed by the
    retriever name and dpi, so a later process (a failed-only re-run, or a task
    reusing another's retrieval) reads them back instead of recomputing.

    `reuse_only` makes a memo miss raise `RetrievalMemoMiss` rather than re-ranking,
    for an inference pass meant to consume an existing memo (`--skip-retrieval`).
    """

    def __init__(self, inner: Retriever, *, persist_dir: Path | str | None = None,
                 reuse_only: bool = False) -> None:
        self.inner = inner
        self.name = inner.name
        self.modality = getattr(inner, "modality", "")
        self.dpi = int(getattr(inner, "dpi", 0))
        self.reuse_only = bool(reuse_only)
        self._cache: dict[tuple[str, int], tuple[int, ...]] = {}
        # Questions recorded as failed (a status row, no ranking) -> the reason; tracked
        # so a failure is written once and its reason can be surfaced on a reuse miss.
        self._failed: dict[tuple[str, int], str] = {}
        self._persist_path: Path | None = None
        if persist_dir is not None:
            self._persist_path = Path(persist_dir) / f"{self.name}__dpi{self.dpi}.jsonl"
            self._load_persisted()

    def _load_persisted(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        for line in self._persist_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (record["question_id"], int(record["page_count"]))
            if record.get("status", "ok") != "ok":
                self._failed[key] = record.get("skipped_reason", "")  # no valid ranking to cache
                continue
            self._cache[key] = tuple(record["ranking"])

    def _persist(self, question_id: str, page_count: int, ranking: tuple[int, ...],
                 seq_stats: dict | None = None) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"question_id": question_id, "page_count": int(page_count), "ranking": list(ranking)}
        if seq_stats:
            # Truncation telemetry from a dense text retriever; extra keys are ignored
            # by _load_persisted, so this is additive.
            row.update(seq_stats)
        with self._persist_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def persist_failure(self, question_id: str, page_count: int, status: str, reason: str) -> None:
        """Record a question the inner retriever could not rank (e.g. an OOM) as a memo
        row carrying `status` + `skipped_reason`, mirroring a failed predictions cell.

        Written once per (question, page count): a question already ranked or already
        recorded as failed is left alone, so repeated resumes do not pile up rows.
        """

        key = (question_id, int(page_count))
        if key in self._cache or key in self._failed:
            return
        self._failed[key] = reason
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._persist_path.open("a") as handle:
            handle.write(json.dumps(
                {"question_id": question_id, "page_count": int(page_count),
                 "ranking": [], "status": status, "skipped_reason": reason},
                sort_keys=True,
            ) + "\n")

    def rank(self, question: Question, page_count: int) -> tuple[int, ...]:
        key = (question.id, int(page_count))
        if key not in self._cache:
            if self.reuse_only:
                earlier = self._failed.get(key)
                detail = f"retrieval failed earlier: {earlier}" if earlier else "no ranking in the retrieval memo"
                raise RetrievalMemoMiss(
                    f"{self.name}: {detail} for question {question.id} "
                    f"(reuse-only, e.g. --skip-retrieval; complete the memo or drop --skip-retrieval)"
                )
            ranking = tuple(self.inner.rank(question, int(page_count)))
            self._cache[key] = ranking
            self._persist(question.id, int(page_count), ranking,
                          seq_stats=getattr(self.inner, "last_seq_stats", None))
        return self._cache[key]

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return self.rank(question, int(page_count))[: int(k)]

    def unload(self) -> None:
        """Drop the inner retriever's model but keep the memoized rankings."""

        self.inner.unload()


DEFAULT_DATA_DIR = DEFAULT_PATHS.data_dir
DEFAULT_CACHE_DIR = DEFAULT_PATHS.cache_dir
