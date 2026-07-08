"""Define page-ranking retrievers for text and vision covariates.

Purpose:
    A `Retriever` ranks document pages for a question so `RetrievedTopK` can feed
    realistic page sets instead of oracle evidence. Retrieval is a measured
    covariate in the paper: it determines whether the pipeline can locate the
    page before the representation/reasoner tries to answer.

Pipeline role:
    Input conditioning calls `retrieve(question, page_count, k)` and retrieval
    metrics compare returned page indices with gold evidence pages. This module
    keeps the frozen method signature while adding concrete Stage-M6
    implementations:

    - `BM25BGERetriever` scores page text with BM25 plus optional BGE embeddings.
    - `ColQwenRetriever` scores rendered page images with an optional ColQwen
      multi-vector model, falling back to a deterministic text/order heuristic
      when the heavy scorer is unavailable.
    - `MemoizedRetriever` wraps any retriever so metrics and conditioners can
      share one expensive ranking call.

Arguments:
    None. This module is import-only; callers instantiate retriever classes with
    optional root-relative data/cache paths, injected test scorers, and model ids,
    then call `retrieve()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any

from config import DEFAULT_PATHS
from data.loader import resolve_pdf
from data.render import render_pdf
from schema import Question


TEXT_RETRIEVER_MODEL_ID = "BAAI/bge-small-en-v1.5"
VISION_RETRIEVER_MODEL_ID = "vidore/colqwen2.5-v0.2"


def _tokenize(text: str) -> list[str]:
    """Return a compact lower-case token stream for lexical ranking."""

    return re.findall(r"[A-Za-z0-9]+", text.casefold())


def _normalise_scores(scores: Sequence[float]) -> list[float]:
    """Scale finite scores to [0, 1], preserving ties as zeros."""

    finite = [float(score) if math.isfinite(float(score)) else 0.0 for score in scores]
    if not finite:
        return []
    lo = min(finite)
    hi = max(finite)
    if hi <= lo:
        return [0.0 for _ in finite]
    return [(score - lo) / (hi - lo) for score in finite]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
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


def _as_vectors(value: Any) -> list[list[float]]:
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
        if isinstance(row, list):
            out.append([float(item) for item in row])
        else:
            try:
                out.append([float(item) for item in row])
            except Exception:
                continue
    return out


def _rank_pages(scores: Sequence[float], k: int) -> tuple[int, ...]:
    """Return page indices sorted by score desc, then page order asc."""

    limit = max(0, int(k))
    ranked = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return tuple(ranked[:limit])


def _render_document_pages(
    question: Question,
    page_count: int,
    *,
    data_dir: Path,
    cache_dir: Path,
    dpi: int,
    render_images: bool,
) -> list:
    """Render or extract all pages required for one document ranking."""

    if page_count <= 0:
        return []
    pdf = resolve_pdf(question.doc_id, data_dir)
    return render_pdf(
        pdf,
        tuple(range(page_count)),
        cache_dir=cache_dir,
        dpi=dpi,
        render_images=render_images,
        extract_text=True,
    )


def _simple_bm25_scores(query_tokens: Sequence[str], page_tokens: Sequence[Sequence[str]]) -> list[float]:
    """Small BM25 implementation used if `rank_bm25` is unavailable."""

    if not page_tokens:
        return []
    query = [token for token in query_tokens if token]
    doc_count = len(page_tokens)
    avg_len = sum(len(tokens) for tokens in page_tokens) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for tokens in page_tokens:
        doc_freq.update(set(tokens))
    k1 = 1.5
    b = 0.75
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


class Retriever(ABC):
    """Rank a document's pages for a question."""

    name: str = "retriever"

    @abstractmethod
    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        """Return page indices ranked most- to least-relevant (at most `k`)."""

    def unload(self) -> None:
        """Release any GPU-resident model weights (no-op by default).

        The generate phase calls this after warming the retrieval cache so the
        retriever's weights free the GPU before the reasoner loads. Any cached
        rankings survive, so later cache-hit `retrieve` calls need no model.
        """


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
        self._cache: dict[tuple[str, int, int], tuple[int, ...]] = {}

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        key = (question.id, int(page_count), int(k))
        if key not in self._cache:
            self._cache[key] = self.inner.retrieve(question, page_count, k)
        return self._cache[key]

    def unload(self) -> None:
        """Drop the inner retriever's model but keep the memoized rankings."""

        self.inner.unload()


class BM25BGERetriever(Retriever):
    """Rank pages using BM25 plus optional BGE dense similarity over page text."""

    name = "bm25_bge_text"
    modality = "text"

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
        dpi: int = 96,
        model_id: str = TEXT_RETRIEVER_MODEL_ID,
        embedder: Any | None = None,
        use_bge: bool = True,
        allow_bm25_fallback: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir or DEFAULT_PATHS.data_dir)
        self.cache_dir = Path(cache_dir or DEFAULT_PATHS.cache_dir)
        self.dpi = int(dpi)
        self.model_id = model_id
        self.embedder = embedder
        self.use_bge = bool(use_bge)
        self.allow_bm25_fallback = bool(allow_bm25_fallback)

    def _bm25_scores(self, query_tokens: Sequence[str], page_tokens: Sequence[Sequence[str]]) -> list[float]:
        """Return lexical BM25 scores using the package when present."""

        try:
            from rank_bm25 import BM25Okapi

            return [float(score) for score in BM25Okapi(list(page_tokens)).get_scores(list(query_tokens))]
        except Exception:
            return _simple_bm25_scores(query_tokens, page_tokens)

    def _load_embedder(self) -> Any:
        """Load the configured BGE embedder lazily."""

        if self.embedder is None:
            from FlagEmbedding import FlagModel

            try:
                import torch

                use_fp16 = bool(torch.cuda.is_available())
            except Exception:
                use_fp16 = False
            self.embedder = FlagModel(self.model_id, use_fp16=use_fp16)
        return self.embedder

    def unload(self) -> None:
        """Drop the BGE embedder so it stops holding GPU memory."""

        self.embedder = None

    def _bge_scores(self, query: str, page_texts: Sequence[str]) -> list[float]:
        """Return dense query/page cosine scores, or raise if BGE is unavailable."""

        embedder = self._load_embedder()
        vectors = _as_vectors(embedder.encode([query, *page_texts], batch_size=8))
        if len(vectors) != len(page_texts) + 1:
            raise RuntimeError("BGE embedder returned an unexpected embedding shape")
        query_vector = vectors[0]
        return [_cosine(query_vector, vector) for vector in vectors[1:]]

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        pages = _render_document_pages(
            question,
            page_count,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            dpi=self.dpi,
            render_images=False,
        )
        page_texts = [page.text for page in pages]
        if not page_texts:
            return ()
        query_tokens = _tokenize(question.question)
        page_tokens = [_tokenize(text) for text in page_texts]
        bm25 = _normalise_scores(self._bm25_scores(query_tokens, page_tokens))
        bge = [0.0 for _ in page_texts]
        if self.use_bge:
            try:
                bge = _normalise_scores(self._bge_scores(question.question, page_texts))
            except Exception:
                if not self.allow_bm25_fallback:
                    raise
        combined = [0.55 * bm25[index] + 0.45 * bge[index] for index in range(len(page_texts))]
        return _rank_pages(combined, k)


@dataclass(frozen=True)
class VisionPageScore:
    """One ColQwen/fallback score for a rendered page."""

    page_index: int
    score: float


class ColQwenRetriever(Retriever):
    """Rank rendered page images with ColQwen or an injected compatible scorer."""

    name = "colqwen_vision"
    modality = "vision"

    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
        dpi: int = 96,
        model_id: str = VISION_RETRIEVER_MODEL_ID,
        scorer: Callable[[Question, Sequence[Any]], Sequence[float]] | None = None,
        allow_text_fallback: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir or DEFAULT_PATHS.data_dir)
        self.cache_dir = Path(cache_dir or DEFAULT_PATHS.cache_dir)
        self.dpi = int(dpi)
        self.model_id = model_id
        self.scorer = scorer
        self.allow_text_fallback = bool(allow_text_fallback)
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load_colqwen(self) -> tuple[Any, Any]:
        """Load ColQwen model/processor classes matching the configured repo."""

        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        import torch
        from colpali_engine.models import (
            ColQwen2,
            ColQwen2Processor,
            ColQwen2_5,
            ColQwen2_5_Processor,
        )
        from transformers.utils.import_utils import is_flash_attn_2_available

        if "2.5" in self.model_id or "qwen2_5" in self.model_id.casefold():
            model_cls = ColQwen2_5
            processor_cls = ColQwen2_5_Processor
        else:
            model_cls = ColQwen2
            processor_cls = ColQwen2Processor

        cuda = bool(torch.cuda.is_available())
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if cuda else torch.float32,
            "device_map": "cuda:0" if cuda else "cpu",
        }
        if cuda and is_flash_attn_2_available():
            kwargs["attn_implementation"] = "flash_attention_2"
        self._model = model_cls.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = processor_cls.from_pretrained(self.model_id)
        return self._model, self._processor

    def unload(self) -> None:
        """Drop the ColQwen weights/processor so they free the GPU."""

        self._model = None
        self._processor = None

    def _colqwen_scores(self, question: Question, pages: Sequence[Any]) -> list[float]:
        """Return ColQwen multi-vector scores for the rendered page images."""

        import torch
        from PIL import Image

        model, processor = self._load_colqwen()
        images = [Image.open(page.image_path).convert("RGB") for page in pages if page.image_path]
        if len(images) != len(pages):
            raise ValueError("all pages need image_path for ColQwen retrieval")
        batch_images = processor.process_images(images).to(model.device)
        batch_queries = processor.process_queries([question.question]).to(model.device)
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            query_embeddings = model(**batch_queries)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        try:
            row = scores[0].tolist()
        except Exception:
            row = scores.tolist()[0]
        return [float(score) for score in row]

    def _fallback_scores(self, question: Question, pages: Sequence[Any]) -> list[float]:
        """Return a deterministic image-smoke fallback score from page text/order."""

        query = set(_tokenize(question.question))
        scores: list[float] = []
        for page in pages:
            overlap = len(query.intersection(_tokenize(page.text)))
            # Prefer lexical overlap, then earlier pages. This keeps smoke tests
            # deterministic without claiming to be a vision model.
            scores.append(float(overlap) + 1.0 / (1 + int(page.index)))
        return scores

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        pages = _render_document_pages(
            question,
            page_count,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            dpi=self.dpi,
            render_images=True,
        )
        if not pages:
            return ()
        try:
            raw_scores = (
                list(self.scorer(question, pages))
                if self.scorer is not None
                else self._colqwen_scores(question, pages)
            )
        except Exception:
            if not self.allow_text_fallback:
                raise
            raw_scores = self._fallback_scores(question, pages)
        scores = _normalise_scores(raw_scores)
        return _rank_pages(scores, k)
