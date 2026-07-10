"""Text page retrievers across cost rungs: BM25, BGE-M3, and Qwen3-Embedding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from retrievers import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    Retriever,
    as_vectors,
    cosine,
    document_page_texts,
    normalise_scores,
    rank_pages,
    simple_bm25_scores,
    tokenize,
)
from schema import Question

BGE_M3_MODEL_ID = "BAAI/bge-m3"
QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-4B"


class Bm25Retriever(Retriever):
    """Cheap lexical rung: BM25 over each page's embedded text (no model)."""

    name = "bm25"
    modality = "text"

    def __init__(self, *, data_dir: Path | None = None, cache_dir: Path | None = None, dpi: int = 96) -> None:
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.dpi = int(dpi)
        # Retrieval cost (pivot 6.3): cumulative per-doc index build time and the
        # last query's scoring time, read by the retrieval side-artifact.
        self.index_build_s = 0.0
        self.last_query_s = 0.0
        # Per-doc BM25 index (page tokens + built index), so building is counted
        # once per document and every question/k reuses it.
        self._index: dict[tuple[str, int], tuple[Any, list[list[str]]]] = {}

    def _page_texts(self, question: Question, page_count: int) -> list[str]:
        return document_page_texts(question, page_count, data_dir=self.data_dir,
                                   cache_dir=self.cache_dir, dpi=self.dpi)

    def _doc_index(self, question: Question, page_count: int) -> tuple[Any, list[list[str]]]:
        key = (question.doc_id, int(page_count))
        cached = self._index.get(key)
        if cached is not None:
            return cached
        start = perf_counter()
        page_tokens = [tokenize(t) for t in self._page_texts(question, page_count)]
        bm25 = None
        if page_tokens:
            try:
                from rank_bm25 import BM25Okapi

                bm25 = BM25Okapi(list(page_tokens))
            except Exception:
                bm25 = None
        self.index_build_s += perf_counter() - start
        entry = (bm25, page_tokens)
        self._index[key] = entry
        return entry

    def rank(self, question: Question, page_count: int) -> tuple[int, ...]:
        bm25, page_tokens = self._doc_index(question, page_count)
        if not page_tokens:
            self.last_query_s = 0.0
            return ()
        start = perf_counter()
        query_tokens = tokenize(question.question)
        if bm25 is not None:
            scores = [float(s) for s in bm25.get_scores(list(query_tokens))]
        else:
            scores = simple_bm25_scores(query_tokens, page_tokens)
        ranked = rank_pages(normalise_scores(scores), page_count)
        self.last_query_s = perf_counter() - start
        return ranked

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return self.rank(question, page_count)[: int(k)]


class DenseTextRetriever(Retriever):
    """Dense rung: cosine similarity of a query embedding against page embeddings.

    Subclasses set `model_id` and provide `_encode`; both dense rungs (BGE-M3 and
    Qwen3-Embedding) share this ranking body and load their embedder lazily so
    importing this module pulls no model.
    """

    modality = "text"
    model_id = ""

    def __init__(self, *, data_dir: Path | None = None, cache_dir: Path | None = None,
                 dpi: int = 96, embedder: Any | None = None, allow_bm25_fallback: bool = True) -> None:
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.dpi = int(dpi)
        self.embedder = embedder
        self.allow_bm25_fallback = bool(allow_bm25_fallback)
        # Page embeddings are query-independent, so they are encoded once per
        # document and reused across every question and k.
        self._page_emb: dict[tuple[str, int], list[list[float]]] = {}
        # Retrieval cost (pivot 6.3): cumulative page-embed build time, last query time.
        self.index_build_s = 0.0
        self.last_query_s = 0.0

    def _load_embedder(self) -> Any:
        """Load the sentence embedder lazily (subclass-specific)."""

        raise NotImplementedError

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        embedder = self.embedder if self.embedder is not None else self._load_embedder()
        self.embedder = embedder
        return as_vectors(embedder.encode(list(texts), batch_size=8))

    def unload(self) -> None:
        """Drop the embedder so it frees the GPU before the reasoner loads."""

        self.embedder = None

    def _page_embeddings(self, question: Question, page_texts: Sequence[str], page_count: int) -> list[list[float]]:
        key = (question.doc_id, int(page_count))
        cached = self._page_emb.get(key)
        if cached is not None:
            return cached
        start = perf_counter()
        vectors = self._encode(list(page_texts))
        self.index_build_s += perf_counter() - start
        self._page_emb[key] = vectors
        return vectors

    def rank(self, question: Question, page_count: int) -> tuple[int, ...]:
        page_texts = self._page_texts(question, page_count)
        if not page_texts:
            self.last_query_s = 0.0
            return ()
        try:
            # Page-embed build is timed inside _page_embeddings (index cost), so the
            # query timer below only covers the query encode + scoring.
            page_vectors = self._page_embeddings(question, page_texts, page_count)
            start = perf_counter()
            query_vectors = self._encode([question.question])
            if len(page_vectors) != len(page_texts) or not query_vectors:
                raise RuntimeError("embedder returned an unexpected shape")
            scores = [cosine(query_vectors[0], v) for v in page_vectors]
            self.last_query_s = perf_counter() - start
        except Exception:
            if not self.allow_bm25_fallback:
                raise
            start = perf_counter()
            scores = normalise_scores(
                simple_bm25_scores(tokenize(question.question), [tokenize(t) for t in page_texts])
            )
            self.last_query_s = perf_counter() - start
        return rank_pages(normalise_scores(scores), page_count)

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        return self.rank(question, page_count)[: int(k)]

    def _page_texts(self, question: Question, page_count: int) -> list[str]:
        return document_page_texts(question, page_count, data_dir=self.data_dir,
                                   cache_dir=self.cache_dir, dpi=self.dpi)


class BgeM3Retriever(DenseTextRetriever):
    """Mid dense rung: BGE-M3 (self-hosted workhorse)."""

    name = "bge-m3"
    model_id = BGE_M3_MODEL_ID

    def _load_embedder(self) -> Any:
        from FlagEmbedding import BGEM3FlagModel

        try:
            import torch

            use_fp16 = bool(torch.cuda.is_available())
        except Exception:
            use_fp16 = False
        model = BGEM3FlagModel(self.model_id, use_fp16=use_fp16)

        class _Wrap:
            def encode(self, texts, batch_size=8):
                out = model.encode(list(texts), batch_size=batch_size)
                return out["dense_vecs"] if isinstance(out, dict) else out

        return _Wrap()


class Qwen3EmbeddingRetriever(DenseTextRetriever):
    """Expensive dense rung: Qwen3-Embedding-4B (SOTA-class)."""

    name = "qwen3-embedding"
    model_id = QWEN3_EMBEDDING_MODEL_ID

    def _load_embedder(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_id)


TEXT_RETRIEVERS = {
    "bm25": Bm25Retriever,
    "bge-m3": BgeM3Retriever,
    "qwen3-embedding": Qwen3EmbeddingRetriever,
}


def get_text_retriever(name: str, **kwargs: Any) -> Retriever:
    """Return a text retriever by cost-rung name."""

    if name not in TEXT_RETRIEVERS:
        raise KeyError(f"unknown text retriever {name!r}; use one of {tuple(TEXT_RETRIEVERS)}")
    return TEXT_RETRIEVERS[name](**kwargs)
