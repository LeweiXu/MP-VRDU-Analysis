"""Text page retrievers across cost rungs: BM25, BGE-M3, and Qwen3-Embedding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retrievers import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    Retriever,
    as_vectors,
    cosine,
    normalise_scores,
    rank_pages,
    render_document_pages,
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

    def _page_texts(self, question: Question, page_count: int) -> list[str]:
        pages = render_document_pages(question, page_count, data_dir=self.data_dir,
                                      cache_dir=self.cache_dir, dpi=self.dpi, render_images=False)
        return [page.text for page in pages]

    def _bm25(self, query_tokens: Sequence[str], page_tokens: Sequence[Sequence[str]]) -> list[float]:
        try:
            from rank_bm25 import BM25Okapi

            return [float(s) for s in BM25Okapi(list(page_tokens)).get_scores(list(query_tokens))]
        except Exception:
            return simple_bm25_scores(query_tokens, page_tokens)

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        page_texts = self._page_texts(question, page_count)
        if not page_texts:
            return ()
        scores = self._bm25(tokenize(question.question), [tokenize(t) for t in page_texts])
        return rank_pages(normalise_scores(scores), k)


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

    def retrieve(self, question: Question, page_count: int, k: int) -> tuple[int, ...]:
        pages = render_document_pages(question, page_count, data_dir=self.data_dir,
                                      cache_dir=self.cache_dir, dpi=self.dpi, render_images=False)
        page_texts = [page.text for page in pages]
        if not page_texts:
            return ()
        try:
            vectors = self._encode([question.question, *page_texts])
            if len(vectors) != len(page_texts) + 1:
                raise RuntimeError("embedder returned an unexpected shape")
            scores = [cosine(vectors[0], v) for v in vectors[1:]]
        except Exception:
            if not self.allow_bm25_fallback:
                raise
            scores = normalise_scores(
                simple_bm25_scores(tokenize(question.question), [tokenize(t) for t in page_texts])
            )
        return rank_pages(normalise_scores(scores), k)


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
