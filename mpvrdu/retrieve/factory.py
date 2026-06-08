"""Selector factory: RunConfig -> EvidenceSelector (Stage 1-4 dispatch)."""

from __future__ import annotations

from ..config import RunConfig
from .base import (EvidenceSelector, NoRetrieval, Oracle, RetrieverSelector)
from .hybrid import HybridSelector


def _retriever_selector(method: str, cfg: RunConfig) -> RetrieverSelector:
    from .retrievers import build_retriever

    retriever = build_retriever(method, cfg.retrieval)
    return RetrieverSelector(
        retriever=retriever,
        top_k=cfg.retrieval.top_k,
        parser_name=cfg.representation.parser,
        dpi=cfg.representation.dpi,
        chunking=cfg.representation.chunking,
    )


def build_selector(cfg: RunConfig) -> EvidenceSelector:
    method = cfg.retrieval.method

    if method == "none":
        return NoRetrieval(n_pages=cfg.retrieval.no_retrieval_pages)
    if method == "oracle":
        return Oracle()
    if method in {"bm25", "tfidf", "dense", "colpali", "colqwen"}:
        return _retriever_selector(method, cfg)
    if method == "hybrid":
        subs = [_retriever_selector(m, cfg) for m in cfg.retrieval.hybrid_methods]
        return HybridSelector(subs, top_k=cfg.retrieval.top_k,
                              rrf_k=cfg.retrieval.rrf_k)
    raise ValueError(f"unknown retrieval method {method!r}")
