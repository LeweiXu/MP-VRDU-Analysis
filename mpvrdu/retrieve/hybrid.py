"""Hybrid retrieval via Reciprocal Rank Fusion (Stage 4, built LAST).

Fusion happens at the PAGE level in selector space, not at the unit level. This
makes hybrid work uniformly whether the fused methods share a modality
(sparse+dense, both text) or cross it (text+visual) — each sub-selector builds
its own units and we fuse the resulting ranked page lists. RRF:

    score(page) = sum_over_selectors 1 / (rrf_k + rank_in_that_selector)
"""

from __future__ import annotations

from ..data.dataset import Document, Question
from .base import EvidenceSelector, Selection


class HybridSelector(EvidenceSelector):
    name = "hybrid"

    def __init__(self, selectors: list[EvidenceSelector], top_k: int,
                 rrf_k: int = 60, component_k: int = 20):
        if len(selectors) < 2:
            raise ValueError("hybrid needs >= 2 selectors")
        self.selectors = selectors
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.component_k = component_k  # how deep each component contributes

    def select(self, question: Question, document: Document) -> Selection:
        fused: dict[int, float] = {}
        for sel in self.selectors:
            pages = sel.select(question, document).page_indices[: self.component_k]
            for rank, page in enumerate(pages):
                fused[page] = fused.get(page, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
        return Selection(
            page_indices=[p for p, _ in ranked],
            scores=[s for _, s in ranked],
            meta={"selector": self.name,
                  "components": [s.name for s in self.selectors]},
        )

    def unload(self) -> None:
        for sel in self.selectors:
            sel.unload()
