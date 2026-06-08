"""Retrieval-recall evaluation WITHOUT the generator (plan Stage 4).

Fast + local-able: validate every retriever on recall@k before spending compute
on downstream accuracy. A retriever scoring ~0 recall where evidence exists is
broken — catch it here. Recall is reported over ANSWERABLE questions only
(those that actually have evidence pages); unanswerable questions have nothing
to retrieve and would otherwise inflate recall to 1.0.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..data.dataset import Dataset
from ..eval.metrics import recall_at_k
from .base import EvidenceSelector


def _set_depth(selector: EvidenceSelector, depth: int) -> dict:
    """Temporarily widen a selector to return up to `depth` pages."""
    saved = {}
    for attr in ("top_k", "component_k"):
        if hasattr(selector, attr):
            saved[attr] = getattr(selector, attr)
            setattr(selector, attr, max(getattr(selector, attr), depth))
    return saved


def _restore(selector: EvidenceSelector, saved: dict) -> None:
    for attr, val in saved.items():
        setattr(selector, attr, val)


def evaluate_retrieval(selector: EvidenceSelector, dataset: Dataset,
                       ks: Iterable[int] = (1, 2, 4, 8)) -> dict:
    ks = sorted(set(ks))
    max_k = max(ks)
    saved = _set_depth(selector, max_k)

    recalls: dict[int, list[float]] = {k: [] for k in ks}
    rows = []
    try:
        for q in dataset.questions:
            doc = dataset.get_document(q.doc_id)
            pages = selector.select(q, doc).page_indices
            per_q = {}
            for k in ks:
                r = recall_at_k(pages, q.evidence_pages_zero_based, k=k)
                per_q[k] = r
                if q.evidence_pages:        # answerable only
                    recalls[k].append(r)
            rows.append({
                "qid": q.qid, "doc_id": q.doc_id,
                "question_type": q.question_type.value,
                "evidence_pages_0based": q.evidence_pages_zero_based,
                "ranked_pages": pages,
                "recall_at_k": per_q,
                "evidence_sources": q.evidence_sources,
            })
    finally:
        _restore(selector, saved)

    table = {k: (sum(v) / len(v) if v else 0.0) for k, v in recalls.items()}
    n_answerable = len(recalls[ks[0]])
    return {"selector": selector.name, "recall_at_k": table,
            "n_answerable": n_answerable, "rows": rows}
