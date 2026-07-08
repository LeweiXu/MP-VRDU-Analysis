"""Resolves and samples a run's question set: document-coherent per-bin
sampling, the full/limit/id sampling modes, and the answerable pool a task draws
from."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

# A task draws only from its pool, so a spec cannot cross-contaminate: the
# hallucination task lives on the unanswerable questions, everything else on the
# answerable ones.
UNANSWERABLE_TASKS = frozenset({"G3_hallucination"})


def pool_for_task(task_name: str) -> str:
    """Return the question pool a task draws from: `answerable` or `unanswerable`."""

    return "unanswerable" if task_name in UNANSWERABLE_TASKS else "answerable"


def filter_by_pool(corpus: Sequence[Any], pool: str) -> list[Any]:
    """Keep only the answerable or unanswerable questions of a corpus."""

    if pool == "unanswerable":
        return [q for q in corpus if q.is_unanswerable]
    if pool == "answerable":
        return [q for q in corpus if not q.is_unanswerable]
    raise ValueError(f"pool must be 'answerable' or 'unanswerable', got {pool!r}")


def _draw_documents(bin_questions: Sequence[Any], target: int | None, seed: int) -> set[str]:
    """Question ids of whole documents summing to about `target` questions.

    Questions cluster within a document, so the draw is at the document level:
    documents are shuffled by `seed` and added whole until the bin reaches
    `target`. A bin already at or below `target` (or an unset target) is kept
    whole. Drawing whole documents is what keeps the doc-level bootstrap valid.
    """

    if target is None or len(bin_questions) <= target:
        return {q.id for q in bin_questions}

    by_doc: dict[str, list[Any]] = {}
    for question in bin_questions:
        by_doc.setdefault(question.doc_id, []).append(question)

    doc_ids = list(by_doc)
    random.Random(seed).shuffle(doc_ids)

    kept: set[str] = set()
    count = 0
    for doc_id in doc_ids:
        if count >= target:
            break
        kept.update(question.id for question in by_doc[doc_id])
        count += len(by_doc[doc_id])
    return kept


def sample_per_bin(corpus: Sequence[Any], per_bin: int | None, seed: int = 0) -> list[Any]:
    """Subset to about `per_bin` questions per bin_label by drawing whole documents.

    Within each bin, documents are shuffled by `seed` and taken whole until the
    bin reaches `per_bin` questions; the returned list preserves the original
    corpus order. A different `seed` yields a different (largely disjoint) subset.
    """

    grouped: dict[str, list[Any]] = {}
    for question in corpus:
        grouped.setdefault(question.bin_label, []).append(question)

    keep_ids: set[str] = set()
    for bin_questions in grouped.values():
        keep_ids |= _draw_documents(bin_questions, per_bin, seed)

    return [question for question in corpus if question.id in keep_ids]


def resolve_corpus(spec: Mapping[str, Any], corpus: Sequence[Any]) -> list[Any]:
    """Apply a spec's `corpus` block to a question list.

    Modes: `full` (everything), `{per_bin: N, seed: S}` (document-coherent
    subset), `{limit: N}` or `{ids: [...]}` (a fast smoke/debug slice).
    """

    sampling: Any = spec.get("sampling", "full") if isinstance(spec, Mapping) else spec
    if sampling in (None, "full"):
        return list(corpus)
    if isinstance(sampling, Mapping):
        if "limit" in sampling:
            return list(corpus)[: max(0, int(sampling["limit"]))]
        if "ids" in sampling:
            wanted = {str(i) for i in sampling["ids"]}
            return [q for q in corpus if q.id in wanted]
        if "per_bin" in sampling:
            return sample_per_bin(corpus, int(sampling["per_bin"]), int(sampling.get("seed", 0)))
    raise ValueError(f"unrecognised corpus sampling: {sampling!r}")
