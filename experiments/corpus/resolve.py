"""Resolves and samples a run's question set: document-coherent per-bin
sampling, the full/limit/id sampling modes, and the answerable pool a task draws
from."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger("mpvrdu.corpus")

# A task draws only from its pool, so a spec cannot cross-contaminate: the
# hallucination task lives on the unanswerable questions, everything else on the
# answerable ones.
UNANSWERABLE_TASKS = frozenset({"G3_hallucination"})


def pool_for_task(task_name: str) -> str:
    """Return the question pool a task draws from: `answerable` or `unanswerable`."""

    return "unanswerable" if task_name in UNANSWERABLE_TASKS else "answerable"


def filter_by_pool(corpus: Sequence[Any], pool: str) -> list[Any]:
    """Keep the answerable, unanswerable, or (`all`) both questions of a corpus."""

    if pool in ("all", None, ""):
        return list(corpus)
    if pool == "unanswerable":
        return [q for q in corpus if q.is_unanswerable]
    if pool == "answerable":
        return [q for q in corpus if not q.is_unanswerable]
    raise ValueError(f"pool must be 'all', 'answerable', or 'unanswerable', got {pool!r}")


def filter_by_hop(corpus: Sequence[Any], hop: str) -> list[Any]:
    """Keep questions by gold-evidence-page count: `single`, `multi`, or `any`.

    `hop == "none"` questions (no gold pages) are excluded by both `single` and
    `multi`: gold-removal page rules are undefined without a gold set. Required
    by page_set rules that remove or isolate a gold page, where a one-gold
    question makes top and bottom coincide.
    """

    if hop in ("any", None, ""):
        return list(corpus)
    if hop not in ("single", "multi"):
        raise ValueError(f"hop must be 'any', 'single', or 'multi', got {hop!r}")
    return [q for q in corpus if q.hop == hop]


def auto_scan_labels(doc_ids, data_dir, csv_path) -> dict[str, str]:
    """Map each doc_id to a PyMuPDF-detected `digital`/`scanned` label, cached to CSV.

    Reads any existing `annotations/auto_scan.csv` (columns `doc_id,auto_scan`),
    classifies only the doc_ids missing from it (via `data.render.classify_scanned`,
    the same detector the annotation script uses), and writes the accumulated cache
    back. A document whose PDF can't be resolved gets an empty label (matches no
    filter).
    """

    import csv
    from pathlib import Path

    from data.loader import resolve_pdf
    from data.render import classify_scanned

    csv_path = Path(csv_path)
    cache: dict[str, str] = {}
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                doc_id = (row.get("doc_id") or "").strip()
                if doc_id:
                    cache[doc_id] = (row.get("auto_scan") or "").strip()

    missing = [d for d in dict.fromkeys(doc_ids) if d not in cache]
    for doc_id in sorted(missing):
        try:
            cache[doc_id] = classify_scanned(resolve_pdf(doc_id, data_dir)).label
        except Exception as exc:  # noqa: BLE001 - an unresolvable PDF is an unknown label, not a crash
            log.warning("auto-scan: could not classify %s: %s", doc_id, exc)
            cache[doc_id] = ""

    if missing:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["doc_id", "auto_scan"])
            for doc_id in sorted(cache):
                writer.writerow([doc_id, cache[doc_id]])
    return cache


def filter_by_scan(corpus: Sequence[Any], scan: str, *, data_dir, annotations_dir) -> list[Any]:
    """Keep only questions whose document is `digital` or `scanned` (auto-detected).

    Applied before the pool + sampling. `any` (or empty) is a no-op. The label comes
    from `auto_scan_labels`, cached in `annotations/auto_scan.csv`.
    """

    from pathlib import Path

    if scan in (None, "", "any"):
        return list(corpus)
    if scan not in ("digital", "scanned"):
        raise ValueError(f"scan must be 'any', 'digital', or 'scanned', got {scan!r}")
    labels = auto_scan_labels({q.doc_id for q in corpus}, data_dir, Path(annotations_dir) / "auto_scan.csv")
    return [q for q in corpus if labels.get(q.doc_id) == scan]


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


def sample_per_doc_type(corpus: Sequence[Any], per_doc_type: int | None, seed: int = 0) -> list[Any]:
    """Subset to EXACTLY `per_doc_type` questions per doc_type label.

    Documents are shuffled by `seed` and taken whole (the doc-coherent draw shared
    with `sample_per_bin`) until the group reaches `per_doc_type` questions, then the
    group is capped to exactly that many (in corpus order). So `per_doc_type: 1` runs
    one question per label (seven labels -> seven questions), which the doc-coherent
    draw alone could not do (it keeps whole documents, usually several questions). A
    label with fewer than `per_doc_type` questions is kept whole. The returned list
    preserves the original corpus order.

    The exact cap can slice the last drawn document (a partial document), which the
    plain whole-document draw never does; keep that in mind for the doc-level
    bootstrap on small `per_doc_type`.
    """

    if per_doc_type is None:
        return list(corpus)

    grouped: dict[str, list[Any]] = {}
    for question in corpus:
        grouped.setdefault(question.doc_type, []).append(question)

    keep_ids: set[str] = set()
    for group in grouped.values():
        drawn_ids = _draw_documents(group, per_doc_type, seed)
        drawn = [question for question in group if question.id in drawn_ids]  # corpus order
        keep_ids |= {question.id for question in drawn[:per_doc_type]}

    return [question for question in corpus if question.id in keep_ids]


def resolve_corpus(spec: Mapping[str, Any], corpus: Sequence[Any]) -> list[Any]:
    """Apply a spec's `corpus` block to a question list.

    Modes: `full` (everything), `{per_bin: N, seed: S}` or `{per_doc_type: N,
    seed: S}` (document-coherent subsets), `{limit: N}` or `{ids: [...]}` (a fast
    smoke/debug slice).
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
        if "per_doc_type" in sampling:
            return sample_per_doc_type(corpus, int(sampling["per_doc_type"]), int(sampling.get("seed", 0)))
    raise ValueError(f"unrecognised corpus sampling: {sampling!r}")
