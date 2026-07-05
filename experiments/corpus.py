"""Resolve the question set an experiment runs on, for smoke or full.

Purpose:
    One place that turns an `ExperimentConfig` into the list of `Question`s to
    run, so every experiment is corpus-agnostic and identical between the smoke
    and full runs. Smoke → the frozen ~7-doc smoke corpus; full → the whole
    MMLongBench-Doc set (optionally capped for a gate pilot).

Pipeline role:
    `experiments/generation.py` and `experiments/judge.py` call
    `load_questions(config)` once and hand the same list to each generation
    task's `resolve_questions` / `generation_cells`.

Arguments:
    None. Import-only module; callers use `load_questions(config, limit=...)`.
"""

from __future__ import annotations

import random

from config import ExperimentConfig
from data.binning import doc_type_bin
from experiments.smoke import load_smoke_questions
from data.loader import load_longdocurl, load_mmlongbench
from schema import Question


def load_questions(config: ExperimentConfig, *, limit: int | None = None) -> list[Question]:
    """Return the questions for this run: frozen smoke corpus or the full set.

    Precedence for the full mmlongbench run: an explicit global cap
    (`limit`/`config.sample`, i.e. `--questions N`) wins and takes the first N;
    otherwise the per-bin document-level subset (`config.per_bin_sample`, default
    100/bin) applies; otherwise the whole corpus. Smoke and LongDocURL never use
    the per-bin subset.
    """

    if config.smoke:
        questions = list(load_smoke_questions(config.paths.data_dir))
    elif config.dataset == "longdocurl":
        questions = list(load_longdocurl(data_dir=config.paths.data_dir))
    else:
        questions = list(load_mmlongbench(data_dir=config.paths.data_dir))

    # A global first-N cap (e.g. from --questions) wins and also reaches
    # experiments that resolve their own corpus, like T4.
    cap = limit if limit is not None else config.sample
    if cap is not None:
        return questions[: max(1, cap)]

    # Otherwise, on the full mmlongbench run, keep each Option-A bin to ~N
    # questions by drawing whole documents (document-level sampling).
    if config.dataset == "mmlongbench" and config.per_bin_sample:
        return sample_questions_per_bin(
            questions,
            config.per_bin_sample,
            bins=config.bins,
            seed=config.sample_seed,
        )
    return questions


def sample_questions_per_bin(
    questions: list[Question],
    target: int,
    *,
    bins: tuple[str, ...],
    seed: int = 0,
) -> list[Question]:
    """Subset to ~`target` questions per bin by drawing whole documents.

    Questions cluster within documents (USER_GUIDE §9), so the subset is drawn
    at the document level: within each bin, documents are shuffled by `seed` and
    added whole until the bin reaches `target` questions. A bin already at or
    below `target` is kept whole. The returned list preserves the original
    dataset order. A different `seed` yields a different (largely disjoint)
    subset for a robustness rerun.
    """

    keep_per_bin: dict[str, set[str]] = {}
    grouped: dict[str, list[Question]] = {bin_name: [] for bin_name in bins}
    for question in questions:
        try:
            bin_name = doc_type_bin(question.doc_type)
        except (KeyError, ValueError):
            continue  # doc_type outside the Option-A bins: not sampled
        if bin_name in grouped:
            grouped[bin_name].append(question)

    for bin_name, bin_questions in grouped.items():
        keep_per_bin[bin_name] = _draw_documents(bin_questions, target, seed)

    kept_ids = {qid for ids in keep_per_bin.values() for qid in ids}
    return [question for question in questions if question.id in kept_ids]


def sample_table4_replication(
    questions: list[Question],
    target: int,
    *,
    bins: tuple[str, ...],
    seed: int = 0,
    reuse_bins: tuple[str, ...] = ("visual_heavy",),
) -> list[Question]:
    """Held-out MMLongBench subset for the Table-4 dataset replication.

    Table 4 asks whether the per-domain recipe replicates on a *different* set of
    documents. For the big bins (text_heavy, in_between) this draws ~`target`
    questions from documents NOT in the primary (T1) per-bin subset, so the
    replication runs on a disjoint document set. The thin `reuse_bins`
    (visual_heavy: only 101 Q / 15 docs) cannot be held out, so their questions
    are reused from the primary subset unchanged; SlideVQA is the planned
    visual-heavy replication (out of scope here). `seed` must match T1's
    `sample_seed` so "primary" means the exact set T1 ran.
    """

    primary = sample_questions_per_bin(questions, target, bins=bins, seed=seed)
    primary_ids = {question.id for question in primary}
    primary_docs = {question.doc_id for question in primary}

    kept_ids: set[str] = set()
    for bin_name in bins:
        bin_questions = [q for q in questions if _bin_or_none(q) == bin_name]
        if bin_name in reuse_bins:
            kept_ids.update(q.id for q in bin_questions if q.id in primary_ids)
        else:
            held_out = [q for q in bin_questions if q.doc_id not in primary_docs]
            subset = sample_questions_per_bin(held_out, target, bins=(bin_name,), seed=seed)
            kept_ids.update(q.id for q in subset)
    return [question for question in questions if question.id in kept_ids]


def _bin_or_none(question: Question) -> str | None:
    """Return a question's Option-A bin, or None if its doc_type is unmapped."""

    try:
        return doc_type_bin(question.doc_type)
    except (KeyError, ValueError):
        return None


def _draw_documents(bin_questions: list[Question], target: int, seed: int) -> set[str]:
    """Return the question ids of whole documents summing to ~`target` questions."""

    if len(bin_questions) <= target:
        return {question.id for question in bin_questions}

    by_doc: dict[str, list[Question]] = {}
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
