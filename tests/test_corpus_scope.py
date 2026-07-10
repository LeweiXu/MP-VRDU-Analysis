"""Corpus resolution: sampling is document-coherent, the three sampling modes
resolve the expected sets, and the answerable pool is bound by the task (G1/G2
answerable, G3 unanswerable) so a spec cannot cross-contaminate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import require


def _q(qid: str, doc: str, bin_label: str, unanswerable: bool = False, doc_type: str = "report"):
    return SimpleNamespace(id=qid, doc_id=doc, bin_label=bin_label,
                           is_unanswerable=unanswerable, doc_type=doc_type)


def _corpus():
    # 3 docs of doc_type "report" (2 questions each) and 1 doc of doc_type "manual".
    return [
        _q("q1", "docA", "text-dominant", doc_type="report"), _q("q2", "docA", "text-dominant", doc_type="report"),
        _q("q3", "docB", "text-dominant", doc_type="report"), _q("q4", "docB", "text-dominant", doc_type="report"),
        _q("q5", "docC", "text-dominant", doc_type="report"), _q("q6", "docC", "text-dominant", doc_type="report"),
        _q("q7", "docD", "visual-dominant", doc_type="manual"), _q("q8", "docD", "visual-dominant", doc_type="manual"),
    ]


def test_sampling_is_document_coherent() -> None:
    sample_per_bin = require("experiments.corpus.resolve", "sample_per_bin")
    kept = sample_per_bin(_corpus(), per_bin=1, seed=0)
    kept_docs = {q.doc_id for q in kept}
    # Whichever doc is drawn, ALL of its questions come along (doc-level draw).
    for doc in kept_docs:
        picked = {q.id for q in kept if q.doc_id == doc}
        allq = {q.id for q in _corpus() if q.doc_id == doc}
        assert picked == allq, "must sample whole documents, not questions"


def test_sampling_is_deterministic_under_seed() -> None:
    sample_per_bin = require("experiments.corpus.resolve", "sample_per_bin")
    a = {q.id for q in sample_per_bin(_corpus(), per_bin=1, seed=7)}
    b = {q.id for q in sample_per_bin(_corpus(), per_bin=1, seed=7)}
    assert a == b


def test_limit_mode_caps_question_count() -> None:
    resolve_corpus = require("experiments.corpus.resolve", "resolve_corpus")
    got = resolve_corpus({"sampling": {"limit": 3}}, _corpus())
    assert len(list(got)) == 3


def test_full_mode_returns_everything() -> None:
    resolve_corpus = require("experiments.corpus.resolve", "resolve_corpus")
    got = list(resolve_corpus({"sampling": "full"}, _corpus()))
    assert len(got) == len(_corpus())


def test_per_doc_type_caps_to_exact_count_per_label() -> None:
    sample_per_doc_type = require("experiments.corpus.resolve", "sample_per_doc_type")
    kept = sample_per_doc_type(_corpus(), per_doc_type=1, seed=0)
    # Exactly one question per doc_type label (report, manual) -> 2 total, even though
    # each document holds 2 questions (the last drawn document is sliced to hit N).
    assert {q.doc_type for q in kept} == {"report", "manual"}
    for doc_type in ("report", "manual"):
        n = len([q for q in kept if q.doc_type == doc_type])
        assert n == 1, f"per_doc_type=1 keeps exactly one question for {doc_type}, got {n}"
    assert len(kept) == 2


def test_per_doc_type_cap_spans_documents_and_keeps_small_labels_whole() -> None:
    sample_per_doc_type = require("experiments.corpus.resolve", "sample_per_doc_type")
    # report has 3 docs x 2 q = 6 q -> capped to exactly 3; manual has only 2 q -> kept whole.
    kept = sample_per_doc_type(_corpus(), per_doc_type=3, seed=0)
    assert len([q for q in kept if q.doc_type == "report"]) == 3
    assert len([q for q in kept if q.doc_type == "manual"]) == 2


def test_per_doc_type_deterministic_under_seed() -> None:
    sample_per_doc_type = require("experiments.corpus.resolve", "sample_per_doc_type")
    a = {q.id for q in sample_per_doc_type(_corpus(), per_doc_type=1, seed=3)}
    b = {q.id for q in sample_per_doc_type(_corpus(), per_doc_type=1, seed=3)}
    assert a == b


def test_per_doc_type_mode_resolves_via_corpus() -> None:
    resolve_corpus = require("experiments.corpus.resolve", "resolve_corpus")
    got = list(resolve_corpus({"sampling": {"per_doc_type": 1, "seed": 0}}, _corpus()))
    assert len(got) == 2


def test_answerable_pool_is_bound_by_task() -> None:
    pool_for_task = require("experiments.corpus.resolve", "pool_for_task")
    assert pool_for_task("G1_oracle_ladder") == "answerable"
    assert pool_for_task("G2_retrieval") == "answerable"
    assert pool_for_task("G3_hallucination") == "unanswerable"
