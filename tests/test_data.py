"""Stage 1 — data layer: loader, evidence parsing, rendering, dev slice."""

from mpvrdu.data.dataset import (QuestionType, parse_evidence_pages,
                                 parse_str_list)
from mpvrdu.data.render import render_page
from mpvrdu.data.slice import build_dev_slice


def test_loader_counts(synthetic_ds):
    assert len(synthetic_ds) == 7              # 3 single + 2 cross + 2 unanswerable
    assert len(synthetic_ds.documents) == 3


def test_evidence_pages_parsed_to_ints():
    assert parse_evidence_pages("[3, 5]") == [3, 5]
    assert parse_evidence_pages("[]") == []
    assert parse_evidence_pages(None) == []
    assert parse_evidence_pages([1, 2]) == [1, 2]
    pages = parse_evidence_pages("[10]")
    assert pages == [10] and all(isinstance(p, int) for p in pages)


def test_str_list_parsed():
    assert parse_str_list("['Chart', 'Table']") == ["Chart", "Table"]
    assert parse_str_list("[]") == []


def test_question_type_classification(synthetic_ds):
    counts = synthetic_ds.type_counts()
    assert counts[QuestionType.SINGLE.value] == 3
    assert counts[QuestionType.CROSS.value] == 2
    assert counts[QuestionType.UNANSWERABLE.value] == 2


def test_evidence_zero_based(synthetic_ds):
    q = next(q for q in synthetic_ds.questions if q.qid == "s1")
    assert q.evidence_pages == [2]            # 1-based
    assert q.evidence_pages_zero_based == [1]  # 0-based for renderer


def test_referenced_pages_render_nonempty(synthetic_ds, tmp_path):
    for q in synthetic_ds.questions:
        if q.is_unanswerable:
            continue
        doc = synthetic_ds.get_document(q.doc_id)
        for p0 in q.evidence_pages_zero_based:
            rp = render_page(doc.pdf_path, p0, dpi=72,
                             cache_dir=tmp_path / "renders", doc_id=doc.doc_id)
            assert not rp.is_empty
            assert rp.width > 0 and rp.height > 0
            assert rp.path.exists() and rp.path.stat().st_size > 0


def test_render_cache_reused(synthetic_ds, tmp_path):
    doc = synthetic_ds.get_document("alpha.pdf")
    a = render_page(doc.pdf_path, 0, dpi=72, cache_dir=tmp_path / "c", doc_id="alpha.pdf")
    mtime = a.path.stat().st_mtime_ns
    b = render_page(doc.pdf_path, 0, dpi=72, cache_dir=tmp_path / "c", doc_id="alpha.pdf")
    assert b.path == a.path
    assert b.path.stat().st_mtime_ns == mtime  # not re-rendered


def test_dev_slice_covers_all_three_types(synthetic_ds, tmp_path):
    slice_ds = build_dev_slice(synthetic_ds, out_dir=tmp_path / "dev_slice",
                               max_docs=5, max_questions=30)
    counts = slice_ds.type_counts()
    assert counts[QuestionType.SINGLE.value] >= 1
    assert counts[QuestionType.CROSS.value] >= 1
    assert counts[QuestionType.UNANSWERABLE.value] >= 1
    # slice PDFs were copied and load
    for doc_id in slice_ds.doc_ids():
        assert slice_ds.get_document(doc_id).pdf_path.exists()
