"""Test MMLongBench loading and PDF rendering on a tiny fixture.

Purpose:
    Verifies row normalisation, evidence-page parsing, unanswerable detection,
    PDF resolution, page rendering, and embedded text extraction.

Test role:
    Protects the data layer consumed by every later pipeline stage.

Arguments:
    None. Run with `python -m pytest tests/test_data.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.loader import load_mmlongbench, resolve_pdf
from data.render import render_question_pages, validate_gold_pages
from schema import PageSet


def write_pdf(path: Path, pages: list[str]) -> None:
    """Write a tiny text PDF for data/render tests."""

    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def write_fixture(tmp_path: Path) -> Path:
    """Create a staged MMLongBench-like fixture and return `.data`."""

    data_dir = tmp_path / ".data"
    root = data_dir / "mmlongbench"
    (root / "data").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)
    rows = [
        {
            "doc_id": "alpha.pdf",
            "doc_type": "Academic paper",
            "question": "Which pages contain the evidence?",
            "answer": "pages one and two",
            "evidence_pages": "[1, 2]",
            "evidence_sources": "['Pure-text (Plain-text)', 'Table']",
            "answer_format": "String",
        },
        {
            "doc_id": "beta.pdf",
            "doc_type": "Brochure",
            "question": "What is the missing value?",
            "answer": "Not answerable",
            "evidence_pages": "[]",
            "evidence_sources": "[]",
            "answer_format": "None",
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "data" / "train-00000-of-00001.parquet")
    write_pdf(root / "documents" / "alpha.pdf", ["alpha page one", "alpha page two"])
    write_pdf(root / "documents" / "beta.pdf", ["beta page one"])
    return data_dir


def test_load_mmlongbench_normalises_questions(tmp_path: Path) -> None:
    data_dir = write_fixture(tmp_path)

    questions = load_mmlongbench(data_dir=data_dir)

    assert len(questions) == 2
    first = questions[0]
    assert first.id == "mmlongbench:000000"
    assert first.doc_id == "alpha.pdf"
    assert first.evidence_pages == (0, 1)
    assert first.evidence_sources == ("Pure-text (Plain-text)", "Table")
    assert first.hop == "multi"
    assert not first.is_unanswerable
    assert first.raw_fields["evidence_pages"] == "[1, 2]"
    assert PageSet.oracle(first).page_indices == (0, 1)

    second = questions[1]
    assert second.evidence_pages == ()
    assert second.hop == "none"
    assert second.is_unanswerable


def test_resolve_and_render_question_pages(tmp_path: Path) -> None:
    data_dir = write_fixture(tmp_path)
    question = load_mmlongbench(data_dir=data_dir, sample=1)[0]

    assert resolve_pdf(question.doc_id, data_dir=data_dir).is_file()
    assert validate_gold_pages(question, data_dir=data_dir)

    pages = render_question_pages(
        question,
        data_dir=data_dir,
        cache_dir=tmp_path / "results" / "cache",
        dpi=72,
    )

    assert [page.index for page in pages] == [0, 1]
    assert all(page.image_path and page.image_path.is_file() for page in pages)
    assert "alpha page one" in pages[0].text
    assert "alpha page two" in pages[1].text
