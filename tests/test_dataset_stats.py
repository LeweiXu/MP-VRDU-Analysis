"""Tests for dataset-scoped descriptive statistics."""

from __future__ import annotations

import csv

import ops.scripts.dataset_stats as stats


def _records():
    return [
        {
            "doc_id": "a.pdf",
            "doc_type": "Report",
            "question": "q1",
            "answer": "a1",
            "evidence_pages": "[1]",
            "evidence_sources": "['Figure']",
            "answer_format": "Str",
        },
        {
            "doc_id": "a.pdf",
            "doc_type": "Report",
            "question": "q2",
            "answer": "a2",
            "evidence_pages": "[1, 2]",
            "evidence_sources": "['Figure', 'Table']",
            "answer_format": "Str",
        },
        {
            "doc_id": "b.pdf",
            "doc_type": "Guide",
            "question": "q3",
            "answer": "a3",
            "evidence_pages": "[]",
            "evidence_sources": "['Table']",
            "answer_format": "Str",
        },
    ]


def test_dataset_name_accepts_key_and_display_name() -> None:
    assert stats.resolve_dataset_key("mmlongbench") == "mmlongbench"
    assert stats.resolve_dataset_key("MMLongBench-Doc") == "mmlongbench"
    assert stats.resolve_dataset_key("mmlongbenchdoc") == "mmlongbench"


def test_mmlongbench_profile_adds_requested_breakdowns(tmp_path, monkeypatch) -> None:
    annotations = tmp_path / "doc_labels.csv"
    with annotations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "scan_label"])
        writer.writeheader()
        writer.writerows([
            {"doc_id": "a.pdf", "scan_label": "digital"},
            {"doc_id": "b.pdf", "scan_label": "scanned"},
        ])
    monkeypatch.setattr(stats, "ANNOTATION_SHEET", annotations)
    records = _records()
    keys = list(records[0])
    monkeypatch.setattr(
        stats,
        "load_local_mmlongbench",
        lambda _limit: (records, "fixture", keys, []),
    )

    markdown, summary, details = stats.profile(
        "mmlongbench",
        stats.REGISTRY["mmlongbench"],
        100,
    )

    assert "| Class | Questions | Documents | Digital docs | Scanned docs | Unknown scan docs |" in markdown
    assert "| Report | 2 | 1 | 1 | 0 | 0 |" in markdown
    assert "### Questions per document" in markdown
    assert "#### Report" in markdown
    assert "| a.pdf | Report | 2 |" in markdown
    assert "| Class | Questions | Documents | Report | Guide |" in markdown
    assert "| Figure | 2 | 1 | 2 | 0 |" in markdown
    assert summary["records_scanned"] == 3
    assert any(
        row["section"] == "document_questions"
        and row["doc_id"] == "a.pdf"
        and row["n_questions"] == 2
        for row in details
    )
    assert any(
        row["field"] == "evidence_sources"
        and row["class"] == "Figure"
        and row["Report"] == 2
        and row["Guide"] == 0
        for row in details
    )
