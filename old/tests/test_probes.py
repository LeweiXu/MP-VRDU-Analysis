"""Test Stage-1 feasibility probes against a tiny local fixture.

Purpose:
    Verifies that loader, scanned-page, bbox, unanswerable, doc-type, model, and
    retrieval probes produce stable verdicts without requiring network or GPU.

Test role:
    Protects the feasibility layer that records early assumptions in
    `docs/AGENT_GUIDE.md`.

Arguments:
    None. Run with `python -m pytest tests/test_probes.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_probe import (
    NEEDS_HARDWARE,
    PARTIAL,
    PASS,
    ProbeConfig,
    parse_int_list,
    parse_list,
    probe_doc_type_distribution,
    probe_in_page_boxes,
    probe_loader_smoke,
    probe_model_family,
    probe_scanned_vs_born_digital,
    probe_unanswerable_abstention,
    probe_vision_retrieval,
    run_selected,
)


def _write_pdf(path: Path, pages: list[str]) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _fixture_config(tmp_path: Path) -> ProbeConfig:
    data_root = tmp_path / ".data" / "mmlongbench"
    (data_root / "data").mkdir(parents=True)
    (data_root / "documents").mkdir(parents=True)

    rows = [
        {
            "doc_id": "alpha.pdf",
            "doc_type": "Academic paper",
            "question": "What value is stated?",
            "answer": "42",
            "evidence_pages": "[1]",
            "evidence_sources": "['Pure-text (Plain-text)']",
            "answer_format": "Int",
        },
        {
            "doc_id": "beta.pdf",
            "doc_type": "Brochure",
            "question": "What is shown?",
            "answer": "Not answerable",
            "evidence_pages": "[]",
            "evidence_sources": "[]",
            "answer_format": "None",
        },
        {
            "doc_id": "gamma.pdf",
            "doc_type": "Financial report",
            "question": "Which amount changed?",
            "answer": "7",
            "evidence_pages": "[1, 2]",
            "evidence_sources": "['Table', 'Chart', 'Pure-text (Plain-text)']",
            "answer_format": "Float",
        },
    ]
    pd.DataFrame(rows).to_parquet(data_root / "data" / "train-00000-of-00001.parquet")

    _write_pdf(data_root / "documents" / "alpha.pdf", ["embedded text page with enough characters"])
    _write_pdf(data_root / "documents" / "beta.pdf", [""])
    _write_pdf(data_root / "documents" / "gamma.pdf", ["financial text with enough characters", "more text"])

    return ProbeConfig(
        root=tmp_path,
        data_dir=tmp_path / ".data",
        sample=3,
        pdf_sample=3,
        max_pages_per_pdf=1,
    )


def test_parse_list_helpers() -> None:
    assert parse_list("['Chart', 'Table']") == ["Chart", "Table"]
    assert parse_int_list("[1, 2, '3']") == [1, 2, 3]
    assert parse_list("[]") == []


def test_loader_probe_parses_fields_and_resolves_pdfs(tmp_path: Path) -> None:
    verdict = probe_loader_smoke(_fixture_config(tmp_path))

    assert verdict.status == PASS
    assert verdict.details["records_total"] == 3
    assert verdict.details["pdfs_resolved"] == 3
    assert verdict.details["unanswerable_count"] == 1


def test_scanned_probe_detects_mixed_text_layers(tmp_path: Path) -> None:
    verdict = probe_scanned_vs_born_digital(_fixture_config(tmp_path))

    assert verdict.status == PASS
    assert verdict.details["scanned_like"] == 1
    assert verdict.details["born_digital_like"] == 2


def test_box_probe_confirms_page_level_only(tmp_path: Path) -> None:
    verdict = probe_in_page_boxes(_fixture_config(tmp_path))

    assert verdict.status == PASS
    assert verdict.details["candidate_box_fields"] == {}
    assert "page-level" in verdict.details["crop_decision"]


def test_unanswerable_probe_counts_signal_and_definition(tmp_path: Path) -> None:
    verdict = probe_unanswerable_abstention(_fixture_config(tmp_path))

    assert verdict.status == PASS
    assert verdict.details["unanswerable_count"] == 1
    proposal = verdict.details["abstention_definition_proposal"]
    assert "not answerable" in proposal["abstains_if_prediction_contains"]


def test_doc_type_probe_reports_counts_and_mapping(tmp_path: Path) -> None:
    verdict = probe_doc_type_distribution(_fixture_config(tmp_path))

    assert verdict.status == PASS
    assert len(verdict.details["question_counts"]) == 3
    mapping = verdict.details["spectrum_mapping_proposal"]
    assert mapping["Academic paper"] == "text-heavy"
    assert mapping["Financial report"] == "in-between"


def test_hardware_probes_are_safe_without_gpu_or_network(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)

    model = probe_model_family(config)
    retrieval = probe_vision_retrieval(config)

    assert model.status == NEEDS_HARDWARE
    assert "repo_checks" in model.details
    assert retrieval.status in {PARTIAL, PASS}
    assert retrieval.details["bm25"]["status"] == "pass"


def test_run_selected_local_group_returns_verdict_objects(tmp_path: Path) -> None:
    verdicts = run_selected("local", _fixture_config(tmp_path))

    assert [verdict.name for verdict in verdicts] == [
        "loader",
        "scanned",
        "boxes",
        "unanswerable",
        "doc-type",
    ]
    assert all({"name", "status", "summary", "details"} <= set(verdict.to_dict()) for verdict in verdicts)
