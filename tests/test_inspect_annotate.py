"""Tests for the inference-inspection and document-annotation tooling.

Purpose:
    Covers `experiments/inspect.py` (joining cached predictions + judged rows +
    questions, filtering, and the all-fields markdown), the shared
    `data.render.classify_scanned` heuristic, and `scripts/annotate_docs.py`
    seeding + scoring. No GPU, no API, no real corpus: everything runs on a tiny
    fixture PDF + parquet.

Arguments:
    None. Run with `python -m pytest tests/test_inspect_annotate.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import ExperimentConfig, ProjectPaths
from data.loader import load_mmlongbench
from data.render import classify_scanned
from experiments.inspect import select_items, write_item
from experiments.paths import experiment_paths
from pipeline.orchestrator import CachedPrediction, PredictionCache, ResultCache, ResultRow
from scripts.annotate_docs import invalid_values, row_is_annotated, score_sheet

TASK = "G1_sufficiency"


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


def _fixture_config(tmp_path: Path) -> ExperimentConfig:
    data_dir = tmp_path / ".data"
    root = data_dir / "mmlongbench"
    (root / "data").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)
    rows = [
        {
            "doc_id": "alpha.pdf",
            "doc_type": "Academic paper",
            "question": "what is the answer?",
            "answer": "gold answer",
            "evidence_pages": "[1]",
            "evidence_sources": "['Pure-text (Plain-text)']",
            "answer_format": "String",
        }
    ]
    pd.DataFrame(rows).to_parquet(root / "data" / "train-00000-of-00001.parquet")
    _write_pdf(root / "documents" / "alpha.pdf", ["cover page", "the gold answer lives here"])
    paths = ProjectPaths(root=tmp_path, data_dir=data_dir, cache_dir=tmp_path / "results" / "cache")
    return ExperimentConfig(smoke=True, paths=paths, dpi=72)


def _prediction(representation: str, text: str) -> CachedPrediction:
    return CachedPrediction(
        prediction_key=f"pk-{representation}",
        question_id="mmlongbench:000000",
        doc_id="alpha.pdf",
        condition="oracle",
        representation=representation,
        model_spec="qwen3vl-8b-local",
        provenance="oracle",
        page_indices=(0,),
        note="",
        text=text,
        input_text_tokens=10,
        input_visual_tokens=0,
        output_tokens=2,
        latency_s=1.0,
    )


def _row(representation: str, *, correct: bool) -> ResultRow:
    return ResultRow(
        cache_key=f"ck-{representation}",
        question_id="mmlongbench:000000",
        doc_id="alpha.pdf",
        doc_type="Academic paper",
        hop="single",
        is_unanswerable=False,
        evidence_sources=("Pure-text (Plain-text)",),
        condition="oracle",
        provenance="oracle",
        page_indices=(0,),
        representation=representation,
        model_spec="qwen3vl-8b-local",
        judge_spec="gemini",
        answer="gold answer" if correct else "wrong",
        input_text_tokens=10,
        input_visual_tokens=0,
        output_tokens=2,
        latency_s=1.0,
        score=1.0 if correct else 0.0,
        correct=correct,
        abstained=False,
        metadata={"note": "", "source_dataset": "mmlongbench"},
    )


def _seed_cache(config: ExperimentConfig) -> None:
    paths = experiment_paths(config, TASK)
    predictions = PredictionCache(paths.predictions)
    predictions.put(_prediction("T", "gold answer"))
    predictions.put(_prediction("V", "wrong"))
    results = ResultCache(paths.results)
    results.put(_row("T", correct=True))
    results.put(_row("V", correct=False))


def test_select_items_joins_and_filters(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    _seed_cache(config)

    items = select_items(config, TASK)
    assert [(i.prediction.representation, i.judged) for i in items] == [("T", True), ("V", True)]
    # question and judged row are attached
    assert items[0].question is not None and items[0].question.gold_answer == "gold answer"
    assert items[0].row is not None and items[0].row.correct

    assert [i.prediction.representation for i in select_items(config, TASK, representation="T")] == ["T"]
    assert [i.prediction.representation for i in select_items(config, TASK, incorrect_only=True)] == ["V"]
    assert len(select_items(config, TASK, limit=1)) == 1


def test_write_item_dumps_all_generate_and_judge_fields(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    _seed_cache(config)
    item = select_items(config, TASK, representation="T")[0]

    dest = write_item(item, tmp_path / "inspect", config)
    info = (dest / "info.md").read_text()

    # every generate + judge field name is present
    for field in ("prediction_key", "page_indices", "input_visual_tokens", "latency_s"):
        assert f"`{field}`" in info
    for field in ("cache_key", "judge_spec", "score", "correct", "abstained", "metadata"):
        assert f"`{field}`" in info
    # question context + the model's answer
    assert "gold answer" in info
    assert "the judge's free-text rationale is not persisted" in info
    # the fed page was rendered and copied in
    assert list(dest.glob("*.png"))


def test_classify_scanned_labels_digital_and_scanned(tmp_path: Path) -> None:
    digital = tmp_path / "digital.pdf"
    _write_pdf(digital, ["plenty of embedded text " * 20])
    scanned = tmp_path / "scanned.pdf"
    _write_pdf(scanned, [""])  # blank page => no text layer

    assert classify_scanned(digital).label == "digital"
    assert classify_scanned(scanned).label == "scanned"


def test_row_is_annotated_and_invalid_values() -> None:
    full = {"bin_label": "text_heavy", "scan_label": "digital", "dominant_visual": "tables", "multi_column": "single"}
    assert row_is_annotated(full)
    assert not row_is_annotated({**full, "multi_column": ""})

    bad = [{"doc_id": "d.pdf", "bin_label": "text_heavy", "scan_label": "nope"}]
    problems = invalid_values(bad)
    assert any("scan_label" in p for p in problems)


def test_score_sheet_reports_bin_and_scan_agreement() -> None:
    rows = [
        {"doc_id": "a", "doc_type": "Academic paper", "auto_bin": "text_heavy", "bin_label": "text_heavy",
         "auto_scan": "digital", "scan_label": "digital", "dominant_visual": "tables;charts", "multi_column": "single"},
        {"doc_id": "b", "doc_type": "Brochure", "auto_bin": "visual_heavy", "bin_label": "in_between",
         "auto_scan": "scanned", "scan_label": "scanned", "dominant_visual": "photos", "multi_column": "multi"},
        {"doc_id": "c", "doc_type": "Brochure", "auto_bin": "visual_heavy", "bin_label": "",
         "auto_scan": "digital", "scan_label": "", "dominant_visual": "", "multi_column": ""},
    ]
    summary = score_sheet(rows)

    assert summary["total"] == 3
    assert summary["bin_labelled"] == 2
    assert summary["bin_agree"] == 1  # only doc "a" matches its auto_bin
    assert summary["mismatches"][0]["doc_id"] == "b"
    assert summary["auto_scanned"] == 1
    assert summary["scan_agree"] == 2
    # multi-value dominant_visual: each token counted separately
    assert summary["dominant_visual"] == {"tables": 1, "charts": 1, "photos": 1}
    assert summary["multi_column"] == {"single": 1, "multi": 1}


def test_dominant_visual_accepts_multiple_values() -> None:
    from scripts.annotate_docs import invalid_values, split_multi

    assert split_multi("tables;charts") == ["tables", "charts"]
    # both tokens valid -> no complaints
    assert invalid_values([{"doc_id": "d.pdf", "dominant_visual": "tables;charts"}]) == []
    # one bad token is flagged
    assert any("dominant_visual" in p for p in invalid_values([{"doc_id": "d.pdf", "dominant_visual": "tables;bogus"}]))
