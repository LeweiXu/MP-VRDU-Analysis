"""Test the two-phase full-smoke mechanism: predict once, judge later, 8 tables.

Purpose:
    Covers the machinery that lets the Section-1 "full run, tiny data" sweep
    split across a GPU phase and an internet phase:

    - the `PredictionCache` stores a reasoner output keyed without the judge, so
      a second (judge-only) pass reuses it and never calls the reasoner again;
    - `make_prediction_key` is judge-independent, so two judges share a key;
    - `_CachedOnlyRetriever` fails loudly if the judge phase ever needs to
      retrieve (i.e. the generate phase missed a cell);
    - all eight paper tables build from the judged rows.

Test role:
    Uses fixture PDFs plus injected text/layout channels and fake reasoner/judge
    so the mechanism is exercised without Marker, Qwen, ColQwen, or OpenAI. The
    real end-to-end `run_generate`/`run_judge` are validated on hardware via
    `cli.run_smoke` / the Kaya smoke scripts.

Arguments:
    None. Run with `python -m pytest tests/test_smoke_run.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import ExperimentConfig, ProjectPaths
from data.loader import load_mmlongbench
from experiments.smoke_run import _CachedOnlyRetriever
from experiments.tables import build_all_tables
from models.payload import ModelInput
from pipeline.conditioner import OracleConditioner
from pipeline.judge import Judge, StubJudge
from pipeline.orchestrator import (
    Orchestrator,
    PredictionCache,
    ResultCache,
    make_cache_key,
    make_prediction_key,
)
from pipeline.reasoner import Reasoner
from schema import Prediction, Question, Score


class CountingReasoner(Reasoner):
    """Reasoner fake that counts how many times it is actually called."""

    def __init__(self, spec: str = "smoke-fake-2b") -> None:
        self.spec = spec
        self.calls = 0

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        self.calls += 1
        return Prediction(
            text=f"{question.gold_answer} said",
            model_spec=self.spec,
            input_text_tokens=sum(len(part.text.split()) for part in model_input.text_parts),
            input_visual_tokens=13 * len(model_input.image_parts),
            output_tokens=2,
            latency_s=0.01,
        )


class KeywordJudge(Judge):
    """Judge fake: correct when the gold answer appears in the model answer."""

    spec = "keyword-judge"

    def score(self, question: Question, prediction: Prediction) -> Score:
        correct = question.gold_answer.casefold() in prediction.text.casefold()
        return Score(value=1.0 if correct else 0.0, correct=correct, judge_spec=self.spec)


def _write_pdf(path: Path, pages: list[str]) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _make_config(tmp_path: Path) -> ExperimentConfig:
    data_dir = tmp_path / ".data"
    root = data_dir / "mmlongbench"
    (root / "data").mkdir(parents=True)
    (root / "documents").mkdir(parents=True)
    rows = [
        {
            "doc_id": "smoke-text.pdf",
            "doc_type": "Academic paper",  # text_heavy
            "question": "What is on the evidence page?",
            "answer": "alpha answer",
            "evidence_pages": "[1]",
            "evidence_sources": "['Pure-text (Plain-text)']",
            "answer_format": "String",
        },
        {
            "doc_id": "smoke-visual.pdf",
            "doc_type": "Brochure",  # visual_heavy
            "question": "What does the figure show?",
            "answer": "beta answer",
            "evidence_pages": "[2]",
            "evidence_sources": "['Chart']",
            "answer_format": "String",
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "data" / "smoke.parquet")
    _write_pdf(root / "documents" / "smoke-text.pdf", ["cover", "alpha answer here"])
    _write_pdf(root / "documents" / "smoke-visual.pdf", ["cover", "mid", "beta answer here"])
    paths = ProjectPaths(root=tmp_path, data_dir=data_dir, cache_dir=tmp_path / "results" / "cache")
    return ExperimentConfig(smoke=True, paths=paths, dpi=72)


@pytest.fixture()
def fake_channels(monkeypatch):
    import pipeline.representation as representation

    monkeypatch.setattr(
        representation, "text_channel", lambda pages: tuple(f"text {p.index}: {p.text}" for p in pages)
    )
    monkeypatch.setattr(
        representation, "layout_channel", lambda pages: tuple("{}" for _ in pages)
    )


def _run_ladder(orchestrator: Orchestrator, config: ExperimentConfig, questions) -> list:
    oracle = OracleConditioner()
    rows = []
    for question in questions:
        for rung in config.representations:
            rows.append(orchestrator.run_cell(question, oracle, rung))
    return rows


def test_prediction_key_is_judge_independent() -> None:
    """The prediction key must exclude the judge spec so judges can share it."""

    question = Question(
        id="q1",
        doc_id="d1.pdf",
        question="?",
        gold_answer="x",
        answer_format="String",
        doc_type="Academic paper",
        evidence_pages=(1,),
        evidence_sources=("Pure-text (Plain-text)",),
        hop="single",
        is_unanswerable=False,
    )
    pkey = make_prediction_key(question, "oracle", "T", "spec-a", 72)
    assert pkey == make_prediction_key(question, "oracle", "T", "spec-a", 72)
    # Two full cache keys with different judges differ, but the prediction key does not.
    key_stub = make_cache_key(question, "oracle", "T", "spec-a", "stub", 72)
    key_gpt = make_cache_key(question, "oracle", "T", "spec-a", "gpt", 72)
    assert key_stub != key_gpt


def test_two_phase_predicts_once_and_judges_later(tmp_path: Path, fake_channels) -> None:
    """Generate populates the prediction cache; judge reuses it without the model."""

    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    prediction_path = config.paths.cache_dir / "smoke" / "predictions.jsonl"

    # Phase 1: generate with a throwaway judge; the reasoner runs for every cell.
    gen_reasoner = CountingReasoner()
    generate = Orchestrator(
        config,
        reasoner=gen_reasoner,
        judge=StubJudge("throwaway"),
        cache=ResultCache(config.paths.cache_dir / "smoke" / "generate.jsonl"),
        prediction_cache=PredictionCache(prediction_path),
    )
    gen_rows = _run_ladder(generate, config, questions)
    n_cells = len(questions) * len(config.representations)
    assert gen_reasoner.calls == n_cells
    assert len(generate.prediction_cache) == n_cells

    # Phase 2: a fresh reasoner that must never be called, plus the real judge.
    judge_reasoner = CountingReasoner()
    judge = Orchestrator(
        config,
        reasoner=judge_reasoner,
        judge=KeywordJudge(),
        cache=ResultCache(config.paths.cache_dir / "smoke" / "results.jsonl"),
        prediction_cache=PredictionCache(prediction_path),
    )
    judged_rows = _run_ladder(judge, config, questions)

    assert judge_reasoner.calls == 0  # every cell was a prediction-cache hit
    assert len(judged_rows) == n_cells
    assert all(row.judge_spec == "keyword-judge" for row in judged_rows)
    # The judged answer text came from the cached generate-phase prediction.
    assert all(row.answer.endswith("said") for row in judged_rows)
    # Gold answers appear in the fake predictions, so the keyword judge marks correct.
    assert all(row.correct for row in judged_rows)
    # Generate and judge rows share predictions but have different full cache keys.
    assert {r.cache_key for r in gen_rows}.isdisjoint({r.cache_key for r in judged_rows})


def test_all_eight_tables_build_from_judged_rows(tmp_path: Path, fake_channels) -> None:
    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    orchestrator = Orchestrator(
        config,
        reasoner=CountingReasoner(),
        judge=KeywordJudge(),
        cache=ResultCache(config.paths.cache_dir / "smoke" / "results.jsonl"),
        prediction_cache=PredictionCache(config.paths.cache_dir / "smoke" / "predictions.jsonl"),
    )
    rows = _run_ladder(orchestrator, config, questions)

    tables = build_all_tables(rows, bins=config.bins, margin_points=config.sufficiency_margin, n_bootstrap=50)
    assert set(tables) == {f"table{i}" for i in range(1, 9)}
    assert list(tables["table1"]["bin"]) == list(config.bins)
    # Table 1 has one frontier column and a latency-at-frontier column.
    assert "frontier" in tables["table1"].columns
    assert "latency_at_frontier_s" in tables["table1"].columns
    # Every table is non-empty (right-shaped even on tiny smoke data).
    for name, table in tables.items():
        assert len(table) > 0, name


def test_cached_only_retriever_refuses_to_run() -> None:
    """The judge-phase guard retriever must raise, not silently retrieve."""

    guard = _CachedOnlyRetriever("colqwen_vision")
    question = Question(
        id="q1",
        doc_id="d1.pdf",
        question="?",
        gold_answer="x",
        answer_format="String",
        doc_type="Brochure",
        evidence_pages=(1,),
        evidence_sources=("Chart",),
        hop="single",
        is_unanswerable=False,
    )
    with pytest.raises(RuntimeError, match="judge phase"):
        guard.retrieve(question, 5, 1)
