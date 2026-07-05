"""Test the role-split experiment pipeline: generation tasks, judge, build.

Purpose:
    Covers the refactor that organizes `experiments/` by generation task (G1..G6)
    rather than by paper table, with three separate roles:

    - `experiments.generation` runs a task's cells once per reasoner spec and
      caches predictions (the `PredictionCache` lets judging reuse them);
    - `experiments.judge` re-scores those predictions with guards that refuse to
      run a reasoner/retriever (`_GuardRetriever`, `_SpecOnlyReasoner`);
    - `experiments.build` routes each task's judged rows into the eight table
      CSVs; a table with no source rows still emits a skeleton.

Test role:
    Fixture PDFs + injected fake reasoner/judge, so the wiring is exercised
    without Marker, Qwen, ColQwen, or a judge API. The heavy generate path is
    validated on hardware via `python -m experiments.generation`.

Arguments:
    None. Run with `python -m pytest tests/test_experiments.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from config import ExperimentConfig, ProjectPaths
from data.loader import load_mmlongbench
from experiments import build, generation, judge
from experiments.generation import GENERATION_TASKS, Retrievers, resolve
from experiments.judge import _GuardRetriever, _SpecOnlyReasoner
from experiments.paths import experiment_paths
from experiments.tables import build_table1_headline
from models.payload import ModelInput
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
    def __init__(self, spec: str = "qwen3vl-2b-local") -> None:
        self.spec = spec
        self.calls = 0

    def answer(self, question: Question, model_input: ModelInput) -> Prediction:
        self.calls += 1
        return Prediction(
            text=f"{question.gold_answer} said",
            model_spec=self.spec,
            input_text_tokens=sum(len(p.text.split()) for p in model_input.text_parts),
            input_visual_tokens=11 * len(model_input.image_parts),
            output_tokens=2,
            latency_s=0.01,
        )


class KeywordJudge(Judge):
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
            "doc_type": "Academic paper",
            "question": "value?",
            "answer": "alpha answer",
            "evidence_pages": "[1]",
            "evidence_sources": "['Pure-text (Plain-text)']",
            "answer_format": "String",
        },
        {
            "doc_id": "smoke-visual.pdf",
            "doc_type": "Brochure",
            "question": "figure?",
            "answer": "beta answer",
            "evidence_pages": "[1]",
            "evidence_sources": "['Chart']",
            "answer_format": "String",
        },
    ]
    pd.DataFrame(rows).to_parquet(root / "data" / "smoke.parquet")
    _write_pdf(root / "documents" / "smoke-text.pdf", ["cover", "alpha answer here"])
    _write_pdf(root / "documents" / "smoke-visual.pdf", ["cover", "beta answer here"])
    paths = ProjectPaths(root=tmp_path, data_dir=data_dir, cache_dir=tmp_path / "results" / "cache")
    return ExperimentConfig(smoke=True, paths=paths, dpi=72)


@pytest.fixture()
def fake_channels(monkeypatch):
    import pipeline.representation as representation

    monkeypatch.setattr(representation, "text_channel", lambda pages: tuple(f"t{p.index}:{p.text}" for p in pages))
    monkeypatch.setattr(representation, "layout_channel", lambda pages: tuple("{}" for _ in pages))


def test_prediction_key_is_judge_independent() -> None:
    q = Question(
        id="q1", doc_id="d.pdf", question="?", gold_answer="x", answer_format="String",
        doc_type="Academic paper", evidence_pages=(1,), evidence_sources=("Text",),
        hop="single", is_unanswerable=False,
    )
    assert make_prediction_key(q, "oracle", "T", "s", 72) == make_prediction_key(q, "oracle", "T", "s", 72)
    assert make_cache_key(q, "oracle", "T", "s", "stub", 72) != make_cache_key(q, "oracle", "T", "s", "gpt", 72)


def test_sufficiency_generate_then_judge_reuses_predictions(tmp_path: Path, fake_channels) -> None:
    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    task = GENERATION_TASKS["G1_sufficiency"]
    paths = experiment_paths(config, task.name)

    # Phase 1: generate with a real (fake) reasoner into G1's prediction cache.
    gen_reasoner = CountingReasoner()
    gen = Orchestrator(
        config, reasoner=gen_reasoner, judge=StubJudge("gen"),
        cache=ResultCache(paths.generate_results), prediction_cache=PredictionCache(paths.predictions),
    )
    guards = Retrievers(_GuardRetriever("t"), _GuardRetriever("v"))
    cells = task.generation_cells(config, questions, retrievers=guards)
    for cell in cells:
        gen.run_cell(cell.question, cell.conditioner, cell.representation)
    assert gen_reasoner.calls == len(questions) * len(config.representations)

    # Phase 2: judge with a spec-only reasoner that must never be called.
    judge_orch = Orchestrator(
        config, reasoner=_SpecOnlyReasoner(gen_reasoner.spec), judge=KeywordJudge(),
        cache=ResultCache(paths.results), prediction_cache=PredictionCache(paths.predictions),
    )
    for cell in cells:
        judge_orch.run_cell(cell.question, cell.conditioner, cell.representation)
    rows = list(judge_orch.cache)
    assert len(rows) == len(cells)
    assert all(r.judge_spec == "keyword-judge" for r in rows)
    assert all(r.correct for r in rows)  # gold appears in the fake prediction

    table1 = build_table1_headline(rows, bins=config.bins, n_bootstrap=0)
    assert list(table1["bin"]) == list(config.bins)


def test_guards_refuse_to_run() -> None:
    q = Question(
        id="q1", doc_id="d.pdf", question="?", gold_answer="x", answer_format="String",
        doc_type="Brochure", evidence_pages=(1,), evidence_sources=("Chart",), hop="single", is_unanswerable=False,
    )
    with pytest.raises(RuntimeError, match="judge phase"):
        _GuardRetriever("colqwen").retrieve(q, 5, 1)
    with pytest.raises(RuntimeError, match="judge phase"):
        _SpecOnlyReasoner("spec").answer(q, ModelInput(parts=()))


def test_resolve_generation_tasks() -> None:
    assert [t.name for t in resolve("G1_sufficiency")] == ["G1_sufficiency"]
    assert [t.name for t in resolve("reasoners")] == [
        "G1_sufficiency", "G2_family", "G3_dataset", "G5_retrieval",
    ]
    assert [t.name for t in resolve("all")] == [
        "G1_sufficiency", "G2_family", "G3_dataset", "G5_retrieval", "G6_classifier",
    ]
    # comma list de-dupes and keeps registry order
    assert [t.name for t in resolve("G5_retrieval,G1_sufficiency,G1_sufficiency")] == [
        "G1_sufficiency", "G5_retrieval",
    ]
    with pytest.raises(ValueError):
        resolve("nope")


def test_run_generate_continue_on_error_records_per_task_status(tmp_path: Path, monkeypatch) -> None:
    config = ExperimentConfig(smoke=True, paths=ProjectPaths(root=tmp_path, cache_dir=tmp_path / "results" / "cache"))
    tasks = [SimpleNamespace(name="ok"), SimpleNamespace(name="bad"), SimpleNamespace(name="after")]

    def fake_generate(config, task, questions):  # noqa: ANN001
        if task.name == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(generation, "resolve", lambda selector: tasks)
    monkeypatch.setattr(generation, "generate", fake_generate)

    statuses = generation.run_generate(config, "fake", [], continue_on_error=True)

    assert [status.status for status in statuses] == ["success", "failed", "success"]
    bad_status = experiment_paths(config, "bad").root / "generate_status.json"
    after_status = experiment_paths(config, "after").root / "generate_status.json"
    assert json.loads(bad_status.read_text())["error"] == "boom"
    assert json.loads(after_status.read_text())["status"] == "success"


def test_build_routes_tasks_into_all_eight_tables(tmp_path: Path, fake_channels, monkeypatch) -> None:
    """Generate G1 + G5 (reasoner cells), judge all, build all eight table CSVs."""

    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)

    fake = CountingReasoner()
    monkeypatch.setattr(generation, "reasoner_for", lambda spec, config=None: fake)

    # G1: oracle ladder via the real generate() path (no side work).
    generation.generate(config, GENERATION_TASKS["G1_sufficiency"], questions)
    # G5: run its matched/cross cells directly with a stub retriever, skipping the
    # real-model run_side (retrieval R/P/F1 needs ColQwen/BM25 weights).
    _generate_cells_with_stub(config, "G5_retrieval", questions, fake)

    judge.run_judge(config, "all", questions, judge_impl=KeywordJudge(), continue_on_error=True)
    written = build.build_tables(config, config.paths.results_dir / "tables" / "smoke",
                                 n_bootstrap=0, markdown_path=tmp_path / "all_tables.md")

    assert {f"table{i}" for i in range(1, 9)} <= set(written)
    assert written["markdown"].exists()
    for i in range(1, 9):
        assert written[f"table{i}"].exists(), f"table{i}"
    # Tables sourced from the generated tasks (G1 -> table1, G5 -> table6) have rows.
    assert len(pd.read_csv(written["table1"])) > 0


def _generate_cells_with_stub(config, task_name, questions, reasoner) -> None:
    """Run a task's cells with a deterministic first-page retriever (no run_side)."""

    from covariates.retriever import StubRetriever

    task = GENERATION_TASKS[task_name]
    paths = experiment_paths(config, task.name)
    orch = Orchestrator(
        config, reasoner=reasoner, judge=StubJudge("gen"),
        cache=ResultCache(paths.generate_results), prediction_cache=PredictionCache(paths.predictions),
    )
    retrievers = Retrievers(StubRetriever(), StubRetriever())
    for cell in task.generation_cells(config, questions, retrievers=retrievers):
        orch.run_cell(cell.question, cell.conditioner, cell.representation)
