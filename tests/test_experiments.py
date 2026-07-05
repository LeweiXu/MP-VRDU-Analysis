"""Test the per-experiment pipeline: two-phase driver, registry, and 8 tables.

Purpose:
    Covers the refactor that makes each paper table its own reusable experiment
    run in two phases (generate on GPU, judge/build anywhere):

    - the `PredictionCache` lets the judge phase reuse a prediction without
      re-running the reasoner (and `make_prediction_key` is judge-independent);
    - `experiments.driver` generate → judge → build produces each table CSV from
      per-experiment caches, with `depends_on` rows pulled in;
    - the phase-2 guards (`_GuardRetriever`, `_SpecOnlyReasoner`) refuse to run;
    - the registry resolves names/groups; every experiment builds its table.

Test role:
    Fixture PDFs + injected fake reasoner/judge, so the wiring is exercised
    without Marker, Qwen, ColQwen, or a judge API. The heavy generate path is
    validated on hardware via `cli.experiments --phase generate`.

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
from experiments import driver, registry
from experiments.base import Retrievers, oracle_ladder_cells
from experiments.driver import _GuardRetriever, _SpecOnlyReasoner, experiment_paths
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


def test_headline_generate_then_judge_reuses_predictions(tmp_path: Path, fake_channels) -> None:
    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    headline = registry.EXPERIMENTS["T1_headline"]
    paths = experiment_paths(config, "T1_headline")

    # Phase 1: generate with a real (fake) reasoner into T1's prediction cache.
    gen_reasoner = CountingReasoner()
    gen = Orchestrator(
        config, reasoner=gen_reasoner, judge=StubJudge("gen"),
        cache=ResultCache(paths.generate_results), prediction_cache=PredictionCache(paths.predictions),
    )
    guards = Retrievers(_GuardRetriever("t"), _GuardRetriever("v"))
    cells = headline.generation_cells(config, questions, retrievers=guards)
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

    tables = headline.build(config, rows, paths.side_dir)
    assert list(tables["table1"]["bin"]) == list(config.bins)


def test_guards_refuse_to_run() -> None:
    q = Question(
        id="q1", doc_id="d.pdf", question="?", gold_answer="x", answer_format="String",
        doc_type="Brochure", evidence_pages=(1,), evidence_sources=("Chart",), hop="single", is_unanswerable=False,
    )
    with pytest.raises(RuntimeError, match="judge phase"):
        _GuardRetriever("colqwen").retrieve(q, 5, 1)
    with pytest.raises(RuntimeError, match="judge phase"):
        _SpecOnlyReasoner("spec").answer(q, ModelInput(parts=()))


def test_registry_resolve() -> None:
    assert [e.name for e in registry.resolve("T1_headline")] == ["T1_headline"]
    assert [e.name for e in registry.resolve("rq2")] == ["T5_composition", "T6_matched_cross"]
    assert [e.name for e in registry.resolve("section2")] == [
        "T1_headline",
        "T2_analytical",
        "T3_family",
        "T4_dataset",
        "T5_composition",
        "T6_matched_cross",
        "T7_routing",
    ]
    assert len(registry.resolve("all")) == 8
    with pytest.raises(ValueError):
        registry.resolve("nope")


def test_run_generate_continue_on_error_records_per_experiment_status(tmp_path: Path, monkeypatch) -> None:
    config = ExperimentConfig(smoke=True, paths=ProjectPaths(root=tmp_path, cache_dir=tmp_path / "results" / "cache"))
    experiments = [SimpleNamespace(name="ok"), SimpleNamespace(name="bad"), SimpleNamespace(name="after")]

    def fake_generate(config, exp, questions):  # noqa: ANN001
        if exp.name == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(driver, "resolve", lambda selector: experiments)
    monkeypatch.setattr(driver, "generate", fake_generate)

    statuses = driver.run_generate(config, "fake", [], continue_on_error=True)

    assert [status.status for status in statuses] == ["success", "failed", "success"]
    bad_status = experiment_paths(config, "bad").root / "generate_status.json"
    after_status = experiment_paths(config, "after").root / "generate_status.json"
    assert json.loads(bad_status.read_text())["error"] == "boom"
    assert json.loads(after_status.read_text())["status"] == "success"


def test_every_experiment_builds_its_table(tmp_path: Path, fake_channels) -> None:
    """Generate T1+T6, judge+build every experiment, expect all 8 table CSVs."""

    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)

    # Inject a fake reasoner into the driver by monkeypatching the backend lookup.
    import experiments.driver as drv

    drv_reasoner = CountingReasoner()
    orig = drv._reasoner_for
    drv._reasoner_for = lambda spec, config=None: drv_reasoner  # type: ignore[assignment]
    try:
        # Generate the two experiments that have reasoner cells (T1, T6). The
        # judge/build path never calls run_side, so the classifier/retrieval side
        # work (real models) is not exercised here.
        driver.generate(config, registry.EXPERIMENTS["T1_headline"], questions)
        _generate_t6_with_stub(config, questions, drv_reasoner)
        written = driver.run_judge(config, "all", questions, judge_impl=KeywordJudge())
    finally:
        drv._reasoner_for = orig  # type: ignore[assignment]

    assert set(written) == {f"table{i}" for i in range(1, 9)}
    for key, path in written.items():
        assert path.exists(), key
        if key != "table6":
            assert len(pd.read_csv(path)) > 0, key


def _generate_t6_with_stub(config, questions, reasoner) -> None:
    """Generate T6 cells with a deterministic first-page retriever."""

    from covariates.retriever import StubRetriever

    exp = registry.EXPERIMENTS["T6_matched_cross"]
    paths = experiment_paths(config, exp.name)
    orch = Orchestrator(
        config, reasoner=reasoner, judge=StubJudge("gen"),
        cache=ResultCache(paths.generate_results), prediction_cache=PredictionCache(paths.predictions),
    )
    retrievers = Retrievers(StubRetriever(), StubRetriever())
    for cell in exp.generation_cells(config, questions, retrievers=retrievers):
        orch.run_cell(cell.question, cell.conditioner, cell.representation)
