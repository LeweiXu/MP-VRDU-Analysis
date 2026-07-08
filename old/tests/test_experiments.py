"""Test the role-split experiment pipeline: generation tasks, judge, build.

Purpose:
    Covers the refactor that organizes `experiments/` by generation task (G1..G6)
    rather than by paper table, with three separate roles:

    - `experiments.driver` runs a task's cells once per reasoner spec and caches
      predictions (the `PredictionCache` lets judging reuse them), then re-scores
      them with guards that refuse to run a reasoner/retriever (`_GuardRetriever`,
      `_SpecOnlyReasoner`);
    - task definitions live one-per-file in `experiments/G*_*.py`
      (`experiments.registry` collects them);
    - `reporting.build` routes each task's judged rows into the eight table
      CSVs; a table with no source rows still emits a skeleton.

Test role:
    Fixture PDFs + injected fake reasoner/judge, so the wiring is exercised
    without Marker, Qwen, ColQwen, or a judge API. The heavy generate path is
    validated on hardware via `python -m cli.generate`.

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
from experiments import driver
from experiments.base import Retrievers
from experiments.driver import _GuardRetriever, _SpecOnlyReasoner
from experiments.paths import experiment_paths
from experiments.registry import GENERATION_TASKS, resolve
from reporting import build as reporting_build
from reporting.tables import build_table1_headline
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


def test_matched_cross_sweep_cells_cover_every_k() -> None:
    from covariates.retriever import StubRetriever
    from experiments.base import matched_cross_sweep_cells

    question = Question(
        id="q1", doc_id="d.pdf", question="?", gold_answer="x", answer_format="String",
        doc_type="Brochure", evidence_pages=(0,), evidence_sources=("Chart",),
        hop="single", is_unanswerable=False,
    )
    cells = matched_cross_sweep_cells(
        [question], retrievers=Retrievers(StubRetriever(), StubRetriever()), ks=(1, 3, 5)
    )
    # question-major, k-minor, matched (vision) then cross (text) per k
    assert [c.conditioner.name for c in cells] == [
        "retrieved_vision_k1", "retrieved_text_k1",
        "retrieved_vision_k3", "retrieved_text_k3",
        "retrieved_vision_k5", "retrieved_text_k5",
    ]
    assert all(c.representation == "TLV" for c in cells)


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
    with pytest.raises(RuntimeError, match="no weights loaded"):
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

    def fake_generate(config, task, questions, *, skip_failed_cells=False):  # noqa: ANN001
        if task.name == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(driver, "resolve", lambda selector: tasks)
    monkeypatch.setattr(driver, "generate", fake_generate)

    statuses = driver.run_generate(config, "fake", [], continue_on_error=True)

    assert [status.status for status in statuses] == ["success", "failed", "success"]
    bad_status = experiment_paths(config, "bad").root / "generate_status.json"
    after_status = experiment_paths(config, "after").root / "generate_status.json"
    assert json.loads(bad_status.read_text())["error"] == "boom"
    assert json.loads(after_status.read_text())["status"] == "success"


def test_build_gates_tables_on_finished_dependencies(tmp_path: Path, fake_channels, monkeypatch) -> None:
    """Only tables whose source tasks' generate+judge finished get a CSV."""

    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)

    fake = CountingReasoner()
    monkeypatch.setattr(driver, "reasoner_for", lambda spec, config=None: fake)

    # Only G1 is generated and judged; G2/G3/G5/G6 never run.
    driver.generate(config, GENERATION_TASKS["G1_sufficiency"], questions)
    driver.run_judge(config, "G1_sufficiency", questions, judge_impl=KeywordJudge(), continue_on_error=True)
    written = reporting_build.build_tables(config, config.paths.results_dir / "tables" / "smoke",
                                 n_bootstrap=0, markdown_path=tmp_path / "all_tables.md")

    # G1-sourced tables build and have rows.
    for key in ("table1", "table2", "table5"):
        assert key in written and written[key].exists(), key
    assert len(pd.read_csv(written["table1"])) > 0
    # Tables whose dependencies aren't finished are not written at all (no
    # misleading skeleton). table8 is blocked because its G4 scale task
    # doesn't exist; table3/4/6/7 wait on G2/G3/G5/G6.
    from reporting.tables import TABLE_FILENAMES
    for key in ("table3", "table4", "table6", "table7", "table8"):
        assert key not in written, key
        assert not (config.paths.results_dir / "tables" / "smoke" / TABLE_FILENAMES[key]).exists(), key
    assert written["markdown"].exists()


def test_prewarm_runs_before_reasoner_is_built(tmp_path: Path, fake_channels, monkeypatch) -> None:
    """The parse pre-pass must warm caches before reasoner_for builds any model.

    Regression guard: the driver used to construct the reasoner at the top of the
    per-spec loop, before the pre-pass, contradicting the "reasoner not loaded"
    design. This asserts the first thing that happens is a prewarm, not a reasoner
    build, while still confirming the real reasoner is built for inference.
    """

    config = _make_config(tmp_path)
    questions = load_mmlongbench(config.paths.data_dir)
    task = GENERATION_TASKS["G1_sufficiency"]

    events: list[str] = []

    def spy_reasoner_for(spec, config=None):  # noqa: ANN001
        events.append("reasoner_for")
        return CountingReasoner()

    monkeypatch.setattr(driver, "reasoner_for", spy_reasoner_for)

    real_prewarm = Orchestrator.prewarm_cell

    def spy_prewarm(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        events.append("prewarm")
        return real_prewarm(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, "prewarm_cell", spy_prewarm)

    driver.generate(config, task, questions)

    assert "prewarm" in events and "reasoner_for" in events
    # The pre-pass starts before any reasoner is built, so the very first event is
    # a prewarm; the old bug would have made reasoner_for come first.
    assert events[0] == "prewarm"


def test_one_task_failure_does_not_abort_later_tasks(tmp_path: Path, monkeypatch) -> None:
    """A failing task (e.g. a G2 OOM) records failed and the next task still runs.

    Regression guard: the run loop used to re-raise on the first task failure when
    continue_on_error was off, so an OOM in G2 silently dropped G5. Task-level
    isolation is now unconditional; continue_on_error only governs cell skipping.
    """

    config = ExperimentConfig(smoke=True, paths=ProjectPaths(root=tmp_path, cache_dir=tmp_path / "results" / "cache"))
    tasks = [SimpleNamespace(name="g1"), SimpleNamespace(name="g2"), SimpleNamespace(name="g5")]

    def fake_generate(config, task, questions, *, skip_failed_cells=False):  # noqa: ANN001
        if task.name == "g2":
            raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(driver, "resolve", lambda selector: tasks)
    monkeypatch.setattr(driver, "generate", fake_generate)

    # Note: continue_on_error is False (the default the failing job used).
    statuses = driver.run_generate(config, "fake", [], continue_on_error=False)

    assert [s.status for s in statuses] == ["success", "failed", "success"]
    assert json.loads((experiment_paths(config, "g5").root / "generate_status.json").read_text())["status"] == "success"


def test_generate_fails_when_every_cell_is_skipped(tmp_path: Path, fake_channels, monkeypatch) -> None:
    """A task whose every cell fails must not report generate-success (no predictions)."""

    class AlwaysBoom(CountingReasoner):
        def answer(self, question: Question, model_input: ModelInput) -> Prediction:
            raise RuntimeError("every cell fails (e.g. a missing dependency like timm)")

    config = _make_config(tmp_path)
    task = GENERATION_TASKS["G1_sufficiency"]
    monkeypatch.setattr(driver, "reasoner_for", lambda spec, config=None: AlwaysBoom())

    # Even with skip-on-error, an all-skipped task raises so run_generate records failed.
    with pytest.raises(RuntimeError, match="all .* reasoner cell"):
        driver.generate(config, task, load_mmlongbench(config.paths.data_dir), skip_failed_cells=True)


def test_generate_skips_failed_cells_when_continue_on_error(tmp_path: Path, fake_channels, monkeypatch) -> None:
    """A cell that raises (e.g. a many-page OOM) is skipped, not fatal, when asked."""

    class BoomOnVision(CountingReasoner):
        def answer(self, question: Question, model_input: ModelInput) -> Prediction:
            if model_input.image_parts:  # stand-in for the many-page attention OOM
                raise RuntimeError("CUDA out of memory (simulated)")
            return super().answer(question, model_input)

    task = GENERATION_TASKS["G1_sufficiency"]

    # Without the flag, the failing vision cell aborts the task.
    strict = _make_config(tmp_path / "strict")
    monkeypatch.setattr(driver, "reasoner_for", lambda spec, config=None: BoomOnVision())
    with pytest.raises(RuntimeError, match="out of memory"):
        driver.generate(strict, task, load_mmlongbench(strict.paths.data_dir), skip_failed_cells=False)

    # With the flag, vision cells are skipped and the text cells still cache.
    lenient = _make_config(tmp_path / "lenient")
    monkeypatch.setattr(driver, "reasoner_for", lambda spec, config=None: BoomOnVision())
    driver.generate(lenient, task, load_mmlongbench(lenient.paths.data_dir), skip_failed_cells=True)
    cached = PredictionCache(experiment_paths(lenient, task.name).predictions)
    reps = {row.representation for row in cached}
    assert reps and reps <= {"T", "TL"}  # the image-bearing TLV/V cells were skipped


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
