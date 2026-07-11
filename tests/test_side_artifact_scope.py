"""Side-artifact scope for the unified task: the classifier prices the answerable
doc set (not the unanswerable cells) and only when configured; the retrieval
benchmark re-filters to its answerable pool."""

from __future__ import annotations

from types import SimpleNamespace

from config import ExperimentConfig
from experiments.tasks.task import Task


def _q(qid: str, doc: str, unanswerable: bool = False, doc_type: str = "report"):
    return SimpleNamespace(id=qid, doc_id=doc, bin_label="text-dominant",
                           is_unanswerable=unanswerable, doc_type=doc_type)


def _corpus():
    # docA/docB answerable, docU unanswerable (must be excluded from both side jobs).
    return [
        _q("q1", "docA"), _q("q2", "docA"),
        _q("q3", "docB"),
        _q("u1", "docU", unanswerable=True), _q("u2", "docU", unanswerable=True),
    ]


def test_classifier_scores_answerable_docs_when_configured(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_write(config, questions, side_dir, *, filename):
        captured["docs"] = list(questions)
        captured["filename"] = filename

    monkeypatch.setattr("experiments.tasks.task.write_classifier_eval", fake_write)

    cfg = ExperimentConfig(classifier_spec="qwen3vl-2b-local", per_doc_type_sample=None, sample_seed=0)
    Task("G3_hallucination").run_side(cfg, _corpus(), tmp_path)

    assert {q.doc_id for q in captured["docs"]} == {"docA", "docB"}
    assert all(not q.is_unanswerable for q in captured["docs"])
    assert captured["filename"] == "classifier.jsonl"


def test_classifier_is_noop_when_unset(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}
    monkeypatch.setattr(
        "experiments.tasks.task.write_classifier_eval",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
    )
    Task("G3_hallucination").run_side(ExperimentConfig(classifier_spec=None), _corpus(), tmp_path)
    assert calls["n"] == 0
    # "none" (any case) is normalized to disabled too.
    Task("G3_hallucination").run_side(ExperimentConfig(classifier_spec="None"), _corpus(), tmp_path)
    assert calls["n"] == 0


def test_classifier_respects_smoke_limit(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "experiments.tasks.task.write_classifier_eval",
        lambda config, questions, side_dir, *, filename: captured.__setitem__("docs", list(questions)),
    )
    cfg = ExperimentConfig(classifier_spec="qwen3vl-2b-local", per_doc_type_sample=None)
    Task("G3_hallucination").run_side(cfg, _corpus(), tmp_path, limit=1)
    assert len(captured["docs"]) == 1


def test_retrieval_benchmark_refilters_to_answerable(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_write(config, questions, side_dir, *, single_ks, joint_ks, filename, **kwargs):
        captured["docs"] = list(questions)

    monkeypatch.setattr("experiments.tasks.task.write_retrieval_eval", fake_write)

    cfg = ExperimentConfig(per_doc_type_sample=None)  # pool defaults to answerable
    Task("G2_retrieval").run_retrieval_benchmark(cfg, _corpus(), tmp_path)

    assert all(not q.is_unanswerable for q in captured["docs"]), "benchmark must exclude unanswerable"
    assert {q.doc_id for q in captured["docs"]} == {"docA", "docB"}
