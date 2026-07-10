"""Side-artifact scope after the driver hands `run_side` the full corpus: G3's
folded-in classifier prices G1's answerable doc set (not G3's unanswerable cells)
and only when a classifier is configured; G2's retrieval benchmark still re-filters
to its answerable pool."""

from __future__ import annotations

from types import SimpleNamespace

from config import ExperimentConfig
from experiments.tasks.G2_retrieval import G2Retrieval
from experiments.tasks.G3_hallucination import G3Hallucination


def _q(qid: str, doc: str, unanswerable: bool = False, doc_type: str = "report"):
    return SimpleNamespace(id=qid, doc_id=doc, bin_label="text-dominant",
                           is_unanswerable=unanswerable, doc_type=doc_type)


def _corpus():
    # docA/docB answerable, docU unanswerable (G3's own pool, must be excluded here).
    return [
        _q("q1", "docA"), _q("q2", "docA"),
        _q("q3", "docB"),
        _q("u1", "docU", unanswerable=True), _q("u2", "docU", unanswerable=True),
    ]


def test_g3_classifier_scores_answerable_docs_when_configured(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_write(config, questions, side_dir, *, filename):
        captured["docs"] = list(questions)
        captured["filename"] = filename

    monkeypatch.setattr("experiments.tasks.G3_hallucination.write_classifier_eval", fake_write)

    cfg = ExperimentConfig(classifier_spec="qwen3vl-2b-local", per_doc_type_sample=None, sample_seed=0)
    G3Hallucination().run_side(cfg, _corpus(), tmp_path)

    doc_ids = {q.doc_id for q in captured["docs"]}
    assert doc_ids == {"docA", "docB"}, "classifier must price the answerable pool"
    assert all(not q.is_unanswerable for q in captured["docs"])
    assert captured["filename"] == "classifier.jsonl"


def test_g3_classifier_is_noop_when_unset(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}
    monkeypatch.setattr(
        "experiments.tasks.G3_hallucination.write_classifier_eval",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
    )
    G3Hallucination().run_side(ExperimentConfig(classifier_spec=None), _corpus(), tmp_path)
    assert calls["n"] == 0

    # "none" (any case) is normalized to disabled too.
    G3Hallucination().run_side(ExperimentConfig(classifier_spec="None"), _corpus(), tmp_path)
    assert calls["n"] == 0


def test_g3_classifier_respects_smoke_limit(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "experiments.tasks.G3_hallucination.write_classifier_eval",
        lambda config, questions, side_dir, *, filename: captured.__setitem__("docs", list(questions)),
    )
    cfg = ExperimentConfig(classifier_spec="qwen3vl-2b-local", per_doc_type_sample=None)
    G3Hallucination().run_side(cfg, _corpus(), tmp_path, limit=1)
    assert len(captured["docs"]) == 1


def test_g2_retrieval_side_refilters_to_answerable(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_write(config, questions, side_dir, *, single_ks, joint_ks, filename):
        captured["docs"] = list(questions)

    monkeypatch.setattr("experiments.tasks.G2_retrieval.write_retrieval_eval", fake_write)

    cfg = ExperimentConfig(per_doc_type_sample=None)
    G2Retrieval().run_side(cfg, _corpus(), tmp_path)

    assert all(not q.is_unanswerable for q in captured["docs"]), "G2 must exclude unanswerable questions"
    assert {q.doc_id for q in captured["docs"]} == {"docA", "docB"}
