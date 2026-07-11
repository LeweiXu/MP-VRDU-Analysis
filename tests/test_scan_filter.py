"""The document scan filter (digital/scanned): PyMuPDF auto-detection cached to
annotations/auto_scan.csv, applied before the pool + sampling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import ExperimentConfig


def test_auto_scan_labels_classifies_then_caches(tmp_path, monkeypatch) -> None:
    import data.loader
    import data.render
    from experiments.corpus.resolve import auto_scan_labels

    monkeypatch.setattr(data.loader, "resolve_pdf", lambda doc_id, data_dir: f"/pdf/{doc_id}")
    by_doc = {"dA": "digital", "dB": "scanned"}
    monkeypatch.setattr(data.render, "classify_scanned",
                        lambda pdf: SimpleNamespace(label=by_doc[str(pdf).rsplit("/", 1)[-1]]))

    csv_path = tmp_path / "auto_scan.csv"
    got = auto_scan_labels(["dA", "dB"], None, csv_path)
    assert got == {"dA": "digital", "dB": "scanned"}
    assert csv_path.read_text().splitlines()[0] == "doc_id,auto_scan"

    # Second call reads the cache and does not re-classify.
    monkeypatch.setattr(data.render, "classify_scanned",
                        lambda pdf: (_ for _ in ()).throw(AssertionError("should be cached")))
    assert auto_scan_labels(["dA", "dB"], None, csv_path) == {"dA": "digital", "dB": "scanned"}


def test_filter_by_scan(tmp_path, monkeypatch) -> None:
    from experiments.corpus import resolve
    from experiments.corpus.resolve import filter_by_scan

    monkeypatch.setattr(resolve, "auto_scan_labels",
                        lambda doc_ids, data_dir, csv_path: {"dA": "digital", "dB": "scanned"})
    corpus = [SimpleNamespace(doc_id="dA", id="q1"), SimpleNamespace(doc_id="dB", id="q2")]

    assert [q.id for q in filter_by_scan(corpus, "digital", data_dir=None, annotations_dir=tmp_path)] == ["q1"]
    assert [q.id for q in filter_by_scan(corpus, "scanned", data_dir=None, annotations_dir=tmp_path)] == ["q2"]
    assert len(filter_by_scan(corpus, "any", data_dir=None, annotations_dir=tmp_path)) == 2
    with pytest.raises(ValueError):
        filter_by_scan(corpus, "bogus", data_dir=None, annotations_dir=tmp_path)


def test_resolve_questions_applies_scan_before_pool_and_sampling(monkeypatch) -> None:
    from experiments.corpus import resolve
    from experiments.tasks.task import Task

    # dA digital+answerable, dB scanned+answerable, dC digital+unanswerable.
    qs = [
        SimpleNamespace(id="q1", doc_id="dA", is_unanswerable=False, doc_type="r"),
        SimpleNamespace(id="q2", doc_id="dB", is_unanswerable=False, doc_type="r"),
        SimpleNamespace(id="q3", doc_id="dC", is_unanswerable=True, doc_type="r"),
    ]
    monkeypatch.setattr(resolve, "auto_scan_labels",
                        lambda doc_ids, data_dir, csv_path: {"dA": "digital", "dB": "scanned", "dC": "digital"})

    cfg = ExperimentConfig(scan_filter="digital", pool="answerable", per_doc_type_sample=None)
    out = Task("G1_oracle_ladder").resolve_questions(cfg, qs)
    # digital -> {dA, dC}; then answerable -> {dA}. Scan must run first: dB (scanned,
    # answerable) is dropped by scan, dC (digital, unanswerable) by the pool.
    assert [q.id for q in out] == ["q1"]

    # "any" is a no-op: the pool alone keeps both answerable questions.
    cfg_any = ExperimentConfig(scan_filter="any", pool="answerable", per_doc_type_sample=None)
    assert {q.id for q in Task("G1_oracle_ladder").resolve_questions(cfg_any, qs)} == {"q1", "q2"}


def test_pool_all_keeps_both() -> None:
    from experiments.corpus.resolve import filter_by_pool

    corpus = [
        SimpleNamespace(id="q1", is_unanswerable=False),
        SimpleNamespace(id="q2", is_unanswerable=True),
    ]
    assert {q.id for q in filter_by_pool(corpus, "all")} == {"q1", "q2"}
    assert [q.id for q in filter_by_pool(corpus, "answerable")] == ["q1"]
    assert [q.id for q in filter_by_pool(corpus, "unanswerable")] == ["q2"]
    with pytest.raises(ValueError):
        filter_by_pool(corpus, "bogus")


def test_resolve_questions_honours_all_sampling_strategies() -> None:
    from experiments.tasks.task import Task

    # 3 answerable single-question docs across 2 bins.
    qs = [
        SimpleNamespace(id="q0", doc_id="d0", is_unanswerable=False, doc_type="r", bin_label="text-dominant"),
        SimpleNamespace(id="q1", doc_id="d1", is_unanswerable=False, doc_type="r", bin_label="text-dominant"),
        SimpleNamespace(id="q2", doc_id="d2", is_unanswerable=False, doc_type="r", bin_label="visual-dominant"),
    ]
    task = Task("G1_oracle_ladder")

    assert len(task.resolve_questions(ExperimentConfig(pool="all", sampling="full"), qs)) == 3
    assert len(task.resolve_questions(ExperimentConfig(pool="all", sampling={"limit": 2}), qs)) == 2
    got_ids = task.resolve_questions(ExperimentConfig(pool="all", sampling={"ids": ["q0", "q2"]}), qs)
    assert {q.id for q in got_ids} == {"q0", "q2"}
    got_bin = task.resolve_questions(ExperimentConfig(pool="all", sampling={"per_bin": 1, "seed": 0}), qs)
    assert {q.bin_label for q in got_bin} == {"text-dominant", "visual-dominant"}
    got_dt = task.resolve_questions(ExperimentConfig(pool="all", sampling={"per_doc_type": 1, "seed": 0}), qs)
    assert len(got_dt) == 1  # exactly one question for the single doc_type


def test_config_from_spec_maps_corpus_scan() -> None:
    from experiments.corpus.yaml_spec import config_from_spec, parse_specs

    (spec,) = parse_specs({
        "task_name": "G1_oracle_ladder", "run_tag": "g1",
        "corpus": {"scan": "scanned", "pool": "answerable", "sampling": "full"},
        "retrieval_representation": ["oracle"], "reasoner_spec": ["qwen3vl-8b-local"],
    })
    assert config_from_spec(spec).scan_filter == "scanned"
