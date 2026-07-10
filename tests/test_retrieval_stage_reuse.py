"""The retrieval benchmark (stage 1) persists each method's ranking to the shared
retrieval memo, so the inference stage reuses those rankings instead of ranking the
same method again. And rows are written per method (a method that fails to load loses
only its own rows). These guard the G2 stage-drift fix."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrievers import MemoizedRetriever


def _cfg(tmp_path: Path):
    paths = SimpleNamespace(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(paths=paths, dpi=200)


def _questions():
    # evidence_pages/sources are what score_retrieval reads; the rest is identity.
    return [
        SimpleNamespace(id="q1", doc_id="docA", bin_label="text-dominant",
                        evidence_pages=(0,), evidence_sources=("Pure-text (Plain-text)",)),
        SimpleNamespace(id="q2", doc_id="docB", bin_label="visual-dominant",
                        evidence_pages=(2, 3), evidence_sources=("Chart",)),
    ]


PAGE_COUNT = 5


def _ranking(name: str, page_count: int) -> tuple[int, ...]:
    """Deterministic, method-dependent page order (a real permutation)."""
    pages = list(range(page_count))
    return tuple(pages if name == "bge-m3" else list(reversed(pages)))


class _FakeRetriever:
    """Stands in for a real retriever: deterministic rank, cheap timers."""

    def __init__(self, name: str, modality: str, dpi: int):
        self.name = name
        self.modality = modality
        self.dpi = dpi
        self.last_query_s = 0.0
        self.index_build_s = 0.5

    def rank(self, question, page_count):
        self.last_query_s = 0.01
        return _ranking(self.name, page_count)

    def unload(self):
        pass


def _patch_page_count(monkeypatch):
    import data.loader
    import data.render

    monkeypatch.setattr(data.loader, "resolve_pdf", lambda doc_id, data_dir: Path(f"/nope/{doc_id}.pdf"))
    monkeypatch.setattr(data.render, "pdf_page_count", lambda _pdf: PAGE_COUNT)


def test_inference_reuses_stage1_ranking(tmp_path, monkeypatch) -> None:
    import experiments.engine.side_artifacts as sa

    _patch_page_count(monkeypatch)
    monkeypatch.setattr(
        sa, "_build_retriever",
        lambda config, name, kind: _FakeRetriever(name, kind, config.dpi),
    )

    cfg = _cfg(tmp_path)
    questions = _questions()
    single_ks = (1, 3)
    sa.write_retrieval_eval(
        cfg, questions, tmp_path,
        single_ks=single_ks, joint_ks=(),
        text_methods=("bge-m3",), vision_methods=("colqwen2.5",), joint_pairs=(),
    )

    # retrieval.jsonl rows carry the stage-1 ranking sliced to k.
    rows = [json.loads(line) for line in (tmp_path / "retrieval.jsonl").read_text().splitlines() if line.strip()]
    written = {(r["retriever"], r["question_id"], r["k"]): tuple(r["retrieved_pages"]) for r in rows}
    assert {r["retriever"] for r in rows} == {"bge-m3", "colqwen2.5"}
    for method in ("bge-m3", "colqwen2.5"):
        for q in questions:
            for k in single_ks:
                assert written[(method, q.id, k)] == _ranking(method, PAGE_COUNT)[:k]

    # The memo the inference stage reads exists, one file per method at this dpi.
    persist_dir = cfg.paths.cache_dir / "retrieval"
    for method in ("bge-m3", "colqwen2.5"):
        assert (persist_dir / f"{method}__dpi{cfg.dpi}.jsonl").exists()

    # Inference reuse: a MemoizedRetriever whose inner raises if ranked must still
    # return the stage-1 pages for the same (question, k) — i.e. it read the memo.
    class _Raises(_FakeRetriever):
        def rank(self, question, page_count):
            raise AssertionError("inference must reuse the stage-1 memo, not recompute")

    for method, modality in (("bge-m3", "text"), ("colqwen2.5", "vision")):
        inf = MemoizedRetriever(_Raises(method, modality, cfg.dpi), persist_dir=persist_dir)
        for q in questions:
            for k in single_ks:
                assert inf.retrieve(q, PAGE_COUNT, k) == written[(method, q.id, k)]


def test_failed_method_loses_only_its_rows(tmp_path, monkeypatch) -> None:
    import experiments.engine.side_artifacts as sa

    _patch_page_count(monkeypatch)

    def build(config, name, kind):
        if name == "qwen3-embedding":
            raise RuntimeError("model failed to load")
        return _FakeRetriever(name, kind, config.dpi)

    monkeypatch.setattr(sa, "_build_retriever", build)

    cfg = _cfg(tmp_path)
    sa.write_retrieval_eval(
        cfg, _questions(), tmp_path,
        single_ks=(1,), joint_ks=(),
        text_methods=("bge-m3", "qwen3-embedding"), vision_methods=(), joint_pairs=(),
    )

    rows = [json.loads(line) for line in (tmp_path / "retrieval.jsonl").read_text().splitlines() if line.strip()]
    methods = {r["retriever"] for r in rows}
    assert "bge-m3" in methods, "the working method's rows survive"
    assert "qwen3-embedding" not in methods, "the failed method loses only its own rows"
    # And no stale memo for the method that never ranked.
    persist_dir = cfg.paths.cache_dir / "retrieval"
    assert (persist_dir / f"bge-m3__dpi{cfg.dpi}.jsonl").exists()
    assert not (persist_dir / f"qwen3-embedding__dpi{cfg.dpi}.jsonl").exists()
