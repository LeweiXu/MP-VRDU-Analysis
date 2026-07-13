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
        SimpleNamespace(id="q1", doc_id="docA", doc_type="Academic paper", bin_label="text-dominant",
                        evidence_pages=(0,), evidence_sources=("Pure-text (Plain-text)",)),
        SimpleNamespace(id="q2", doc_id="docB", doc_type="Brochure", bin_label="visual-dominant",
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


def test_skip_retrieval_gates_stage1(tmp_path) -> None:
    """--skip-retrieval skips the stage-1 benchmark on a normal run (side work still runs)."""

    from experiments.engine import driver

    calls = {"benchmark": 0, "side": 0}

    class FakeTask:
        name = "G2_retrieval"

        def model_specs(self, config):
            return ()  # empty -> no reasoner loop, so generate stays light

        def resolve_questions(self, config, questions):
            return []

        def generation_cells(self, config, questions, *, retrievers):
            return []

        def run_retrieval_benchmark(self, config, questions, side_dir, *, limit=None):
            calls["benchmark"] += 1

        def run_side(self, config, questions, side_dir, *, limit=None):
            calls["side"] += 1

    def make_config():
        paths = SimpleNamespace(cache_dir=tmp_path / "cache", data_dir=tmp_path / "data",
                                results_dir=tmp_path / "results", root=tmp_path)
        for p in (paths.cache_dir, paths.data_dir, paths.results_dir):
            p.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(paths=paths, smoke=False, run_tag="t", dpi=200,
                               text_retrievers=("bge-m3",), inference_text_retriever="none",
                               inference_vision_retriever="none",
                               visual_resolutions=("med",), visual_resolution="med")

    driver.generate(make_config(), FakeTask(), [], skip_retrieval=False)
    assert calls["benchmark"] == 1  # normal run: stage-1 benchmark runs

    calls["benchmark"] = 0
    driver.generate(make_config(), FakeTask(), [], skip_retrieval=True)
    assert calls["benchmark"] == 0  # skipped
    assert calls["side"] == 2       # side work still runs both times


def test_memo_records_truncation_telemetry(tmp_path) -> None:
    """A dense retriever's `last_seq_stats` is written into the memo row, and the
    loader still reads the ranking (extra keys are ignored)."""

    from retrievers import MemoizedRetriever, Retriever

    class _Dense(Retriever):
        name = "qwen3-embedding"
        modality = "text"
        dpi = 200
        last_seq_stats = {"seq_len_cap": 4096, "page_token_lens": [100, 9000], "truncated_pages": 1}

        def rank(self, question, page_count):
            return (1, 0)

        def retrieve(self, question, page_count, k):
            return self.rank(question, page_count)[:k]

    q = SimpleNamespace(id="mmlongbench:1", doc_id="docA")
    MemoizedRetriever(_Dense(), persist_dir=tmp_path).rank(q, 2)

    row = json.loads((tmp_path / "qwen3-embedding__dpi200.jsonl").read_text().strip())
    assert row["seq_len_cap"] == 4096
    assert row["page_token_lens"] == [100, 9000]
    assert row["truncated_pages"] == 1
    # the loader ignores the extra keys and still recovers the ranking
    assert MemoizedRetriever(_Dense(), persist_dir=tmp_path)._cache[("mmlongbench:1", 2)] == (1, 0)


def test_reuse_only_memo(tmp_path) -> None:
    """reuse_only returns a cached ranking but raises RetrievalMemoMiss on a miss
    (carrying any recorded failure reason) instead of re-ranking, so a --skip-retrieval
    inference cell records the failure and rides on rather than silently re-ranking."""

    from retrievers import MemoizedRetriever, RetrievalMemoMiss

    memo = tmp_path / "bge-m3__dpi200.jsonl"
    memo.write_text("\n".join([
        json.dumps({"question_id": "hit", "page_count": 5, "ranking": [2, 0, 1]}),
        json.dumps({"question_id": "oomed", "page_count": 5, "ranking": [],
                    "status": "oom", "skipped_reason": "CUDA out of memory"}),
    ]) + "\n")

    class _Inner:
        name = "bge-m3"
        modality = "text"
        dpi = 200

        def rank(self, question, page_count):
            raise AssertionError("reuse-only must not re-rank")

    r = MemoizedRetriever(_Inner(), persist_dir=tmp_path, reuse_only=True)
    assert r.rank(SimpleNamespace(id="hit", doc_id="d"), 5) == (2, 0, 1)  # cached: served
    with pytest.raises(RetrievalMemoMiss):                                # never ranked
        r.rank(SimpleNamespace(id="missing", doc_id="d"), 5)
    with pytest.raises(RetrievalMemoMiss, match="out of memory"):         # carries earlier reason
        r.rank(SimpleNamespace(id="oomed", doc_id="d"), 5)


def test_skip_oom_drops_oom_cells(tmp_path) -> None:
    """--skip-oom removes cells recorded as oom from the run, keeping ok/missing ones."""

    from experiments.engine import driver

    # A predictions file: one ok cell, one oom cell, at the same resolution.
    preds = tmp_path / "cache" / "predictions.jsonl"
    preds.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"question_id": "q1", "doc_id": "docA", "condition": "oracle", "representation": "TLV",
         "model_spec": "m", "visual_resolution": "med", "status": "ok"},
        {"question_id": "q2", "doc_id": "docB", "condition": "oracle", "representation": "TLV",
         "model_spec": "m", "visual_resolution": "med", "status": "oom"},
    ]
    preds.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    oom = driver._oom_cell_ids(preds)
    assert oom == {("q2", "docB", "oracle", "TLV", "m", "med")}
    # the ok cell is not in the drop set, so it still runs
    assert ("q1", "docA", "oracle", "TLV", "m", "med") not in oom
    # no file rewrite (unlike --failed-only): both rows survive on disk
    assert len(preds.read_text().splitlines()) == 2


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


def test_question_failure_skips_only_that_question(tmp_path, monkeypatch) -> None:
    """A per-question OOM skips only that question; the method's other rows and memo
    entries still land (like predictions.jsonl continuing past a failed cell)."""

    import experiments.engine.side_artifacts as sa

    _patch_page_count(monkeypatch)

    class _OOMOnQ2(_FakeRetriever):
        def rank(self, question, page_count):
            if question.id == "q2":
                raise RuntimeError("CUDA out of memory")
            return super().rank(question, page_count)

    monkeypatch.setattr(sa, "_build_retriever", lambda config, name, kind: _OOMOnQ2(name, kind, config.dpi))
    cfg = _cfg(tmp_path)
    memo_path = cfg.paths.cache_dir / "retrieval" / "bge-m3__dpi200.jsonl"
    sa.write_retrieval_eval(cfg, _questions(), tmp_path, single_ks=(1,), joint_ks=(),
                            text_methods=("bge-m3",), vision_methods=(), joint_pairs=())

    # q2 OOM'd: it is not scored into the benchmark, but IS recorded in the memo.
    rows = [json.loads(line) for line in (tmp_path / "retrieval.jsonl").read_text().splitlines() if line.strip()]
    assert {r["question_id"] for r in rows} == {"q1"}
    memo = {m["question_id"]: m for m in
            (json.loads(line) for line in memo_path.read_text().splitlines() if line.strip())}
    assert set(memo) == {"q1", "q2"}
    assert memo["q1"].get("status", "ok") == "ok"
    assert memo["q2"]["status"] == "oom"
    assert "out of memory" in memo["q2"]["skipped_reason"].lower()

    # A resume re-attempts q2 (still OOMs) but does not duplicate its failure row.
    sa.write_retrieval_eval(cfg, _questions(), tmp_path, single_ks=(1,), joint_ks=(),
                            text_methods=("bge-m3",), vision_methods=(), joint_pairs=())
    lines = [line for line in memo_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2  # q1 ranking + one q2 failure row, no duplicate


def test_fresh_reranks_the_whole_rung(tmp_path, monkeypatch) -> None:
    """fresh=True wipes the method's memo first, so a stale row cannot survive."""

    import experiments.engine.side_artifacts as sa

    _patch_page_count(monkeypatch)
    monkeypatch.setattr(sa, "_build_retriever", lambda config, name, kind: _FakeRetriever(name, kind, config.dpi))
    cfg = _cfg(tmp_path)
    memo_path = cfg.paths.cache_dir / "retrieval" / "bge-m3__dpi200.jsonl"
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text(json.dumps({"question_id": "stale", "page_count": 5, "ranking": [0]}) + "\n")

    sa.write_retrieval_eval(cfg, _questions(), tmp_path, single_ks=(1,), joint_ks=(),
                            text_methods=("bge-m3",), vision_methods=(), joint_pairs=(), fresh=True)

    memo_ids = {json.loads(line)["question_id"] for line in memo_path.read_text().splitlines() if line.strip()}
    assert "stale" not in memo_ids  # fresh deleted the old memo before re-ranking
    assert memo_ids == {"q1", "q2"}
