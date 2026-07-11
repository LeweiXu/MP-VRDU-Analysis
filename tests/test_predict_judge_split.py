"""Generate writes an unjudged predictions.jsonl (every cell); the judge phase adds
the verdict fields to produce results.jsonl as a strict superset, row for row."""
from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

from schema import PredictionRow, Question, ResultRow, Score


def _prediction_row(prediction_key: str, question_id: str, *, status: str = "ok",
                    answer: str = "the answer is 42") -> PredictionRow:
    return PredictionRow(
        prediction_key=prediction_key, question_id=question_id, doc_id="d1",
        doc_type="report", bin_label="text-dominant", scan_label="digital", hop="single",
        is_unanswerable=False, evidence_sources=("text",), condition="oracle__none",
        provenance="oracle", page_indices=(0,), representation="V", model_spec="m",
        machine="local", status=status, skipped_reason="" if status == "ok" else "boom",
        oom_occurred=False, answer=answer if status == "ok" else "",
        total_text_tokens=10, total_visual_tokens=5, text_tokens_fed=10, output_tokens=3,
        tokens_dropped=0, truncation_occurred=False, latency_s=1.0, prefill_latency_s=0.5,
        decode_latency_s=0.5, peak_vram_bytes=123, visual_resolution="med", note="",
        metadata={"source_dataset": "mmlongbench"},
    )


def _question(qid: str, gold: str = "42") -> Question:
    return Question(id=qid, doc_id="d1", question="what?", gold_answer=gold,
                    answer_format="str", doc_type="report", evidence_pages=(0,),
                    evidence_sources=("text",), hop="single", is_unanswerable=False)


def test_prediction_row_omits_judge_fields():
    """predictions.jsonl carries no judge-derived fields; results.jsonl adds exactly those."""

    pred_names = {f.name for f in fields(PredictionRow)}
    result_names = {f.name for f in fields(ResultRow)}
    assert result_names - pred_names == {"result_key", "judge_spec", "score", "correct", "abstained"}
    assert pred_names < result_names  # strict subset


def test_to_result_row_is_a_superset():
    pred = _prediction_row("pk1", "q1")
    row = pred.to_result_row(Score(value=1.0, correct=True, abstained=False, judge_spec="stub"), "rk1")
    # every predictions field is preserved verbatim
    for f in fields(PredictionRow):
        assert getattr(row, f.name) == getattr(pred, f.name), f.name
    # and the judge phase adds exactly the verdict + result key
    assert (row.result_key, row.judge_spec, row.score, row.correct, row.abstained) == ("rk1", "stub", 1.0, True, False)


def test_judge_run_scores_ok_and_carries_failures(tmp_path):
    from ops.judge import judge_run
    from pipeline.judge import StubJudge
    from pipeline.orchestrator import PredictionCache, ResultCache

    config = SimpleNamespace(
        smoke=False, run_tag="t", dataset="mmlongbench",
        paths=SimpleNamespace(cache_dir=tmp_path / "cache", results_dir=tmp_path / "results"),
    )
    from experiments.engine.paths import experiment_paths
    paths = experiment_paths(config, "G1_oracle_ladder")

    cache = PredictionCache(paths.predictions)
    cache.put(_prediction_row("pk-ok", "q1", answer="the answer is 42"))
    cache.put(_prediction_row("pk-bad", "q2", status="error"))

    questions = {"q1": _question("q1"), "q2": _question("q2")}
    written = judge_run(config, "G1_oracle_ladder", questions, StubJudge("stub"))
    assert written == 2

    results = list(ResultCache(paths.results))
    by_pk = {r.prediction_key: r for r in results}
    # strict superset: same rows keyed by prediction_key, judge fields added
    assert set(by_pk) == {"pk-ok", "pk-bad"}
    assert by_pk["pk-ok"].correct is True and by_pk["pk-ok"].judge_spec == "stub"
    # a failed cell passes through unscored
    bad = by_pk["pk-bad"]
    assert bad.status == "error" and bad.score == 0.0 and bad.correct is False and bad.judge_spec == "stub"


def test_run_cell_writes_unjudged_prediction(tmp_path):
    """Generate's run_cell returns a PredictionRow and caches it, with no judging."""

    from pipeline.conditioner import OracleConditioner
    from pipeline.orchestrator import Orchestrator, PredictionCache
    from schema import Prediction

    class FakeReasoner:
        spec = "fake-2b"
        prompt_instruction = ""

        def answer(self, question, model_input):
            return Prediction(text="the answer is 42", model_spec="fake-2b",
                              total_text_tokens=4, total_visual_tokens=0, text_tokens_fed=4,
                              output_tokens=2, latency_s=0.1, peak_vram_bytes=1)

    config = SimpleNamespace(
        smoke=False, run_tag="t", dataset="mmlongbench", parser_tool="paddleocrvl", dpi=200,
        visual_resolution="min", reasoner_spec="fake-2b",
        paths=SimpleNamespace(cache_dir=tmp_path / "cache", data_dir=tmp_path / "data",
                              results_dir=tmp_path / "results"),
    )
    cache = PredictionCache(tmp_path / "predictions.jsonl")
    orch = Orchestrator(config, reasoner=FakeReasoner(), prediction_cache=cache,
                        visual_resolution="min")
    orch.page_count = lambda q: 3          # avoid resolving a real PDF
    orch.render_pages = lambda q, ps: []   # V build over no pages -> empty payload

    row = orch.run_cell(_question("q1"), OracleConditioner(), "V", prompt_mode="none")

    assert row.status == "ok" and row.answer == "the answer is 42" and row.representation == "V"
    # no judge fields exist on a PredictionRow at all
    assert not hasattr(row, "judge_spec") and not hasattr(row, "score")
    # it was persisted to predictions.jsonl and a re-run is a cache hit (no second answer)
    assert cache.get(row.prediction_key) is not None
    assert list(PredictionCache(tmp_path / "predictions.jsonl"))[0].answer == "the answer is 42"


def test_judge_run_is_resumable(tmp_path):
    from ops.judge import judge_run
    from pipeline.judge import StubJudge
    from pipeline.orchestrator import PredictionCache

    config = SimpleNamespace(
        smoke=False, run_tag="t", dataset="mmlongbench",
        paths=SimpleNamespace(cache_dir=tmp_path / "cache", results_dir=tmp_path / "results"),
    )
    from experiments.engine.paths import experiment_paths
    paths = experiment_paths(config, "G1_oracle_ladder")
    PredictionCache(paths.predictions).put(_prediction_row("pk-ok", "q1"))
    questions = {"q1": _question("q1")}

    assert judge_run(config, "G1_oracle_ladder", questions, StubJudge("stub")) == 1
    # a second pass with the same judge finds the row already scored (deduped on result_key)
    assert judge_run(config, "G1_oracle_ladder", questions, StubJudge("stub")) == 0
