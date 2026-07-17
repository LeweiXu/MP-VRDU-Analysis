"""The reporting guards (model_spec pooling, clean cost column), the judge re-judge
no-op (no wasted API call), and a smoke build of each mined_* table."""

from __future__ import annotations

from types import SimpleNamespace

from config import DEFAULT_REASONER_SPEC


def _row(**over):
    """A result-like row with every field the builders / scoring touch."""

    base = dict(
        question_id="q", doc_id="d1", doc_type="Academic paper", condition="oracle",
        representation="T", model_spec=DEFAULT_REASONER_SPEC, visual_resolution="med",
        scan_label="digital", is_unanswerable=False, evidence_sources=("Table",),
        page_indices=(0,), status="ok", correct=True, abstained=False,
        total_text_tokens=100, total_visual_tokens=0, output_tokens=5,
        latency_s=2.0, prefill_latency_s=0.4, decode_latency_s=1.6, peak_vram_bytes=8_000_000_000,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ---- reporting guard: restrict_to_primary_spec -----------------------------------

def test_restrict_to_primary_spec_keeps_primary_when_pooled():
    from reporting.tables._common import restrict_to_primary_spec

    rows = [_row(model_spec=DEFAULT_REASONER_SPEC), _row(model_spec="qwen3vl-2b-local"),
            _row(model_spec="qwen3vl-32b-local")]
    kept = restrict_to_primary_spec(rows)
    assert {r.model_spec for r in kept} == {DEFAULT_REASONER_SPEC}


def test_restrict_to_primary_spec_passes_single_spec_through():
    from reporting.tables._common import restrict_to_primary_spec

    rows = [_row(model_spec="qwen3vl-2b-local"), _row(model_spec="qwen3vl-2b-local")]
    assert restrict_to_primary_spec(rows) == rows  # untouched when only one spec present


def test_headline_does_not_pool_specs():
    """A multi-spec cache must not average 2B/8B into one headline cell."""

    from reporting.tables import headline

    rows = [_row(model_spec=DEFAULT_REASONER_SPEC, correct=True),
            _row(model_spec="qwen3vl-2b-local", correct=False)]
    table = headline.build(rows)
    # n column is the last one; only the primary spec's single row survives
    assert table.rows and table.rows[0][-1] == "1"


# ---- judge re-judge no-op: second pass makes no judge.score() call ----------------

def test_rejudge_makes_no_score_call(tmp_path):
    from experiments.engine.paths import experiment_paths
    from ops.judge import judge_run
    from pipeline.judge import StubJudge
    from pipeline.orchestrator import PredictionCache
    from schema import PredictionRow, Question

    class CountingJudge(StubJudge):
        def __init__(self):
            super().__init__("stub")
            self.calls = 0

        def score(self, question, prediction):
            self.calls += 1
            return super().score(question, prediction)

    config = SimpleNamespace(
        smoke=False, run_tag="t", dataset="mmlongbench",
        paths=SimpleNamespace(cache_dir=tmp_path / "cache", results_dir=tmp_path / "results"),
    )
    paths = experiment_paths(config, "G1_oracle_ladder")
    pred = PredictionRow(
        prediction_key="pk", question_id="q1", doc_id="d1", doc_type="report",
        bin_label="", scan_label="digital", hop="single", is_unanswerable=False,
        evidence_sources=("text",), condition="oracle__none", provenance="oracle",
        page_indices=(0,), representation="V", model_spec="m", machine="local",
        status="ok", skipped_reason="", oom_occurred=False, answer="the answer is 42",
        total_text_tokens=10, total_visual_tokens=5, text_tokens_fed=10, output_tokens=3,
        tokens_dropped=0, truncation_occurred=False, latency_s=1.0, prefill_latency_s=0.5,
        decode_latency_s=0.5, peak_vram_bytes=1, visual_resolution="med", note="", metadata={},
    )
    PredictionCache(paths.predictions).put(pred)
    questions = {"q1": Question(id="q1", doc_id="d1", question="?", gold_answer="42",
                                answer_format="str", doc_type="report", evidence_pages=(0,),
                                evidence_sources=("text",), hop="single", is_unanswerable=False)}

    j1 = CountingJudge()
    assert judge_run(config, "G1_oracle_ladder", questions, j1) == 1 and j1.calls == 1
    j2 = CountingJudge()
    # already scored -> zero rows written AND zero judge calls (no wasted API cost)
    assert judge_run(config, "G1_oracle_ladder", questions, j2) == 0 and j2.calls == 0


# ---- mined builders: each produces a non-empty, well-formed table -----------------

def test_mined_scan_vs_digital():
    from reporting.tables import mined_scan_vs_digital

    rows = [_row(scan_label="digital", representation="T"),
            _row(scan_label="scanned", representation="T", correct=False)]
    table = mined_scan_vs_digital.build(rows)
    assert "digital" in table.columns and "scanned" in table.columns and table.rows


def test_mined_prefill_cost_is_decode_free():
    from reporting.tables import mined_prefill_cost

    table = mined_prefill_cost.build([_row(prefill_latency_s=0.4), _row(prefill_latency_s=0.6)])
    assert table.columns == ["doc_type", "rung", "prefill_ms", "input_tokens", "n"]
    assert table.rows and table.rows[0][2] == "500"  # mean 0.5s -> 500ms


def test_mined_vram_headroom_flags_ceiling():
    from reporting.tables import mined_vram_headroom

    table = mined_vram_headroom.build([_row(peak_vram_bytes=15_000_000_000)])
    assert "headroom_mb" in table.columns and table.rows


def test_mined_quant_sensitivity_deltas():
    from reporting.tables import mined_quant_sensitivity

    rows = [
        _row(model_spec=DEFAULT_REASONER_SPEC, correct=True),                 # 16-bit baseline
        _row(model_spec=f"{DEFAULT_REASONER_SPEC}-4bit", correct=False),      # quantized arm
    ]
    table = mined_quant_sensitivity.build(rows)
    quants = {r[1] for r in table.rows}
    assert {"16bit", "4bit"} <= quants


def test_mined_oom_frontier_reads_status():
    from reporting.tables import mined_oom_frontier

    rows = [_row(status="ok"), _row(status="oom"), _row(status="oom")]
    table = mined_oom_frontier.build(rows)
    # 2 of 3 cells OOM'd in the single (T, med, 1-page) group
    assert table.rows[0][-2:] == ["2", "3"] and table.rows[0][3] == "66.7"


def test_mined_abstention_by_doctype():
    from reporting.tables import mined_abstention_by_doctype

    rows = [_row(is_unanswerable=True, condition="similarity__targeted", abstained=True),
            _row(is_unanswerable=True, condition="similarity__none", abstained=False)]
    table = mined_abstention_by_doctype.build(rows)
    assert "targeted" in table.columns and "none" in table.columns and table.rows


def test_gemini_judge_survives_unparseable_response():
    """An empty / non-JSON Gemini response must not crash the run; it falls back."""

    from types import SimpleNamespace

    from pipeline.judge import GeminiJudge
    from schema import Prediction, Question

    class EmptyClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return SimpleNamespace(text="")  # blocked/empty response

    judge = GeminiJudge(client=EmptyClient())
    q = Question(id="q1", doc_id="d1", question="what?", gold_answer="42",
                 answer_format="str", doc_type="report", evidence_pages=(0,),
                 evidence_sources=("text",), hop="single", is_unanswerable=False)
    pred = Prediction(text="the answer is 42", model_spec="m", total_text_tokens=1,
                      total_visual_tokens=0, text_tokens_fed=1, output_tokens=1,
                      latency_s=0.1, peak_vram_bytes=1)
    score = judge.score(q, pred)  # must not raise
    assert score.judge_spec == "gemini-flash-judge"
    assert score.correct is True  # stub fallback: gold "42" is in the answer text


def test_gemini_judge_recovers_from_rate_limit(monkeypatch):
    """A per-minute 429 across keys must back off and retry, not crash the run."""

    import pipeline.judge as J
    monkeypatch.setattr(J.time, "sleep", lambda *_: None)  # no real backoff wait

    class Quota429(Exception):
        code = 429
        def __str__(self):  # per-minute, NOT per-day
            return "429 RESOURCE_EXHAUSTED PerMinute limit 5"

    class FlakyClient:
        def __init__(self):
            self.n = 0
        class _M:
            def __init__(self, outer): self.outer = outer
            def generate_content(self, model, contents, config):
                self.outer.n += 1
                if self.outer.n <= 3:      # first few calls rate-limited
                    raise Quota429()
                from types import SimpleNamespace
                return SimpleNamespace(text='{"verdict": "correct"}')
        @property
        def models(self): return self._M(self)

    judge = J.GeminiJudge(client=FlakyClient())
    q = J.Question(id="q1", doc_id="d1", question="?", gold_answer="42", answer_format="str",
                   doc_type="report", evidence_pages=(0,), evidence_sources=("text",),
                   hop="single", is_unanswerable=False)
    pred = J.Prediction(text="42", model_spec="m", total_text_tokens=1, total_visual_tokens=0,
                        text_tokens_fed=1, output_tokens=1, latency_s=0.1, peak_vram_bytes=1)
    score = judge.score(q, pred)  # must not raise
    assert score.correct is True
