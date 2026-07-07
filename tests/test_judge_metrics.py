"""Test Stage-M5 judge, metrics, frontier, and table builders.

Purpose:
    Verifies the reporting layer that turns cached smoke/full result rows into
    judged accuracy summaries and all eight paper table CSV shapes.

Test role:
    Keeps the real GPT judge path injectable/offline, protects document-level
    bootstrap accounting, checks the sufficiency-frontier rule, and confirms
    every table builder emits the expected skeleton columns from one cache.

Arguments:
    None. Run with `python -m pytest tests/test_judge_metrics.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from reporting.tables import (
    TABLE_FILENAMES,
    build_all_tables,
    build_table6_matched_vs_cross,
    write_all_tables,
)
from metrics.accuracy import accuracy_summary
from metrics.frontier import FrontierCell, sufficiency_frontier
from pipeline.judge import GeminiJudge, GPT4oMiniJudge, get_judge
from pipeline.orchestrator import ResultRow
from schema import Prediction, Question


class FakeCompletions:
    """Tiny OpenAI-compatible completion endpoint for judge tests."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps(self.payload)}}]}


class FakeOpenAIClient:
    """Tiny OpenAI client exposing `chat.completions.create()`."""

    def __init__(self, payload: dict) -> None:
        self.completions = FakeCompletions(payload)
        self.chat = SimpleNamespace(completions=self.completions)


def question() -> Question:
    """Return one answerable toy question."""

    return Question(
        id="q1",
        doc_id="doc-a.pdf",
        question="What is the value?",
        gold_answer="42",
        answer_format="String",
        doc_type="Academic paper",
        evidence_pages=(0,),
        evidence_sources=("Text",),
        hop="single",
        is_unanswerable=False,
    )


def row(
    *,
    question_id: str,
    doc_id: str,
    doc_type: str,
    representation: str,
    correct: bool,
    model_spec: str = "qwen3vl-2b-local",
    evidence_sources: tuple[str, ...] = ("Text",),
    condition: str = "oracle",
) -> ResultRow:
    """Build a minimal cached result row for table tests."""

    return ResultRow(
        cache_key=f"{question_id}-{representation}-{model_spec}-{condition}",
        question_id=question_id,
        doc_id=doc_id,
        doc_type=doc_type,
        hop="single",
        is_unanswerable=False,
        evidence_sources=evidence_sources,
        condition=condition,
        provenance=condition,
        page_indices=(0,),
        representation=representation,
        model_spec=model_spec,
        judge_spec="stub",
        answer="42" if correct else "wrong",
        input_text_tokens=10 if representation != "V" else 0,
        input_visual_tokens=20 if representation in {"TLV", "V"} else 0,
        output_tokens=3,
        latency_s={"T": 0.1, "TL": 0.2, "TLV": 0.4, "V": 0.3}[representation],
        score=1.0 if correct else 0.0,
        correct=correct,
        abstained=False,
        metadata={},
    )


def table_rows() -> list[ResultRow]:
    """Return rows spanning bins, rungs, evidence sources, and model specs."""

    specs = ["qwen3vl-2b-local", "qwen3vl-8b-local"]
    docs = [
        ("q-text", "doc-text.pdf", "Academic paper", ("Text",)),
        ("q-mid", "doc-mid.pdf", "Financial report", ("Table",)),
        ("q-vis", "doc-vis.pdf", "Brochure", ("Chart",)),
    ]
    rows: list[ResultRow] = []
    for model_spec in specs:
        for question_id, doc_id, doc_type, sources in docs:
            for representation in ("T", "TL", "TLV", "V"):
                rows.append(
                    row(
                        question_id=question_id,
                        doc_id=doc_id,
                        doc_type=doc_type,
                        representation=representation,
                        correct=representation in {"TL", "TLV"},
                        model_spec=model_spec,
                        evidence_sources=sources,
                    )
                )
    return rows


def test_gpt4o_mini_judge_returns_valid_score() -> None:
    client = FakeOpenAIClient(
        {"verdict": "correct", "extracted_answer": "42", "rationale": "same value"}
    )
    judge = GPT4oMiniJudge(client=client)

    score = judge.score(question(), Prediction(text="The value is 42."))

    assert score.correct
    assert score.value == 1.0
    assert score.judge_spec == "gpt4o-mini-judge"
    assert score.metadata["verdict"] == "correct"
    assert score.metadata["extracted_answer"] == "42"
    assert client.completions.calls[0]["response_format"] == {"type": "json_object"}


class FakeGeminiModels:
    """Tiny google-genai `client.models` exposing `generate_content()`."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(self.payload))


class FakeGeminiClient:
    """Tiny google-genai client exposing `models.generate_content()`."""

    def __init__(self, payload: dict) -> None:
        self.models = FakeGeminiModels(payload)


def test_gemini_judge_returns_valid_score() -> None:
    client = FakeGeminiClient(
        {"verdict": "incorrect", "extracted_answer": "7", "rationale": "wrong value"}
    )
    judge = GeminiJudge(client=client)

    score = judge.score(question(), Prediction(text="The value is 7."))

    assert not score.correct
    assert score.value == 0.0
    assert score.judge_spec == "gemini-flash-judge"
    assert score.metadata["verdict"] == "incorrect"
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"


class _DailyQuota(Exception):
    """Stand-in for a google-genai per-day 429 (message the judge sniffs for)."""

    def __init__(self) -> None:
        super().__init__(
            "429 RESOURCE_EXHAUSTED quotaId 'GenerateRequestsPerDayPerProjectPerModel' "
            "metric generativelanguage.googleapis.com/generate_requests_per_model_per_day"
        )
        self.code = 429


class FailingGeminiModels:
    """A `models` endpoint whose first N calls raise a daily-quota 429."""

    def __init__(self, payload: dict, fail_times: int) -> None:
        self.payload = payload
        self.remaining_failures = fail_times
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise _DailyQuota()
        return SimpleNamespace(text=json.dumps(self.payload))


def test_gemini_judge_falls_back_to_second_key_on_daily_quota() -> None:
    payload = {"verdict": "correct", "extracted_answer": "42", "rationale": "match"}
    exhausted = SimpleNamespace(models=FailingGeminiModels(payload, fail_times=99))
    fresh = SimpleNamespace(models=FailingGeminiModels(payload, fail_times=0))
    judge = GeminiJudge(client=exhausted)
    judge._clients = [exhausted, fresh]  # two keys; primary is out of daily quota

    # First cell: primary raises daily-quota (no wasted retries), fall back to fresh.
    score = judge.score(question(), Prediction(text="The value is 42."))
    assert score.correct
    assert exhausted.models.calls, "primary key should have been tried once"
    assert fresh.models.calls, "fallback key should have scored the cell"
    assert judge._index == 1  # switch is sticky

    # Second cell goes straight to the fallback key without re-probing the dead one.
    before = len(exhausted.models.calls)
    judge.score(question(), Prediction(text="The value is 42."))
    assert len(exhausted.models.calls) == before
    assert len(fresh.models.calls) == 2


def test_get_judge_resolves_specs(monkeypatch) -> None:
    # Dummy keys so the SDK clients construct without a real credential.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from pipeline.judge import StubJudge

    assert isinstance(get_judge("stub"), StubJudge)
    assert isinstance(get_judge("gemini"), GeminiJudge)
    assert isinstance(get_judge("gpt-4o-mini"), GPT4oMiniJudge)


def test_document_level_accuracy_summary() -> None:
    rows = [
        row(question_id="q1", doc_id="doc-a.pdf", doc_type="Academic paper", representation="T", correct=True),
        row(question_id="q2", doc_id="doc-a.pdf", doc_type="Academic paper", representation="T", correct=False),
        row(question_id="q3", doc_id="doc-b.pdf", doc_type="Brochure", representation="T", correct=True),
    ]

    summary = accuracy_summary(rows, n_bootstrap=0)

    assert summary.n_rows == 3
    assert summary.n_docs == 2
    assert summary.accuracy == 2 / 3
    assert summary.ci_low == summary.accuracy
    assert summary.ci_high == summary.accuracy


def test_frontier_rule_selects_cheapest_sufficient_rung() -> None:
    cells = {
        "T": FrontierCell(accuracy=0.80, ci_high=0.82),
        "TL": FrontierCell(accuracy=0.82, ci_high=0.88),
        "TLV": FrontierCell(accuracy=0.90, ci_high=0.93),
        "V": FrontierCell(accuracy=0.84, ci_high=0.86),
    }

    assert sufficiency_frontier(cells, margin_points=3.0) == "TL"


def test_all_table_builders_emit_csv_shapes(tmp_path: Path) -> None:
    rows = table_rows()

    tables = build_all_tables(rows, n_bootstrap=0)
    paths = write_all_tables(rows, tmp_path, n_bootstrap=0)

    assert set(tables) == set(TABLE_FILENAMES)
    assert set(paths) == set(TABLE_FILENAMES)
    for key, filename in TABLE_FILENAMES.items():
        path = tmp_path / filename
        assert path.exists()
        if key != "table6":
            assert not pd.read_csv(path).empty

    assert {"bin", "frontier", "latency_at_frontier_s"}.issubset(tables["table1"].columns)
    assert {"bin", "question_type", "TLV_acc"}.issubset(tables["table2"].columns)
    assert {"model_spec", "model_size", "frontier"}.issubset(tables["table3"].columns)
    assert {"dataset", "bin", "frontier"}.issubset(tables["table4"].columns)
    assert {"bin", "evidence_modality", "share", "predicted_bin_frontier"}.issubset(tables["table5"].columns)
    assert {"bin", "pipeline", "accuracy", "delta_accuracy_vs_matched"}.issubset(tables["table6"].columns)
    assert {"policy", "chosen_rungs", "accuracy", "total_latency_bs1_s"}.issubset(tables["table7"].columns)
    assert {"scale_family", "model_spec", "frontier"}.issubset(tables["table8"].columns)


def test_table6_needs_oracle_rows_to_find_vision_bins() -> None:
    """Table 6 selects vision-frontier bins from oracle rows, so they must be routed in.

    Regression: table6 used to be sourced from G5 alone, which has no `oracle`
    rows, so vision-bin selection always came up empty and the table was silently
    blank regardless of the retrieval data.
    """

    docs = [("q-v1", "d1.pdf"), ("q-v2", "d2.pdf"), ("q-v3", "d3.pdf")]
    oracle_rows = [
        row(question_id=q, doc_id=d, doc_type="Brochure", representation=rep,
            correct=(rep in {"TLV", "V"}), evidence_sources=("Chart",))
        for q, d in docs for rep in ("T", "TL", "TLV", "V")
    ]
    retrieval_rows = [
        row(question_id=q, doc_id=d, doc_type="Brochure", representation="TLV",
            correct=matched, evidence_sources=("Chart",),
            condition=("retrieved_vision_k1" if matched else "retrieved_text_k1"))
        for q, d in docs for matched in (True, False)
    ]

    # Retrieval rows alone: no oracle rows -> no vision bins -> empty table.
    assert build_table6_matched_vs_cross(retrieval_rows, n_bootstrap=0).empty
    # With the oracle rows routed in, the vision bin is found and both pipelines emit.
    combined = build_table6_matched_vs_cross(oracle_rows + retrieval_rows, n_bootstrap=0)
    assert not combined.empty
    assert set(combined["pipeline"]) == {"matched_vision", "cross_text_to_vision"}
