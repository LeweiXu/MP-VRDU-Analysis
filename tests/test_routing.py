"""Test Section-F6 routing policy aggregation.

Purpose:
    Verifies that Table 7 reports one corpus-level row per routing policy and
    that predicted-routing total latency equals recipe latency plus amortized
    classifier latency.

Test role:
    Uses constructed oracle-ladder rows and classifier side records, so routing
    policy accounting is tested without running Qwen or the classifier.

Arguments:
    None. Run with `python -m pytest tests/test_routing.py`.
"""

from __future__ import annotations

import pytest

from reporting.tables import build_table7_routing
from pipeline.orchestrator import ResultRow


def row(
    *,
    question_id: str,
    doc_id: str,
    doc_type: str,
    representation: str,
    correct: bool,
    latency_s: float,
) -> ResultRow:
    """Return one minimal routing row."""

    return ResultRow(
        cache_key=f"{question_id}-{representation}",
        question_id=question_id,
        doc_id=doc_id,
        doc_type=doc_type,
        hop="single",
        is_unanswerable=False,
        evidence_sources=("Text",),
        condition="oracle",
        provenance="oracle",
        page_indices=(0,),
        representation=representation,
        model_spec="qwen3vl-8b-local",
        judge_spec="stub",
        answer="ok" if correct else "bad",
        input_text_tokens=2,
        input_visual_tokens=2 if representation in {"TLV", "V"} else 0,
        output_tokens=1,
        latency_s=latency_s,
        score=1.0 if correct else 0.0,
        correct=correct,
        abstained=False,
        metadata={"source_dataset": "mmlongbench"},
    )


def ladder(question_id: str, doc_id: str, doc_type: str, frontier: str) -> list[ResultRow]:
    """Return rows whose first correct rung is `frontier`."""

    order = ("T", "TL", "TLV", "V")
    frontier_index = order.index(frontier)
    return [
        row(
            question_id=question_id,
            doc_id=doc_id,
            doc_type=doc_type,
            representation=rung,
            correct=index >= frontier_index,
            latency_s=0.1 * (index + 1),
        )
        for index, rung in enumerate(order)
    ]


def test_routing_outputs_one_corpus_row_per_policy_and_amortizes_classifier_latency() -> None:
    rows = [
        *ladder("q-text", "doc-text", "Academic paper", "T"),
        *ladder("q-vis", "doc-vis", "Brochure", "TLV"),
    ]
    classifier_records = [
        {"doc_id": "doc-text", "predicted_bin": "text_heavy", "latency_s": 0.5},
        {"doc_id": "doc-vis", "predicted_bin": "text_heavy", "latency_s": 0.5},
    ]

    table = build_table7_routing(
        rows,
        margin_points=0.0,
        classifier_records=classifier_records,
        n_bootstrap=0,
    )

    assert set(table["policy"]) == {
        "oracle_routing",
        "predicted_routing",
        "uniform_cheapest_T",
        "uniform_strongest_TLV",
    }
    assert len(table) == 4

    predicted = table.set_index("policy").loc["predicted_routing"]
    assert predicted["classifier_latency_bs1_s"] == pytest.approx(0.5)
    assert predicted["total_latency_bs1_s"] == pytest.approx(
        predicted["latency_bs1_s"] + predicted["classifier_latency_bs1_s"]
    )
