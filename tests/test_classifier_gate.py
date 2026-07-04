"""Test the Section-F3 classifier feasibility gate.

Purpose:
    Verifies the 100-document pilot sampler, classifier top-1 bin accuracy
    scoring, and the go/no-go threshold that decides whether predicted routing is
    viable for RQ3.

Test role:
    Uses synthetic questions and in-memory classifier records, so the gate is
    covered without rendering PDFs or loading Qwen3-VL.

Arguments:
    None. Run with `python -m pytest tests/test_classifier_gate.py`.
"""

from __future__ import annotations

from experiments.gates import classifier_gate, classifier_pilot_sample, score_classifier_records
from schema import Question


DOC_TYPES = (
    "Academic paper",
    "Financial report",
    "Brochure",
)


def make_question(index: int) -> Question:
    """Return one toy question with a repeating valid MMLongBench doc type."""

    doc_type = DOC_TYPES[index % len(DOC_TYPES)]
    return Question(
        id=f"q{index:03d}",
        doc_id=f"doc-{index:03d}.pdf",
        question=f"classify document {index}",
        gold_answer="answer",
        answer_format="String",
        doc_type=doc_type,
        evidence_pages=(0,),
        evidence_sources=("Text",),
        hop="single",
        is_unanswerable=False,
    )


def test_classifier_gate_threshold() -> None:
    assert classifier_gate(0.70, n_docs=100).passed
    assert not classifier_gate(0.69, n_docs=100).passed


def test_classifier_pilot_sampler_draws_100_distinct_documents() -> None:
    questions = [make_question(index) for index in range(120)]

    sampled = classifier_pilot_sample(questions, n_docs=100, seed=11)

    assert len(sampled) == 100
    assert len({question.doc_id for question in sampled}) == 100
    assert {question.doc_type for question in sampled} == set(DOC_TYPES)


def test_score_classifier_records_uses_bin_accuracy() -> None:
    records = []
    for index in range(10):
        records.append(
            {
                "doc_id": f"doc-{index:03d}.pdf",
                "correct_bin": index < 7,
            }
        )

    result = score_classifier_records(records, threshold=0.70)

    assert result.passed
    assert result.metric == 0.7
    assert result.details["n_docs"] == 10
    assert result.details["correct_docs"] == 7
