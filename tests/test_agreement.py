"""Test the Section-F2 judge-human agreement tooling.

Purpose:
    Covers the Cohen's kappa implementation, the agreement go/no-go threshold,
    the stratified doc_type x question_type sampler, and CSV sheet scoring used
    before main-run numbers are trusted.

Test role:
    Uses small in-memory questions and a temporary completed sheet, so agreement
    tooling is validated without calling GPT-4o-mini or requiring human labels.

Arguments:
    None. Run with `python -m pytest tests/test_agreement.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.gates import (
    agreement_gate,
    cohen_kappa,
    score_agreement_sheet,
    stratified_question_sample,
)
from schema import Question


def make_question(index: int, *, doc_type: str, question_type: str) -> Question:
    """Return one toy question with a raw question_type label."""

    return Question(
        id=f"q{index:03d}",
        doc_id=f"doc-{index:03d}.pdf",
        question=f"question {index}?",
        gold_answer="answer",
        answer_format="String",
        doc_type=doc_type,
        evidence_pages=(0,),
        evidence_sources=("Text",),
        hop="single",
        is_unanswerable=False,
        raw_fields={"question_type": question_type},
    )


def test_cohen_kappa_matches_hand_worked_toy() -> None:
    judge = ["correct", "correct", "incorrect", "incorrect"]
    human = ["correct", "incorrect", "incorrect", "incorrect"]

    assert cohen_kappa(judge, human) == pytest.approx(0.5)


def test_agreement_gate_threshold() -> None:
    assert agreement_gate(0.75).passed
    assert not agreement_gate(0.74).passed


def test_stratified_sampler_covers_every_non_empty_cell() -> None:
    questions = [
        make_question(0, doc_type="Academic paper", question_type="single-hop text"),
        make_question(1, doc_type="Academic paper", question_type="table"),
        make_question(2, doc_type="Brochure", question_type="chart-figure"),
        make_question(3, doc_type="Brochure", question_type="chart-figure"),
        make_question(4, doc_type="Financial report", question_type="multi-hop"),
        make_question(5, doc_type="Financial report", question_type="multi-hop"),
    ]

    sampled = stratified_question_sample(questions, n=5, seed=7)
    cells = {(question.doc_type, question.raw_fields["question_type"]) for question in sampled}

    assert len(sampled) == 5
    assert cells == {
        ("Academic paper", "single-hop text"),
        ("Academic paper", "table"),
        ("Brochure", "chart-figure"),
        ("Financial report", "multi-hop"),
    }


def test_score_agreement_sheet(tmp_path: Path) -> None:
    sheet = tmp_path / "agreement.csv"
    sheet.write_text(
        "question_id,judge_label,human_label\n"
        "q1,correct,correct\n"
        "q2,incorrect,incorrect\n"
        "q3,abstained,abstained\n"
        "q4,correct,incorrect\n"
    )

    result = score_agreement_sheet(sheet, threshold=0.5)

    assert result.passed
    assert result.details["n_items"] == 4
    assert result.details["label_counts"]["human"]["incorrect"] == 2
