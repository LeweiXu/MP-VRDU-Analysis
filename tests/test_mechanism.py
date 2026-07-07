"""Test Section-F5 mechanism tables.

Purpose:
    Covers the evidence-composition mediation and matched-vs-cross aggregation
    required by Stage F5: composition shares sum to one per bin, predicted
    frontiers follow per-modality requirements, and matched/cross retrieval rows
    are well formed for bins whose oracle frontier requires vision.

Test role:
    Uses constructed `ResultRow` caches rather than model calls, so the mechanism
    claims are protected as table logic.

Arguments:
    None. Run with `python -m pytest tests/test_mechanism.py`.
"""

from __future__ import annotations

import pytest

from reporting.tables import (
    build_table5_composition_mediation,
    build_table6_matched_vs_cross,
    predict_frontier_from_composition,
)
from pipeline.orchestrator import ResultRow


def row(
    *,
    question_id: str,
    doc_id: str,
    doc_type: str,
    representation: str,
    correct: bool,
    evidence_sources: tuple[str, ...],
    condition: str = "oracle",
    latency_s: float = 0.1,
) -> ResultRow:
    """Return one minimal mechanism row."""

    return ResultRow(
        cache_key=f"{question_id}-{condition}-{representation}",
        question_id=question_id,
        doc_id=doc_id,
        doc_type=doc_type,
        hop="single",
        is_unanswerable=False,
        evidence_sources=evidence_sources,
        condition=condition,
        provenance="oracle" if condition == "oracle" else "retrieved",
        page_indices=(0,),
        representation=representation,
        model_spec="qwen3vl-8b-local",
        judge_spec="stub",
        answer="ok" if correct else "bad",
        input_text_tokens=1,
        input_visual_tokens=1 if representation in {"TLV", "V"} else 0,
        output_tokens=1,
        latency_s=latency_s,
        score=1.0 if correct else 0.0,
        correct=correct,
        abstained=False,
        metadata={"source_dataset": "mmlongbench"},
    )


def ladder_rows(question_id: str, doc_id: str, doc_type: str, sources: tuple[str, ...]) -> list[ResultRow]:
    """Return oracle ladder rows where vision is the first correct rung."""

    return [
        row(question_id=question_id, doc_id=doc_id, doc_type=doc_type, representation="T", correct=False, evidence_sources=sources),
        row(question_id=question_id, doc_id=doc_id, doc_type=doc_type, representation="TL", correct=False, evidence_sources=sources),
        row(question_id=question_id, doc_id=doc_id, doc_type=doc_type, representation="TLV", correct=True, evidence_sources=sources),
        row(question_id=question_id, doc_id=doc_id, doc_type=doc_type, representation="V", correct=True, evidence_sources=sources),
    ]


def test_composition_shares_sum_to_one_per_bin() -> None:
    rows = []
    rows.extend(ladder_rows("q-text", "d-text", "Academic paper", ("Text",)))
    rows.extend(ladder_rows("q-table", "d-mid", "Financial report", ("Table",)))
    rows.extend(ladder_rows("q-chart", "d-vis", "Brochure", ("Chart",)))

    table = build_table5_composition_mediation(rows, n_bootstrap=0)

    for bin_name, group in table.groupby("bin"):
        assert group["share"].sum() == pytest.approx(1.0), bin_name


def test_predicted_frontier_computation_matches_hand_worked_toy() -> None:
    predicted = predict_frontier_from_composition(
        {"text": 0.80, "table": 0.20},
        {"text": "T", "table": "TLV"},
        min_share=0.10,
    )

    assert predicted == "TLV"


def test_matched_cross_rows_are_well_formed_for_vision_bins() -> None:
    rows = ladder_rows("q-vis", "d-vis", "Brochure", ("Chart",))
    rows.append(row(question_id="q-vis", doc_id="d-vis", doc_type="Brochure", representation="TLV", correct=True, evidence_sources=("Chart",), condition="retrieved_vision_k1", latency_s=0.3))
    rows.append(row(question_id="q-vis", doc_id="d-vis", doc_type="Brochure", representation="TLV", correct=False, evidence_sources=("Chart",), condition="retrieved_text_k1", latency_s=0.2))
    retrieval_records = [
        {"doc_id": "d-vis", "modality": "vision", "precision": 1.0, "recall": 1.0, "f1": 1.0},
        {"doc_id": "d-vis", "modality": "text", "precision": 0.0, "recall": 0.0, "f1": 0.0},
    ]

    table = build_table6_matched_vs_cross(
        rows,
        margin_points=0.0,
        retrieval_records=retrieval_records,
        n_bootstrap=0,
    )

    assert set(table["pipeline"]) == {"matched_vision", "cross_text_to_vision"}
    cross = table.set_index("pipeline").loc["cross_text_to_vision"]
    assert cross["delta_accuracy_vs_matched"] == -1.0
    assert cross["retrieval_f1"] == 0.0


def test_table6_reports_each_swept_k_separately() -> None:
    rows = ladder_rows("q-vis", "d-vis", "Brochure", ("Chart",))
    # vision retrieval improves with k; text retrieval stays wrong. Two k values.
    for k, vision_ok in ((1, False), (3, True)):
        rows.append(row(question_id="q-vis", doc_id="d-vis", doc_type="Brochure", representation="TLV",
                        correct=vision_ok, evidence_sources=("Chart",), condition=f"retrieved_vision_k{k}"))
        rows.append(row(question_id="q-vis", doc_id="d-vis", doc_type="Brochure", representation="TLV",
                        correct=False, evidence_sources=("Chart",), condition=f"retrieved_text_k{k}"))

    table = build_table6_matched_vs_cross(rows, margin_points=0.0, n_bootstrap=0)

    assert set(table["k"]) == {1, 3}
    matched = table[(table["pipeline"] == "matched_vision")].set_index("k")["accuracy"]
    assert matched.loc[1] == 0.0 and matched.loc[3] == 1.0  # per-k, not pooled
